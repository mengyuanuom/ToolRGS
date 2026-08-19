"""DARG: DETRIS-based Asymmetric Rotated Grasping."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from toolrgs.structures import GraspModelResult, GraspOutput, GraspTargets

from .drog import DROG


class DARGProjector(nn.Module):
    """Language-conditioned mask-guided head for asymmetric grasp geometry."""

    def __init__(
        self,
        word_dim=512,
        in_dim=256,
        hidden_dim=256,
        mask_guidance_strength=0.5,
    ):
        super().__init__()
        self.mask_guidance_strength = float(mask_guidance_strength)
        self.visual = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_dim * 2, hidden_dim * 2, 3, padding=1, bias=False),
            nn.GroupNorm(16, hidden_dim * 2),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(16, hidden_dim),
            nn.GELU(),
        )
        self.language_affine = nn.Sequential(
            nn.Linear(word_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        self.segmentation = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.geometry = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(16, hidden_dim),
            nn.GELU(),
        )
        self.quality = nn.Conv2d(hidden_dim, 1, 1)
        self.centerness = nn.Conv2d(hidden_dim, 1, 1)
        self.angle = nn.Conv2d(hidden_dim, 2, 1)
        self.ltrb = nn.Conv2d(hidden_dim, 4, 1)

        # Start with short, positive distances instead of softplus(0)=0.693.
        nn.init.constant_(self.ltrb.bias, -2.0)

    def forward(self, features, text_state):
        features = self.visual(features)
        gamma, beta = self.language_affine(text_state).chunk(2, dim=1)
        gamma = torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        conditioned = features * (1.0 + gamma) + beta

        segmentation = self.segmentation(conditioned)
        language_mask = torch.sigmoid(segmentation)
        guided = conditioned * (
            1.0 + self.mask_guidance_strength * language_mask
        )
        guided = self.geometry(guided)
        return {
            "segmentation": segmentation,
            "quality": self.quality(guided),
            "centerness": self.centerness(guided),
            "angle": self.angle(guided),
            "ltrb": F.softplus(self.ltrb(guided)),
        }


def decode_asymmetric_geometry(ltrb, sine, cosine, size_factor, offset_radius):
    """Convert local side distances into legacy width/short-side/offset maps."""
    left, right, top, bottom = ltrb.chunk(4, dim=1)
    theta = 0.5 * torch.atan2(sine, cosine)
    delta_long = 0.5 * (right - left) * float(size_factor)
    delta_short = 0.5 * (bottom - top) * float(size_factor)
    offset_x = delta_long * torch.cos(theta) - delta_short * torch.sin(theta)
    offset_y = delta_long * torch.sin(theta) + delta_short * torch.cos(theta)
    offset = torch.cat([offset_x, offset_y], dim=1) / float(offset_radius)
    return left + right, top + bottom, offset


def _rotated_gaussian(ltrb, sine, cosine, epsilon=1e-6):
    left, right, top, bottom = ltrb.chunk(4, dim=1)
    theta = 0.5 * torch.atan2(sine, cosine)
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)

    local_x = 0.5 * (right - left)
    local_y = 0.5 * (bottom - top)
    center_x = local_x * cos_theta - local_y * sin_theta
    center_y = local_x * sin_theta + local_y * cos_theta

    variance_x = (0.5 * (left + right)).square() + epsilon
    variance_y = (0.5 * (top + bottom)).square() + epsilon
    covariance_xx = (
        variance_x * cos_theta.square() + variance_y * sin_theta.square()
    )
    covariance_yy = (
        variance_x * sin_theta.square() + variance_y * cos_theta.square()
    )
    covariance_xy = (
        (variance_x - variance_y) * cos_theta * sin_theta
    )
    return center_x, center_y, covariance_xx, covariance_xy, covariance_yy


def rotated_gwd_kld(pred_ltrb, pred_sine, pred_cosine,
                    target_ltrb, target_sine, target_cosine):
    """Return stable per-pixel GWD and symmetric-KLD rotated-box losses."""
    # Geometry inverses and determinants are deliberately evaluated in fp32.
    # This function runs inside AMP during training; fp16 determinants for thin
    # grasp rectangles otherwise underflow and are a common source of NaNs.
    pred_ltrb = pred_ltrb.float()
    pred_sine = pred_sine.float()
    pred_cosine = pred_cosine.float()
    target_ltrb = target_ltrb.float()
    target_sine = target_sine.float()
    target_cosine = target_cosine.float()
    pred = _rotated_gaussian(pred_ltrb, pred_sine, pred_cosine)
    target = _rotated_gaussian(target_ltrb, target_sine, target_cosine)
    p_x, p_y, p_xx, p_xy, p_yy = pred
    t_x, t_y, t_xx, t_xy, t_yy = target

    center_distance = (p_x - t_x).square() + (p_y - t_y).square()
    determinant_p = (p_xx * p_yy - p_xy.square()).clamp_min(1e-12)
    determinant_t = (t_xx * t_yy - t_xy.square()).clamp_min(1e-12)
    trace_product = p_xx * t_xx + 2.0 * p_xy * t_xy + p_yy * t_yy
    covariance_distance = (
        p_xx + p_yy + t_xx + t_yy
        - 2.0 * torch.sqrt(
            (trace_product + 2.0 * torch.sqrt(determinant_p * determinant_t))
            .clamp_min(0.0)
        )
    ).clamp_min(0.0)
    gwd_distance = center_distance + covariance_distance
    gwd = 1.0 - torch.exp(-torch.sqrt(gwd_distance.clamp_min(0.0)))

    delta_x, delta_y = p_x - t_x, p_y - t_y
    trace_p_to_t = (
        t_yy * p_xx - 2.0 * t_xy * p_xy + t_xx * p_yy
    ) / determinant_t
    center_p_to_t = (
        t_yy * delta_x.square()
        - 2.0 * t_xy * delta_x * delta_y
        + t_xx * delta_y.square()
    ) / determinant_t
    trace_t_to_p = (
        p_yy * t_xx - 2.0 * p_xy * t_xy + p_xx * t_yy
    ) / determinant_p
    center_t_to_p = (
        p_yy * delta_x.square()
        - 2.0 * p_xy * delta_x * delta_y
        + p_xx * delta_y.square()
    ) / determinant_p
    symmetric_kld = 0.25 * (
        trace_p_to_t + center_p_to_t + trace_t_to_p + center_t_to_p - 4.0
    )
    kld = 1.0 - torch.exp(-symmetric_kld.clamp_min(0.0))
    return gwd, kld


def _masked_mean(values, weight):
    expanded = weight.expand_as(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)


class DARG(DROG):
    """DETRIS language features plus CenterNet/rotated-FCOS grasp geometry."""

    supports_offset = True
    predicts_grasp_short_side = True
    uses_asymmetric_geometry = True
    grasp_size_loss_activation = "clamp"

    def __init__(self, cfg):
        super().__init__(cfg)
        if not self.use_grasp_masks:
            raise ValueError("DARG requires use_grasp_masks=True")
        hidden_dim = int(getattr(cfg, "darg_head_dim", cfg.vis_dim // 2))
        self.proj = DARGProjector(
            word_dim=cfg.word_dim,
            in_dim=cfg.vis_dim // 2,
            hidden_dim=hidden_dim,
            mask_guidance_strength=float(
                getattr(cfg, "mask_guidance_strength", 0.5)
            ),
        )
        self.size_factor = float(getattr(cfg, "grasp_size_factor", 100.0))
        self.offset_radius = float(getattr(cfg, "offset_r", 20.0))
        if self.size_factor <= 0 or self.offset_radius <= 0:
            raise ValueError("DARG grasp_size_factor and offset_r must be positive")

        self.geometry_loss = str(
            getattr(cfg, "geometry_loss", "hybrid")
        ).lower()
        if self.geometry_loss not in {"gwd", "kld", "hybrid"}:
            raise ValueError("DARG geometry_loss must be gwd, kld, or hybrid")
        self.loss_weights = {
            "seg": float(getattr(cfg, "seg_loss_weight", 1.0)),
            "quality": float(getattr(cfg, "quality_loss_weight", 1.0)),
            "center": float(getattr(cfg, "centerness_loss_weight", 1.0)),
            "ltrb": float(getattr(cfg, "ltrb_loss_weight", 2.0)),
            "angle": float(getattr(cfg, "angle_loss_weight", 1.0)),
            "unit": float(getattr(cfg, "angle_unit_loss_weight", 0.1)),
            "gwd": float(getattr(cfg, "gwd_loss_weight", 1.0)),
            "kld": float(getattr(cfg, "kld_loss_weight", 0.5)),
        }

    @staticmethod
    def _resize(target, output_size, mode="nearest"):
        if target is None:
            return None
        if target.shape[-2:] == output_size:
            return target.detach()
        kwargs = {"size": output_size, "mode": mode}
        if mode in {"bilinear", "bicubic"}:
            kwargs["align_corners"] = False
        return F.interpolate(target, **kwargs).detach()

    def _predictions(self, raw):
        angle = raw["angle"]
        norm = torch.linalg.vector_norm(angle, dim=1, keepdim=True).clamp_min(1e-6)
        sine, cosine = (angle / norm).chunk(2, dim=1)
        width, short_side, offset = decode_asymmetric_geometry(
            raw["ltrb"], sine, cosine, self.size_factor, self.offset_radius
        )
        combined_quality = (
            torch.sigmoid(raw["quality"]) * torch.sigmoid(raw["centerness"])
        ).clamp(1e-6, 1.0 - 1e-6)
        quality_logits = torch.logit(combined_quality)
        return GraspOutput(
            segmentation=raw["segmentation"],
            quality=quality_logits,
            sine=sine,
            cosine=cosine,
            width=width,
            short_side=short_side,
            offset=offset,
        ), sine, cosine

    def forward(
        self,
        img,
        word,
        mask=None,
        grasp_qua_mask=None,
        grasp_sin_mask=None,
        grasp_cos_mask=None,
        grasp_wid_mask=None,
        grasp_off_mask=None,
        grasp_off_weight=None,
        grasp_short_mask=None,
        grasp_ltrb_mask=None,
        grasp_centerness_mask=None,
        grasp_geometry_weight=None,
        grasp_geometry_sin_mask=None,
        grasp_geometry_cos_mask=None,
    ):
        del grasp_off_mask, grasp_off_weight, grasp_sin_mask, grasp_cos_mask
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()
        visual, words, state = self.fusion(
            img, word, self.txt_backbone, self.dinov2
        )
        features = self.neck(visual, state)
        batch, channels, height, width = features.shape
        features = self.decoder(features, words, pad_mask).reshape(
            batch, channels, height, width
        )
        raw = self.proj(features, state)
        predictions, pred_sine, pred_cosine = self._predictions(raw)

        if not self.training or mask is None:
            return GraspModelResult(predictions=predictions.detach())

        required = {
            "grasp_qua_mask": grasp_qua_mask,
            "grasp_ltrb_mask": grasp_ltrb_mask,
            "grasp_centerness_mask": grasp_centerness_mask,
            "grasp_geometry_weight": grasp_geometry_weight,
            "grasp_geometry_sin_mask": grasp_geometry_sin_mask,
            "grasp_geometry_cos_mask": grasp_geometry_cos_mask,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "DARG training requires asymmetric targets: " + ", ".join(missing)
            )

        output_size = raw["segmentation"].shape[-2:]
        mask = self._resize(mask, output_size)
        target_quality = self._resize(grasp_qua_mask, output_size)
        target_ltrb = self._resize(grasp_ltrb_mask, output_size)
        target_center = self._resize(grasp_centerness_mask, output_size)
        geometry_weight = self._resize(grasp_geometry_weight, output_size)
        target_sine = self._resize(grasp_geometry_sin_mask, output_size)
        target_cosine = self._resize(grasp_geometry_cos_mask, output_size)

        seg_weight = 1.0 + 0.5 * mask
        seg_loss = F.binary_cross_entropy_with_logits(
            raw["segmentation"], mask, weight=seg_weight
        )
        quality_loss = F.binary_cross_entropy_with_logits(
            raw["quality"], target_quality
        )
        center_weight = 1.0 + 4.0 * target_center
        center_loss = F.binary_cross_entropy_with_logits(
            raw["centerness"], target_center, weight=center_weight
        )
        ltrb_error = F.smooth_l1_loss(
            raw["ltrb"], target_ltrb, reduction="none"
        )
        ltrb_loss = _masked_mean(ltrb_error, geometry_weight)
        angle_error = (
            F.smooth_l1_loss(pred_sine, target_sine, reduction="none")
            + F.smooth_l1_loss(pred_cosine, target_cosine, reduction="none")
        )
        angle_loss = _masked_mean(angle_error, geometry_weight)
        angle_norm = torch.linalg.vector_norm(raw["angle"], dim=1, keepdim=True)
        unit_loss = _masked_mean((angle_norm - 1.0).square(), geometry_weight)

        gwd_map, kld_map = rotated_gwd_kld(
            raw["ltrb"], pred_sine, pred_cosine,
            target_ltrb, target_sine, target_cosine,
        )
        gwd_loss = _masked_mean(gwd_map, geometry_weight)
        kld_loss = _masked_mean(kld_map, geometry_weight)
        geometry_loss = 0.0
        if self.geometry_loss in {"gwd", "hybrid"}:
            geometry_loss = geometry_loss + self.loss_weights["gwd"] * gwd_loss
        if self.geometry_loss in {"kld", "hybrid"}:
            geometry_loss = geometry_loss + self.loss_weights["kld"] * kld_loss

        total_loss = (
            self.loss_weights["seg"] * seg_loss
            + self.loss_weights["quality"] * quality_loss
            + self.loss_weights["center"] * center_loss
            + self.loss_weights["ltrb"] * ltrb_loss
            + self.loss_weights["angle"] * angle_loss
            + self.loss_weights["unit"] * unit_loss
            + geometry_loss
        )

        target_width, target_short, target_offset = decode_asymmetric_geometry(
            target_ltrb, target_sine, target_cosine,
            self.size_factor, self.offset_radius,
        )
        targets = GraspTargets(
            segmentation=mask,
            quality=target_quality,
            sine=target_sine,
            cosine=target_cosine,
            width=target_width,
            short_side=target_short,
            offset=target_offset,
        )
        losses = {
            "m_ins": seg_loss.detach(),
            "m_qua": quality_loss.detach(),
            "m_sin": angle_loss.detach(),
            "m_cos": unit_loss.detach(),
            "m_wid": ltrb_loss.detach(),
            "m_short": ltrb_loss.detach(),
            "m_off": geometry_loss.detach(),
            "m_center": center_loss.detach(),
            "m_ltrb": ltrb_loss.detach(),
            "m_gwd": gwd_loss.detach(),
            "m_kld": kld_loss.detach(),
        }
        return GraspModelResult(
            predictions=predictions.detach(),
            targets=targets,
            loss=total_loss,
            losses=losses,
        )

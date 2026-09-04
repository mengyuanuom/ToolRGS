"""DROG-LR: DROG with asymmetric left/right grasp-width regression."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from toolrgs.structures import GraspModelResult, GraspOutput, GraspTargets

from .drog import DROG


class DROGLRProjector(nn.Module):
    """Language-conditioned head with selectable left/right parameterization."""

    def __init__(
        self,
        word_dim=512,
        in_dim=256,
        hidden_dim=256,
        mask_guidance_strength=0.5,
        parameterization="total_fraction",
        use_centerness=True,
    ):
        super().__init__()
        self.mask_guidance_strength = float(mask_guidance_strength)
        self.use_centerness = bool(use_centerness)
        self.parameterization = str(parameterization).lower()
        if self.parameterization not in {"total_fraction", "direct"}:
            raise ValueError(
                "DROG-LR parameterization must be total_fraction or direct"
            )
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
        if self.use_centerness:
            self.centerness = nn.Conv2d(hidden_dim, 1, 1)
        self.angle = nn.Conv2d(hidden_dim, 2, 1)
        if self.parameterization == "direct":
            self.left_width = nn.Conv2d(hidden_dim, 1, 1)
            self.right_width = nn.Conv2d(hidden_dim, 1, 1)
            # sigmoid(-2) * 300 ~= 36 px per side.
            nn.init.constant_(self.left_width.bias, -2.0)
            nn.init.constant_(self.right_width.bias, -2.0)
        else:
            self.total_width = nn.Conv2d(hidden_dim, 1, 1)
            self.left_fraction = nn.Conv2d(hidden_dim, 1, 1)
            # A conservative initial total width and a symmetric 50/50 split.
            nn.init.constant_(self.total_width.bias, -1.0)
            nn.init.constant_(self.left_fraction.bias, 0.0)

    def forward(self, features, text_state):
        features = self.visual(features)
        gamma, beta = self.language_affine(text_state).chunk(2, dim=1)
        gamma = torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        conditioned = features * (1.0 + gamma) + beta

        segmentation = self.segmentation(conditioned)
        guided = conditioned * (
            1.0
            + self.mask_guidance_strength * torch.sigmoid(segmentation)
        )
        guided = self.geometry(guided)
        output = {
            "segmentation": segmentation,
            "quality": self.quality(guided),
            "angle": self.angle(guided),
        }
        if self.use_centerness:
            output["centerness"] = self.centerness(guided)
        if self.parameterization == "direct":
            output["left_width"] = self.left_width(guided)
            output["right_width"] = self.right_width(guided)
        else:
            output["total_width"] = self.total_width(guided)
            output["left_fraction"] = self.left_fraction(guided)
        return output


def decode_lr_geometry(
    total_width_logits,
    left_fraction_logits,
    sine,
    cosine,
    offset_size_factor,
    offset_radius,
):
    """Decode bounded left/right widths and their induced center correction.

    Widths are normalized in the configured original-image coordinate system.
    ``offset_size_factor`` converts that normalized displacement to network-input
    pixels before the standard radius-normalized offset decoder is applied.
    """
    width = torch.sigmoid(total_width_logits)
    left_fraction = torch.sigmoid(left_fraction_logits)
    left = width * left_fraction
    right = width * (1.0 - left_fraction)
    theta = 0.5 * torch.atan2(sine, cosine)
    delta_long = 0.5 * (right - left) * float(offset_size_factor)
    offset_x = delta_long * torch.cos(theta)
    offset_y = delta_long * torch.sin(theta)
    offset = torch.cat([offset_x, offset_y], dim=1) / float(offset_radius)
    return width, left, right, offset


def decode_direct_lr_geometry(
    left_width_logits,
    right_width_logits,
    sine,
    cosine,
    offset_size_factor,
    offset_radius,
    max_total_normalized=1.0,
):
    """Decode independent sides and proportionally cap their total width.

    The raw side predictions remain available for direct supervision.  The
    returned decoded sides preserve their ratio while guaranteeing that their
    sum is no larger than ``max_total_normalized``.  With the unified V3
    ``grasp_size_factor=300`` contract, a limit of 1.0 is exactly 300 original
    image pixels.
    """
    raw_left = torch.sigmoid(left_width_logits)
    raw_right = torch.sigmoid(right_width_logits)
    raw_total = raw_left + raw_right
    limit = torch.as_tensor(
        max_total_normalized,
        dtype=raw_total.dtype,
        device=raw_total.device,
    )
    scale = torch.clamp(limit / raw_total.clamp_min(1e-6), max=1.0)
    left = raw_left * scale
    right = raw_right * scale
    width = left + right
    theta = 0.5 * torch.atan2(sine, cosine)
    delta_long = 0.5 * (right - left) * float(offset_size_factor)
    offset_x = delta_long * torch.cos(theta)
    offset_y = delta_long * torch.sin(theta)
    offset = torch.cat([offset_x, offset_y], dim=1) / float(offset_radius)
    return width, left, right, offset, raw_left, raw_right


def _masked_mean(values, weight):
    expanded = weight.expand_as(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)


def _balanced_probability_loss(logits, target, positive_threshold):
    """Give grasp-positive and background pixels equal aggregate influence."""
    error = F.smooth_l1_loss(torch.sigmoid(logits), target, reduction="none")
    positive = target > float(positive_threshold)
    negative = ~positive
    terms = []
    if positive.any():
        terms.append(error[positive].mean())
    if negative.any():
        terms.append(error[negative].mean())
    return sum(terms) / max(1, len(terms))


def combine_quality_logits(quality_logits, centerness_logits=None):
    """Return inference logits with optional centerness score modulation."""
    if centerness_logits is None:
        return quality_logits
    combined = (
        torch.sigmoid(quality_logits) * torch.sigmoid(centerness_logits)
    ).clamp(1e-6, 1.0 - 1e-6)
    return torch.logit(combined)


class DROGLR(DROG):
    """DROG backbone with two-sided long-axis geometry and center correction."""

    supports_offset = True
    predicts_grasp_short_side = False
    uses_asymmetric_geometry = True

    def __init__(self, cfg):
        super().__init__(cfg)
        if not self.use_grasp_masks:
            raise ValueError("DROG-LR requires use_grasp_masks=True")
        hidden_dim = int(getattr(cfg, "droglr_head_dim", cfg.vis_dim // 2))
        self.lr_parameterization = str(
            getattr(cfg, "droglr_parameterization", "total_fraction")
        ).lower()
        self.use_centerness = bool(
            getattr(cfg, "droglr_use_centerness", True)
        )
        self.proj = DROGLRProjector(
            word_dim=cfg.word_dim,
            in_dim=cfg.vis_dim // 2,
            hidden_dim=hidden_dim,
            mask_guidance_strength=float(
                getattr(cfg, "mask_guidance_strength", 0.5)
            ),
            parameterization=self.lr_parameterization,
            use_centerness=self.use_centerness,
        )
        self.size_factor = float(getattr(cfg, "grasp_size_factor", 300.0))
        self.max_total_width = float(
            getattr(cfg, "droglr_max_total_width", self.size_factor)
        )
        self.max_total_normalized = self.max_total_width / self.size_factor
        self.offset_size_factor = float(
            getattr(cfg, "droglr_offset_size_factor", cfg.input_size)
        )
        self.offset_radius = float(getattr(cfg, "offset_r", 20.0))
        if min(
            self.size_factor,
            self.max_total_width,
            self.offset_size_factor,
            self.offset_radius,
        ) <= 0:
            raise ValueError("DROG-LR size and offset factors must be positive")
        self.quality_positive_threshold = float(
            getattr(cfg, "droglr_quality_positive_threshold", 0.05)
        )
        self.loss_weights = {
            "seg": float(getattr(cfg, "seg_loss_weight", 1.0)),
            "quality": float(getattr(cfg, "quality_loss_weight", 1.0)),
            "center": float(getattr(cfg, "centerness_loss_weight", 0.5)),
            "sides": float(getattr(cfg, "lr_width_loss_weight", 2.0)),
            "angle": float(getattr(cfg, "angle_loss_weight", 1.0)),
            "unit": float(getattr(cfg, "angle_unit_loss_weight", 0.1)),
        }

        # The head itself defines these train/decode contracts.
        self.grasp_quality_train_activation = "sigmoid"
        self.grasp_quality_loss_activation = "sigmoid"
        self.grasp_quality_decode_activation = "sigmoid"
        self.grasp_width_loss_activation = "sigmoid"
        self.grasp_size_loss_activation = "sigmoid"
        self.grasp_size_decode_activation = "sigmoid"

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

    def _decode(self, raw):
        angle_norm = torch.linalg.vector_norm(
            raw["angle"], dim=1, keepdim=True
        ).clamp_min(1e-6)
        sine, cosine = (raw["angle"] / angle_norm).chunk(2, dim=1)
        if self.lr_parameterization == "direct":
            width, left, right, offset, loss_left, loss_right = (
                decode_direct_lr_geometry(
                    raw["left_width"],
                    raw["right_width"],
                    sine,
                    cosine,
                    self.offset_size_factor,
                    self.offset_radius,
                    self.max_total_normalized,
                )
            )
            width_logits = torch.logit(width.clamp(1e-6, 1.0 - 1e-6))
        else:
            width, left, right, offset = decode_lr_geometry(
                raw["total_width"],
                raw["left_fraction"],
                sine,
                cosine,
                self.offset_size_factor,
                self.offset_radius,
            )
            loss_left, loss_right = left, right
            width_logits = raw["total_width"]
        quality_logits = combine_quality_logits(
            raw["quality"], raw.get("centerness")
        )
        predictions = GraspOutput(
            segmentation=raw["segmentation"],
            quality=quality_logits,
            sine=sine,
            cosine=cosine,
            width=width_logits,
            offset=offset,
        )
        return (
            predictions,
            sine,
            cosine,
            width,
            left,
            right,
            loss_left,
            loss_right,
        )

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
        del (
            grasp_sin_mask,
            grasp_cos_mask,
            grasp_wid_mask,
            grasp_off_mask,
            grasp_off_weight,
            grasp_short_mask,
        )
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
        (
            predictions,
            pred_sine,
            pred_cosine,
            pred_width,
            pred_left,
            pred_right,
            loss_left,
            loss_right,
        ) = self._decode(raw)

        if not self.training or mask is None:
            return GraspModelResult(predictions=predictions.detach())

        required = {
            "grasp_qua_mask": grasp_qua_mask,
            "grasp_ltrb_mask": grasp_ltrb_mask,
            "grasp_geometry_weight": grasp_geometry_weight,
            "grasp_geometry_sin_mask": grasp_geometry_sin_mask,
            "grasp_geometry_cos_mask": grasp_geometry_cos_mask,
        }
        if self.use_centerness:
            required["grasp_centerness_mask"] = grasp_centerness_mask
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "DROG-LR training requires asymmetric targets: "
                + ", ".join(missing)
            )

        output_size = raw["segmentation"].shape[-2:]
        mask = self._resize(mask, output_size)
        target_quality = self._resize(grasp_qua_mask, output_size)
        target_ltrb = self._resize(grasp_ltrb_mask, output_size)
        target_center = (
            self._resize(grasp_centerness_mask, output_size)
            if self.use_centerness
            else None
        )
        geometry_weight = self._resize(grasp_geometry_weight, output_size)
        target_sine = self._resize(grasp_geometry_sin_mask, output_size)
        target_cosine = self._resize(grasp_geometry_cos_mask, output_size)
        target_left, target_right = target_ltrb[:, 0:1], target_ltrb[:, 1:2]
        target_width = (target_left + target_right).clamp(0.0, 1.0)

        seg_loss = F.binary_cross_entropy_with_logits(
            raw["segmentation"], mask, weight=1.0 + 0.5 * mask
        )
        quality_loss = _balanced_probability_loss(
            raw["quality"], target_quality, self.quality_positive_threshold
        )
        if self.use_centerness:
            center_loss = F.binary_cross_entropy_with_logits(
                raw["centerness"],
                target_center,
                weight=1.0 + 4.0 * target_center,
            )
        else:
            center_loss = raw["quality"].new_zeros(())
        side_error = F.smooth_l1_loss(
            torch.cat([loss_left, loss_right], dim=1),
            torch.cat([target_left, target_right], dim=1),
            reduction="none",
        )
        side_loss = _masked_mean(side_error, geometry_weight)
        angle_error = (
            F.smooth_l1_loss(pred_sine, target_sine, reduction="none")
            + F.smooth_l1_loss(pred_cosine, target_cosine, reduction="none")
        )
        angle_loss = _masked_mean(angle_error, geometry_weight)
        raw_angle_norm = torch.linalg.vector_norm(
            raw["angle"], dim=1, keepdim=True
        )
        unit_loss = _masked_mean(
            (raw_angle_norm - 1.0).square(), geometry_weight
        )

        total_loss = (
            self.loss_weights["seg"] * seg_loss
            + self.loss_weights["quality"] * quality_loss
            + self.loss_weights["center"] * center_loss
            + self.loss_weights["sides"] * side_loss
            + self.loss_weights["angle"] * angle_loss
            + self.loss_weights["unit"] * unit_loss
        )

        _width, _left, _right, target_offset = decode_lr_geometry(
            torch.logit(target_width.clamp(1e-6, 1.0 - 1e-6)),
            torch.logit(
                (target_left / target_width.clamp_min(1e-6)).clamp(
                    1e-6, 1.0 - 1e-6
                )
            ),
            target_sine,
            target_cosine,
            self.offset_size_factor,
            self.offset_radius,
        )
        targets = GraspTargets(
            segmentation=mask,
            quality=target_quality,
            sine=target_sine,
            cosine=target_cosine,
            width=target_width,
            offset=target_offset,
        )
        losses = {
            "m_ins": seg_loss.detach(),
            "m_qua": quality_loss.detach(),
            "m_sin": angle_loss.detach(),
            "m_cos": unit_loss.detach(),
            "m_wid": side_loss.detach(),
            "m_off": side_loss.detach(),
            "m_center": center_loss.detach(),
            "m_lr": side_loss.detach(),
        }
        return GraspModelResult(
            predictions=predictions.detach(),
            targets=targets,
            loss=total_loss,
            losses=losses,
        )

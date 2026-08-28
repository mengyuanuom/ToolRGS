import torch
import torch.nn as nn
import torch.nn.functional as F
from .crog_clip import build_model
from .clip_grasp_fusion import TextVisualFusionFiLM


def balanced_quality_bce_with_logits(
    prediction: torch.Tensor,
    target: torch.Tensor,
    max_pos_weight: float = 32.0,
) -> torch.Tensor:
    """Train a sparse grasp-quality map as logits without background collapse."""
    target = target.to(dtype=prediction.dtype).clamp(0.0, 1.0)
    with torch.no_grad():
        positive_mass = target.sum().clamp_min(1.0)
        negative_mass = (1.0 - target).sum()
        pos_weight = (negative_mass / positive_mass).clamp(
            min=1.0, max=float(max_pos_weight)
        )
    return F.binary_cross_entropy_with_logits(
        prediction,
        target,
        pos_weight=pos_weight,
    )

filter_sizes = [32, 16, 8, 8, 16, 32]
kernel_sizes = [9, 5, 3, 3, 5, 9]
strides = [3, 2, 2, 2, 2, 3]


class GGCNNWithText(nn.Module):
    """
    GG-CNN with text conditioning (e.g. CLIP text embedding).

    This class extends the original GG-CNN by:
      - Adding a FiLM fusion module that conditions the final feature map on text.
      - Modifying forward(...) to take an extra text embedding e_txt.
      - Adding a compute_loss(..., e_txt) that matches the style of GraspModel.
    """

    def __init__(self, input_channels: int = 4, text_dim: int = 512,
                 dropout: bool = False, prob: float = 0.0,
                 max_quality_pos_weight: float = 32.0):
        """
        Args:
            input_channels: Number of input channels (e.g. 4 for RGB-D).
            text_dim:       Dimension of the text embedding (e.g. CLIP text dim).
            dropout:        Whether to apply output-head dropout.
            prob:           Dropout probability when enabled.
        """
        super().__init__()
        self.max_quality_pos_weight = float(max_quality_pos_weight)

        # Encoder
        self.conv1 = nn.Conv2d(
            input_channels, filter_sizes[0],
            kernel_sizes[0], stride=strides[0], padding=3
        )
        self.conv2 = nn.Conv2d(
            filter_sizes[0], filter_sizes[1],
            kernel_sizes[1], stride=strides[1], padding=2
        )
        self.conv3 = nn.Conv2d(
            filter_sizes[1], filter_sizes[2],
            kernel_sizes[2], stride=strides[2], padding=1
        )

        # Decoder
        self.convt1 = nn.ConvTranspose2d(
            filter_sizes[2], filter_sizes[3],
            kernel_sizes[3], stride=strides[3],
            padding=1, output_padding=1
        )
        self.convt2 = nn.ConvTranspose2d(
            filter_sizes[3], filter_sizes[4],
            kernel_sizes[4], stride=strides[4],
            padding=2, output_padding=1
        )
        self.convt3 = nn.ConvTranspose2d(
            filter_sizes[4], filter_sizes[5],
            kernel_sizes[5], stride=strides[5],
            padding=5, output_padding=1
        )

        # Match GRConvNet-CLIP: fuse text at the encoder bottleneck, then let
        # the untouched GG-CNN decoder reconstruct the spatial prediction.
        self.fusion = TextVisualFusionFiLM(
            vis_dim=filter_sizes[2],
            text_dim=text_dim,
            hidden_dim=128
        )

        # Output heads: quality, cos, sin, width
        self.pos_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.cos_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.sin_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.width_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.dropout1 = nn.Dropout(p=prob if dropout else 0.0)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1.0)

    def forward(self, x_in, e_txt):
        """
        Forward pass.

        Args:
            x_in:  Tensor of shape (B, input_channels, H, W) - input image.
            e_txt: Tensor of shape (B, text_dim)             - text embedding.

        Returns:
            pos_output, cos_output, sin_output, width_output
            each of shape (B, 1, H_out, W_out)
        """
        x = F.relu(self.conv1(x_in))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Use the same shared bottleneck FiLM module and placement as
        # GRConvNet-CLIP. The GG-CNN encoder/decoder layers remain original.
        x = self.fusion(x, e_txt)

        x = F.relu(self.convt1(x))
        x = F.relu(self.convt2(x))
        x = F.relu(self.convt3(x))

        pos_output = self.pos_output(self.dropout1(x))
        cos_output = self.cos_output(self.dropout1(x))
        sin_output = self.sin_output(self.dropout1(x))
        width_output = self.width_output(self.dropout1(x))

        return pos_output, cos_output, sin_output, width_output

    def compute_loss(self, xc, yc, e_txt):
        """
        Compute regression losses for grasp prediction.

        Args:
            xc:    Input image, shape (B, input_channels, H, W).
            yc:    Tuple (y_pos, y_cos, y_sin, y_width),
                   each of shape (B, 1, H_out, W_out).
            e_txt: Text embedding, shape (B, text_dim).

        Returns:
            dict with:
              'loss':   total loss
              'losses': dict of individual losses
              'pred':   dict of predicted maps
        """
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc, e_txt)

        p_loss = balanced_quality_bce_with_logits(
            pos_pred, y_pos, self.max_quality_pos_weight
        )
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
        width_loss = F.smooth_l1_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }

    def predict(self, xc, e_txt):
        """
        Inference helper.

        Args:
            xc:    Input image, shape (B, input_channels, H, W).
            e_txt: Text embedding, shape (B, text_dim).

        Returns:
            dict with predicted maps:
              'pos', 'cos', 'sin', 'width'
        """
        pos_pred, cos_pred, sin_pred, width_pred = self(xc, e_txt)
        return {
            'pos': pos_pred,
            'cos': cos_pred,
            'sin': sin_pred,
            'width': width_pred
        }

class GGCNN_CLIP(nn.Module):
    """
    CROG-style wrapper for a GG-CNN + CLIP (text-conditioned) grasp model.

    This class:
      - Uses the CROG/CLIP backbone to encode text into an embedding (state).
      - Uses GGCNNWithText as the grasp head, conditioned on the text embedding.
      - Mimics the CROG forward interface and return format:
            training: returns (pred, target, loss, loss_dict)
            eval:     returns (pred, target)
        where:
            pred   = (ins_pred, qua_pred, sin_pred, cos_pred, wid_pred)
            target = (ins_mask, qua_gt, sin_gt, cos_gt, wid_gt)
    This allows you to plug it directly into train_with_grasp / validate_with_grasp.
    """

    grasp_size_loss_activation = "clamp"
    grasp_quality_loss_activation = "sigmoid"

    def __init__(self, cfg):
        """
        Args:
            cfg: Configuration object (same style as CROG config), expected fields:
                 - clip_pretrain:  path to JITed CLIP checkpoint
                 - word_len:       max text token length
                 - word_dim:       text embedding dimension (backbone state dim)
                 - use_pretrained_clip: bool, whether to load CLIP weights
                 - input_channels: number of image channels (e.g. 3 for RGB, 4 for RGB-D)
        """
        super().__init__()

        self.use_pretrained_clip = cfg.use_pretrained_clip

        # Build CLIP-based backbone (CROG style). We only use encode_text() from it.
        clip_model = torch.jit.load(cfg.clip_pretrain,
                                    map_location="cpu").eval()
        print(f"[CROG_GGCNN_CLIP] Load pretrained CLIP: {self.use_pretrained_clip}")
        self.backbone = build_model(
            clip_model.state_dict(),
            cfg.word_len,
            self.use_pretrained_clip
        ).float()

        # Image channel count (e.g. 3 for RGB or 4 for RGBD)
        in_ch = getattr(cfg, "input_channels", 3)
        text_dim = cfg.word_dim  # must match backbone.encode_text(...) state dimension

        # GG-CNN grasp head with text conditioning
        self.grasp_head = GGCNNWithText(
            input_channels=in_ch,
            text_dim=text_dim,
            dropout=getattr(cfg, "dropout", False),
            prob=getattr(cfg, "dropout_prob", 0.0),
            max_quality_pos_weight=float(
                getattr(cfg, "ggcnn_quality_pos_weight", 32.0)
            ),
        )

    def forward(
        self,
        img,
        word,
        ins_mask=None,
        grasp_qua_mask=None,
        grasp_sin_mask=None,
        grasp_cos_mask=None,
        grasp_wid_mask=None,
        grasp_off_mask=None,
        grasp_off_weight=None,
    ):
        """
        Forward pass in CROG style.

        Args:
            img:              (B, C, H, W) input image tensor.
            word:             (B, L) tokenized text ids.
            ins_mask:         (B, 1, H_gt, W_gt) instance mask GT (used for IoU metric).
            grasp_qua_mask:   (B, 1, H_gt, W_gt) grasp quality GT.
            grasp_sin_mask:   (B, 1, H_gt, W_gt) grasp sin(theta) GT.
            grasp_cos_mask:   (B, 1, H_gt, W_gt) grasp cos(theta) GT.
            grasp_wid_mask:   (B, 1, H_gt, W_gt) grasp width GT.

        Returns:
            If self.training:
                preds, targets, total_loss, loss_dict
            else:
                preds, targets

            where:
                preds   = (ins_pred, qua_pred, sin_pred, cos_pred, wid_pred)
                targets = (ins_mask, qua_gt, sin_gt, cos_gt, wid_gt)
        """
        # Encode text using the CROG backbone's text encoder.
        # backbone.encode_text(...) typically returns (word_feat, state).
        # We use 'state' as the sentence-level embedding.
        _, state = self.backbone.encode_text(word)   # state: (B, text_dim)

        # Run GG-CNN with text conditioning
        pos_pred, cos_pred, sin_pred, wid_pred = self.grasp_head(img, state)

        # We do not truly have a segmentation head here.
        # To keep the CROG engine happy, we:
        #   - use pos_pred as both "instance mask prediction" and "quality map prediction".
        ins_pred = pos_pred
        qua_pred = pos_pred

        if self.training:
            # If GT spatial sizes differ from predictions, resize all masks to match.
            if grasp_qua_mask is not None and grasp_qua_mask.shape[-2:] != pos_pred.shape[-2:]:
                if ins_mask is not None:
                    ins_mask = F.interpolate(ins_mask, pos_pred.shape[-2:], mode='nearest').detach()
                grasp_qua_mask = F.interpolate(grasp_qua_mask, pos_pred.shape[-2:], mode='nearest').detach()
                grasp_sin_mask = F.interpolate(grasp_sin_mask, pos_pred.shape[-2:], mode='nearest').detach()
                grasp_cos_mask = F.interpolate(grasp_cos_mask, pos_pred.shape[-2:], mode='nearest').detach()
                grasp_wid_mask = F.interpolate(grasp_wid_mask, pos_pred.shape[-2:], mode='nearest').detach()

            # Compute regression losses for grasp maps.
            # Note: we treat pos_pred as quality prediction for training.
            p_loss = balanced_quality_bce_with_logits(
                pos_pred,
                grasp_qua_mask,
                self.grasp_head.max_quality_pos_weight,
            )
            cos_loss = F.smooth_l1_loss(cos_pred, grasp_cos_mask)
            sin_loss = F.smooth_l1_loss(sin_pred, grasp_sin_mask)
            wid_loss = F.smooth_l1_loss(wid_pred, grasp_wid_mask)

            total_loss = p_loss + cos_loss + sin_loss + wid_loss

            # Loss dict follows CROG naming convention so that the engine
            # can log loss_dict["m_qua"], ["m_sin"], ["m_cos"], ["m_wid"], ["m_ins"].
            loss_dict = {
                "m_ins": 0.0,                # no dedicated segmentation loss
                "m_qua": p_loss.item(),
                "m_sin": sin_loss.item(),
                "m_cos": cos_loss.item(),
                "m_wid": wid_loss.item(),
            }

            # preds: follow CROG tuple layout
            preds = (
                ins_pred.detach(),           # used for IoU metric in the engine
                qua_pred.detach(),           # quality map (here equal to pos_pred)
                sin_pred.detach(),
                cos_pred.detach(),
                wid_pred.detach(),
            )

            # targets: also follow CROG tuple layout
            targets = (
                ins_mask,
                grasp_qua_mask,
                grasp_sin_mask,
                grasp_cos_mask,
                grasp_wid_mask,
            )

            return preds, targets, total_loss, loss_dict

        else:
            # Evaluation / inference mode: only return predictions and targets.
            preds = (
                ins_pred.detach(),
                qua_pred.detach(),
                sin_pred.detach(),
                cos_pred.detach(),
                wid_pred.detach(),
            )
            targets = (
                ins_mask,
                grasp_qua_mask,
                grasp_sin_mask,
                grasp_cos_mask,
                grasp_wid_mask,
            )
            return preds, targets

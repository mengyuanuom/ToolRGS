"""Validation loop for segmentation and top-k grasp metrics."""

from loguru import logger
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from toolrgs.engine.hooks import LoopState
from toolrgs.engine.loops import BaseLoop
from toolrgs.models.base import dense_grasp_target_kwargs, model_requires_depth
from toolrgs.evaluation import (
    BinarySegmentationMetric,
    DenseGraspPostProcessor,
    GraspSuccessMetric,
    GraspThresholdGridMetric,
    inverse_warp,
    rectangles_to_five,
    refine_with_grasp_relative_offset,
    refine_with_offset,
    resample_grasp_geometry,
    targets_to_six,
)
from toolrgs.registry import LOOPS, METRICS, POSTPROCESSORS
from toolrgs.structures import GraspModelResult
from utils.grasp_eval import (
    calculate_grasp_matches,
    calculate_jacquard_from_matches,
    calculate_jacquard_index,
)
from utils.config import (
    resolve_grasp_quality_activation,
    resolve_grasp_size_activation,
)


def _resize_prediction(tensor, output_hw, mode="bicubic"):
    if tensor.shape[-2:] == tuple(output_hw):
        return tensor
    return F.interpolate(
        tensor,
        size=tuple(output_hw),
        mode=mode,
        # Preserve the historical evaluator's interpolation contract. Offset
        # vectors deliberately use bilinear/False because they are sampled in
        # input coordinates; the five dense maps use bicubic/True.
        align_corners=False if mode == "bilinear" else True,
    )


@LOOPS.register_module(name="grasp_val", aliases=("validate_with_grasp",))
class GraspValLoop(BaseLoop):
    """Evaluate instance IoU plus top-1/top-5 grasp Jacquard success."""

    def __init__(self, dataloader, model, cfg, hooks=None):
        super().__init__(hooks=hooks)
        self.dataloader = dataloader
        self.model = model
        self.cfg = cfg
        self.topk = tuple(getattr(cfg, "grasp_topk", (1, 5)))
        if not self.topk or any(int(value) <= 0 for value in self.topk):
            raise ValueError(f"grasp_topk must contain positive integers, got {self.topk}")
        self.max_topk = max(self.topk)
        self.collect_predictions = bool(
            getattr(cfg, "collect_predictions", False)
        )
        self.prediction_records = []
        self.segmentation_metric = METRICS.build(
            getattr(cfg, "segmentation_metric", None)
            or {
                "type": "binary_segmentation",
                "mask_threshold": float(getattr(cfg, "mask_threshold", 0.35)),
            }
        )
        self.grasp_metric = METRICS.build(
            getattr(cfg, "grasp_metric", None)
            or {"type": "grasp_success", "topk": self.topk}
        )
        self.compute_grasp_msr = bool(
            getattr(cfg, "compute_grasp_msr", False)
        )
        self.grasp_grid_metric = None
        self.grasp_threshold_grid = []
        self.grasp_msr = {}
        if self.compute_grasp_msr:
            self.grasp_grid_metric = GraspThresholdGridMetric(
                iou_thresholds=getattr(
                    cfg, "grasp_iou_thresholds", (0.25, 0.50, 0.75)
                ),
                angle_thresholds=getattr(
                    cfg, "grasp_angle_thresholds", (5.0, 10.0, 20.0, 30.0)
                ),
                topk=self.topk,
            )
        self.postprocessor = POSTPROCESSORS.build(
            getattr(cfg, "grasp_postprocessor", None)
            or {
                "type": "dense_grasp",
                "quality_threshold": float(
                    getattr(cfg, "grasp_quality_threshold", 0.4)
                ),
                "min_distance": int(getattr(cfg, "grasp_min_distance", 2)),
                "width_factor": float(
                    getattr(cfg, "grasp_size_factor", 100.0)
                ),
                "grasp_height": float(getattr(cfg, "grasp_height", 20.0)),
                "size_coordinate": str(
                    getattr(cfg, "grasp_size_coordinate", "canvas")
                ),
            }
        )
        self.grasp_size_activation = resolve_grasp_size_activation(
            getattr(cfg, "grasp_size_activation", "auto"), model=model
        )
        self.grasp_quality_activation = resolve_grasp_quality_activation(
            getattr(cfg, "grasp_quality_activation", "auto"), model=model
        )
        self.offset_decode_mode = str(
            getattr(cfg, "offset_decode_mode", "radius")
        ).strip().lower()
        if self.offset_decode_mode not in {"radius", "grasp_relative"}:
            raise ValueError(
                "offset_decode_mode must be 'radius' or 'grasp_relative', got "
                f"{self.offset_decode_mode!r}"
            )

    def _decode_size(self, prediction):
        if self.grasp_size_activation == "sigmoid":
            return torch.sigmoid(prediction)
        return prediction.clamp(0.0, 1.0)

    def _decode_quality(self, prediction):
        if self.grasp_quality_activation == "sigmoid":
            return torch.sigmoid(prediction)
        return prediction.clamp(0.0, 1.0)

    def _offset_radius(self, input_hw):
        configured = getattr(self.cfg, "offset_r", None)
        if configured is not None and float(configured) > 0:
            return float(configured)
        return max(1.0, min(input_hw) / 20.0)

    def _global_results(self, device):
        ious = np.asarray(self.segmentation_metric.ious, dtype=np.float64)
        values = [float(ious.sum()), float(ious.size)]
        values.extend(float((ious > threshold).sum()) for threshold in self.segmentation_metric.iou_thresholds)
        for topk in self.topk:
            values.extend(
                [self.grasp_metric.correct[topk], self.grasp_metric.total[topk]]
            )
        statistics = torch.tensor(values, dtype=torch.float64, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
        statistics = statistics.cpu().tolist()
        count = max(1.0, statistics[1])
        iou = statistics[0] / count
        precision = {
            f"Pr@{int(round(threshold * 100))}": statistics[2 + index] / count
            for index, threshold in enumerate(self.segmentation_metric.iou_thresholds)
        }
        cursor = 2 + len(self.segmentation_metric.iou_thresholds)
        j_index = []
        for _topk in self.topk:
            correct, total = statistics[cursor], statistics[cursor + 1]
            j_index.append(correct / max(1.0, total))
            cursor += 2
        return float(iou), precision, j_index

    def _global_grid_results(self, device):
        keys = tuple(self.grasp_grid_metric.correct)
        values = []
        for key in keys:
            values.extend(
                [
                    self.grasp_grid_metric.correct[key],
                    self.grasp_grid_metric.total[key],
                ]
            )
        statistics = torch.tensor(values, dtype=torch.float64, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
        statistics = statistics.cpu().tolist()
        for index, key in enumerate(keys):
            self.grasp_grid_metric.correct[key] = statistics[2 * index]
            self.grasp_grid_metric.total[key] = statistics[2 * index + 1]
        return self.grasp_grid_metric.compute()

    @torch.no_grad()
    def run_epoch(self, epoch: int):
        self.state = LoopState(epoch=epoch)
        self.hooks.call("before_epoch", self, self.state)
        self.segmentation_metric.reset()
        self.grasp_metric.reset()
        if self.grasp_grid_metric is not None:
            self.grasp_grid_metric.reset()
        if self.collect_predictions:
            self.prediction_records = []
        self.model.eval()
        rank = int(getattr(self.cfg, "rank", 0))
        progress = tqdm(self.dataloader, disable=rank != 0)
        device = torch.device("cuda", int(getattr(self.cfg, "gpu", 0)))

        for iteration, data in enumerate(progress):
            self.state.iteration = iteration
            self.state.batch = data
            self.hooks.call("before_iter", self, self.state)

            image = data["img"].cuda(non_blocking=True)
            text = data["word_vec"].cuda(non_blocking=True)
            target_segmentation = data["mask"].cuda(non_blocking=True).unsqueeze(1)
            target_quality = data["grasp_masks"]["qua"].cuda(non_blocking=True).unsqueeze(1)
            target_sine = data["grasp_masks"]["sin"].cuda(non_blocking=True).unsqueeze(1)
            target_cosine = data["grasp_masks"]["cos"].cuda(non_blocking=True).unsqueeze(1)
            target_width = data["grasp_masks"]["wid"].cuda(non_blocking=True).unsqueeze(1)
            target_short = data["grasp_masks"].get("short")
            if target_short is not None:
                target_short = target_short.cuda(non_blocking=True).unsqueeze(1)
            target_offset = data["grasp_masks"].get("off")
            if target_offset is not None:
                target_offset = target_offset.cuda(non_blocking=True)
            target_ltrb = data["grasp_masks"].get("ltrb")
            target_centerness = data["grasp_masks"].get("centerness")
            target_geometry_weight = data["grasp_masks"].get("geometry_weight")
            target_geometry_sine = data["grasp_masks"].get("geometry_sin")
            target_geometry_cosine = data["grasp_masks"].get("geometry_cos")
            asymmetric_targets = (
                target_ltrb,
                target_centerness,
                target_geometry_weight,
                target_geometry_sine,
                target_geometry_cosine,
            )
            asymmetric_targets = tuple(
                value.cuda(non_blocking=True) if value is not None else None
                for value in asymmetric_targets
            )

            depth_tensor = None
            if model_requires_depth(self.model):
                depth = data.get("depth")
                if depth is None:
                    raise KeyError(
                        "The selected model requires batch['depth'], but the "
                        "validation dataset did not provide it."
                    )
                depth_tensor = depth.cuda(non_blocking=True)
            model_args = (image, text)
            if depth_tensor is not None:
                model_args = (image, depth_tensor, text)
            model_kwargs = dense_grasp_target_kwargs(
                self.model,
                instance=target_segmentation,
                grasp_qua_mask=target_quality,
                grasp_sin_mask=target_sine,
                grasp_cos_mask=target_cosine,
                grasp_wid_mask=target_width,
                grasp_off_mask=target_offset,
                grasp_short_mask=target_short,
                grasp_ltrb_mask=asymmetric_targets[0],
                grasp_centerness_mask=asymmetric_targets[1],
                grasp_geometry_weight=asymmetric_targets[2],
                grasp_geometry_sin_mask=asymmetric_targets[3],
                grasp_geometry_cos_mask=asymmetric_targets[4],
            )
            unwrapped = getattr(self.model, "module", self.model)
            result = GraspModelResult.from_legacy(
                self.model(*model_args, **model_kwargs), model=unwrapped
            )
            predictions = result.predictions
            input_hw = image.shape[-2:]
            segmentation = _resize_prediction(
                torch.sigmoid(predictions.segmentation), input_hw
            )
            if predictions.quality is None:
                dense_maps = torch.cat(
                    [segmentation, target_segmentation], dim=1
                ).detach().float().cpu().numpy()
                for index in range(image.shape[0]):
                    inverse_matrix = data["inverse"][index]
                    if hasattr(inverse_matrix, "detach"):
                        inverse_matrix = inverse_matrix.detach().cpu().numpy()
                    original_hw = (
                        int(data["ori_size"][index][0]),
                        int(data["ori_size"][index][1]),
                    )
                    self.segmentation_metric.update(
                        inverse_warp(
                            dense_maps[index, 0], inverse_matrix, original_hw
                        ),
                        inverse_warp(
                            dense_maps[index, 1], inverse_matrix, original_hw
                        ) > 0.5,
                    )
                    if self.collect_predictions:
                        self.prediction_records.append(
                            {
                                "segmentation_iou": self.segmentation_metric.ious[-1],
                                "rectangles": np.empty((0, 5), dtype=np.float32),
                                "targets": np.empty((0, 6), dtype=np.float32),
                                "matches": np.empty((0, 3), dtype=np.float32),
                                "target_width_cap": self.postprocessor.width_factor,
                                "target_height": self.postprocessor.grasp_height,
                            }
                        )
                self.state.result = result
                self.hooks.call("after_iter", self, self.state)
                continue
            quality = _resize_prediction(
                self._decode_quality(predictions.quality), input_hw
            )
            sine = _resize_prediction(predictions.sine, input_hw)
            cosine = _resize_prediction(predictions.cosine, input_hw)
            width = _resize_prediction(self._decode_size(predictions.width), input_hw)
            short_side = None
            if predictions.short_side is not None:
                short_side = _resize_prediction(
                    self._decode_size(predictions.short_side), input_hw
                )
            offset = None
            if predictions.offset is not None:
                offset = _resize_prediction(predictions.offset, input_hw, mode="bilinear")

            dense_tensors = [
                segmentation,
                target_segmentation,
                quality,
                sine,
                cosine,
                width,
            ]
            if offset is not None:
                dense_tensors.append(offset)
            if short_side is not None:
                dense_tensors.append(short_side)
            dense_maps = (
                torch.cat(dense_tensors, dim=1)
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            cursor = 6
            offset_maps = None
            if offset is not None:
                offset_maps = dense_maps[:, cursor:cursor + 2]
                cursor += 2
            short_side_maps = (
                dense_maps[:, cursor] if short_side is not None else None
            )

            for index in range(image.shape[0]):
                inverse_matrix = data["inverse"][index]
                if hasattr(inverse_matrix, "detach"):
                    inverse_matrix = inverse_matrix.detach().cpu().numpy()
                original_hw = (
                    int(data["ori_size"][index][0]),
                    int(data["ori_size"][index][1]),
                )
                predicted_segmentation = inverse_warp(
                    dense_maps[index, 0], inverse_matrix, original_hw
                )
                target_segmentation_original = inverse_warp(
                    dense_maps[index, 1], inverse_matrix, original_hw
                )
                self.segmentation_metric.update(
                    predicted_segmentation,
                    target_segmentation_original > 0.5,
                )
                segmentation_iou = self.segmentation_metric.ious[-1]

                quality_original = inverse_warp(
                    dense_maps[index, 2], inverse_matrix, original_hw
                )
                sine_original = inverse_warp(
                    dense_maps[index, 3], inverse_matrix, original_hw
                )
                cosine_original = inverse_warp(
                    dense_maps[index, 4], inverse_matrix, original_hw
                )
                width_original = inverse_warp(
                    dense_maps[index, 5], inverse_matrix, original_hw
                )
                short_side_original = (
                    inverse_warp(
                        short_side_maps[index], inverse_matrix, original_hw
                    )
                    if short_side_maps is not None else None
                )
                grasp_targets = data["grasps"][index]
                if hasattr(grasp_targets, "detach"):
                    grasp_targets = grasp_targets.detach().cpu().numpy()
                target_six = targets_to_six(grasp_targets)

                size_scale = 1.0
                if (
                    self.postprocessor.size_coordinate == "canvas"
                    and bool(getattr(self.cfg, "restore_grasp_size_scale", False))
                ):
                    linear = np.asarray(inverse_matrix, dtype=np.float32)[:, :2]
                    size_scale = float(
                        0.5
                        * (
                            np.linalg.norm(linear[:, 0])
                            + np.linalg.norm(linear[:, 1])
                        )
                    )

                detections = self.postprocessor(
                    quality_original,
                    sine_original,
                    cosine_original,
                    width_original,
                    short_side=short_side_original,
                    num_grasps=self.max_topk,
                    spatial_scale=size_scale,
                )
                rectangles = [detection.as_rectangle() for detection in detections]
                if offset_maps is not None and rectangles:
                    if self.offset_decode_mode == "grasp_relative":
                        rectangles = refine_with_grasp_relative_offset(
                            rectangles,
                            offset_maps[index : index + 1],
                            inverse_matrix,
                        )
                    else:
                        rectangles = refine_with_offset(
                            rectangles,
                            offset_maps[index : index + 1],
                            inverse_matrix,
                            self._offset_radius(input_hw),
                        )
                    if bool(
                        getattr(self.cfg, "offset_resample_geometry", False)
                    ):
                        rectangles = resample_grasp_geometry(
                            rectangles,
                            sine_original,
                            cosine_original,
                            width_original,
                            short_side=short_side_original,
                            width_factor=(
                                self.postprocessor.width_factor * size_scale
                            ),
                        )
                else:
                    rectangles = rectangles_to_five(rectangles)

                target_width_cap = self.postprocessor.width_factor
                if self.postprocessor.size_coordinate == "canvas":
                    target_width_cap *= size_scale
                matches = None
                if self.collect_predictions or self.grasp_grid_metric is not None:
                    matches = calculate_grasp_matches(
                        rectangles[: self.max_topk],
                        target_six,
                        target_width_cap=target_width_cap,
                        target_height=self.postprocessor.grasp_height,
                    )
                for topk in self.topk:
                    if matches is None:
                        success = calculate_jacquard_index(
                            rectangles[:topk],
                            target_six,
                            iou_threshold=float(
                                getattr(self.cfg, "grasp_iou_threshold", 0.25)
                            ),
                            angle_threshold=float(
                                getattr(self.cfg, "grasp_angle_threshold", 30.0)
                            ),
                            target_width_cap=target_width_cap,
                            target_height=self.postprocessor.grasp_height,
                        )
                    else:
                        success = calculate_jacquard_from_matches(
                            matches,
                            topk,
                            iou_threshold=float(
                                getattr(self.cfg, "grasp_iou_threshold", 0.25)
                            ),
                            angle_threshold=float(
                                getattr(self.cfg, "grasp_angle_threshold", 30.0)
                            ),
                        )
                    self.grasp_metric.update(topk, success)
                if self.grasp_grid_metric is not None:
                    for iou_threshold, angle_threshold in (
                        self.grasp_grid_metric.threshold_pairs
                    ):
                        for topk in self.topk:
                            success = calculate_jacquard_from_matches(
                                matches,
                                topk,
                                iou_threshold=iou_threshold,
                                angle_threshold=angle_threshold,
                            )
                            self.grasp_grid_metric.update(
                                iou_threshold,
                                angle_threshold,
                                topk,
                                success,
                            )
                if self.collect_predictions:
                    self.prediction_records.append(
                        {
                            "segmentation_iou": segmentation_iou,
                            "rectangles": np.asarray(
                                rectangles, dtype=np.float32
                            ).reshape(-1, 5),
                            "targets": np.asarray(
                                target_six, dtype=np.float32
                            ).reshape(-1, 6),
                            "matches": np.asarray(
                                matches, dtype=np.float32
                            ).reshape(-1, 3),
                            "target_width_cap": target_width_cap,
                            "target_height": self.postprocessor.grasp_height,
                        }
                    )

            self.state.result = result
            self.hooks.call("after_iter", self, self.state)

        iou, precision, j_index = self._global_results(device)
        self.grasp_threshold_grid = []
        self.grasp_msr = {}
        if self.grasp_grid_metric is not None:
            grid_results = self._global_grid_results(device)
            self.grasp_threshold_grid = grid_results["rows"]
            self.grasp_msr = grid_results["msr"]
        self.state.logs = {
            "iou": iou,
            "precision": precision,
            "j_index": j_index,
            "grasp_threshold_grid": self.grasp_threshold_grid,
            "msr": self.grasp_msr,
        }
        self.hooks.call("after_epoch", self, self.state)
        if rank == 0:
            precision_text = "  ".join(
                f"{name}: {100.0 * value:.2f}" for name, value in precision.items()
            )
            grasp_text = "  ".join(
                f"J_index@{topk}: {100.0 * value:.2f}"
                for topk, value in zip(self.topk, j_index)
            )
            logger.info(
                "Evaluation: Epoch=[{}/{}]  IoU={:.2f}  {}  {}",
                epoch,
                self.cfg.epochs,
                100.0 * iou,
                grasp_text,
                precision_text,
            )
            if self.grasp_msr:
                logger.info(
                    "mSR: {}",
                    "  ".join(
                        f"mSR@{topk}={100.0 * value:.4f}"
                        for topk, value in self.grasp_msr.items()
                    ),
                )
        return iou, precision, j_index

# Real-world ToolRGS deployment

This deployment layer replaces the server CROG demo's hard-coded model, paths,
camera pipeline, IP address, and automatic TCP sends with one YAML file. It
supports every ToolRGS grasp architecture, OpenCV/video, Intel
RealSense, shared-memory GStreamer, an optional MMDetection tab, and optional
Whisper microphone input. The deployed 22-class workflow is also available as
optional components: GelSight classification, segmentation-mask span width,
semantic depth tiers, and receiver-specific angle conversion.

For copy-ready Chinese commands covering the direct RealSense and GI/GStreamer
launchers, model downloads, `fooA`, TCP port `3000`, coordinate sending and
fault diagnosis, see [机械臂 GUI 启动与故障排查](robot_gui_quickstart_zh.md).

Supported RGB grasp models are CROG, CROGOFF, DROG, DROGOFF, MapleGrasp,
GGCNN-CLIP, GR-ConvNet-CLIP, GraspMamba, and LGD. ETRG needs aligned depth and
is therefore not selectable from the current RGB camera GUI. DETRIS is intentionally excluded here: in
this repository it is a segmentation/backbone component, not a grasp-map model.

## 1. Install

Use the same Python 3.9/CUDA environment as training, then install the GUI and
camera extras:

```bash
pip install -r requirement.txt
pip install -r requirement-deploy.txt
```

On Linux, ToolRGS resets `QT_QPA_PLATFORM_PLUGIN_PATH` after importing OpenCV
so that QApplication uses the active PyQt5 installation instead of
`cv2/qt/plugins`. Startup prints the selected directory as
`[gui] PyQt5 platform plugins: ...`. For a non-standard PyQt installation,
set `TOOLRGS_QT_PLUGIN_PATH` to its plugin root (the parent of
`platforms/libqxcb.so`).

For GStreamer on Ubuntu, install the system GI bindings and plugins. PyGObject
is normally installed through `apt`, not `pip`:

```bash
sudo apt install python3-gi gir1.2-gstreamer-1.0 \
  gstreamer1.0-tools gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad
```

Whisper also needs `ffmpeg`. Object detection is optional and needs an
MMCV/MMDetection build compatible with the installed CUDA and PyTorch versions;
it is intentionally not included in the base requirements.
GelSight classification uses `torchvision`; install the torchvision build that
matches the PyTorch/CUDA environment already used by ToolRGS.

The 22-class Grasp-Tools detector environment uses MMEngine `0.10.7`, MMCV
`2.1.0`, and the MMDetection source revision recorded in
`requirement-detector.txt`. Install these only when the Object detection tab
is needed:

```bash
pip install -U openmim
mim install "mmengine==0.10.7" "mmcv==2.1.0"
pip install -r requirement-detector.txt
```

## 2. Put weights in place

Training, detector, and Whisper weights are not committed to Git. Official
CLIP/DINO/Mamba backbones are downloaded automatically on first use when the
machine has network access. A model profile may also specify
`checkpoint_url` and `checkpoint_sha256`; when its local `checkpoint` is
missing, ToolRGS downloads the release asset atomically and validates the hash.
For example:

```text
pretrain/ViT-B-16.pt
pretrain/dinov2_vitb14_reg4_pretrain.pth
exp/grasp_tools/drogoff_grasp_tools/best_jindex_model.pth
weights/faster_rcnn_r50_fpn_grasp_tools_v2_best.pth  # detector.enabled=true
weights/gelsight_best.pt                # only when gelsight.enabled=true
```

The 22-class Faster R-CNN checkpoint belongs at
`weights/faster_rcnn_r50_fpn_grasp_tools_v2_best.pth`. Its class order is
defined by `configs/detection/faster_rcnn_r50_fpn_grasp_tools_v2_24e.py` and
must not be changed. The checked-in Release URL and SHA-256 let ToolRGS download
and verify the detector weight just like the grasp model.
The checked-in lab profile sets `trusted_checkpoint: true` because the
original detector checkpoint was produced by the trusted lab training
environment and contains MMEngine `HistoryBuffer` metadata. On PyTorch 2.6+,
ToolRGS allowlists that metadata type and, for this exact resolved checkpoint
path only, temporarily passes `weights_only=False` while MMDetection
initializes it. The original `torch.load` is restored immediately afterward;
other files and the DROG-OFF loader are unaffected. Never enable this option for
an untrusted checkpoint.

The checked-in lab profiles select `drogoff-grasptools-v2-original300` by
default. It downloads the current original-coordinate checkpoint from the
[ToolRGS weight release](https://github.com/mengyuanuom/ToolRGS/releases/tag/grasp-tools-v2-weights).
This is the epoch-32 J@1-best checkpoint from the original-scale run;
its validation result is segmentation IoU `84.19`, J@1 `62.52`, and J@5
`63.60` under rotated IoU `>0.50` and angle error `<=30` degrees. It has not
yet received a formal test-set score and therefore does not replace the older
published Test result in `performance_summary.md`.

The model and GUI now share the same size contract: grasp long-side values use
an original-image normalization factor of `300`, while the displayed short
side is always `20` original-image pixels. `size_coordinate: original`
prevents a 1280x720 RealSense frame from applying the 448px canvas scale to
the grasp width a second time. The Release asset is named
`drogoff_grasp_tools_v2_original300_20260814_epoch32_best_j1.pth`, is
`952762694` bytes, and has
SHA-256
`38b2824b385a293b883bfe03682fec0e2f30c83457e3f86ba08e14d884fd7ecf`.
ToolRGS validates that digest before loading the checkpoint.

The two Grasp-Tools V3 profiles are deliberately named and ordered first in the
GUI as `V3-DROG-OFF-V1` and `V3-CROG`. `V3-DROG-OFF-V1` provides the epoch-24
best-mSR checkpoint: segmentation IoU `83.78`, multi-IoU/multi-angle mSR
`89.14`, legacy SR@IoU0.25/30deg `99.51`, and SR@IoU0.50/10deg `96.52`. Its
Release asset is
`v3_drogoff_v1_grasp_tools_15k_original300_20260825_epoch24_best_msr_inference.pth`
(`717897106` bytes), with SHA-256
`5f15b5f59e783b9daf3b34bf1d467274591c15fc6f590c36653f128b90dff340`.

`V3-CROG` provides the max-norm-fixed epoch-36 checkpoint resumed from epoch 20.
With the training-matched `clamp` size decoder it reaches segmentation IoU
`81.72`, multi-IoU/multi-angle mSR `75.28`, legacy SR@IoU0.25/30deg `99.15`,
and SR@IoU0.50/10deg `90.81`. Its Release asset is
`v3_crog_grasp_tools_15k_original300_20260823_epoch36_best_msr_clamp_inference.pth`
(`589066850` bytes), with SHA-256
`2d1270024beedde710b8a78b83c83591d3166debed479ad20450a88b80530a4f`.
Both assets contain only inference state plus geometry metadata; optimizer and
scheduler state are deliberately omitted. The activation contract is explicit
at every layer: `V3-DROG-OFF-V1` uses `sigmoid`, while `V3-CROG` uses `clamp`,
matching how each size head was trained.
Selecting either profile downloads and verifies it on demand; running the
preflight downloads both in advance with the other selectable profiles.

Paths in deployment YAML are resolved from the ToolRGS repository root, so the
command can be run from any working directory.

## 3. Preflight the checked-in lab configuration

```bash
python tools/check_deployment.py
python tools/check_deployment.py \
  --probe-camera --build-model --build-detector
# When GelSight is enabled and connected:
python tools/check_deployment.py --config config/deployment/lab.yaml \
  --probe-camera --probe-gelsight --build-model
```

`config/deployment/lab.yaml` is committed as the ready-to-run physical
profile: RealSense, DROG-OFF, and the 22-class Faster R-CNN detector are
enabled, while robot output remains disabled. `lab.example.yaml` remains the
extended reference. A normal deployment no longer needs a YAML copy/edit step.

The preflight never connects to the robot and never sends a command. It downloads
and verifies every selectable `model_profiles` checkpoint, not only the active
one. In the GUI, expand **Model & Post-processing** to switch among the published
`V3-DROG-OFF-V1`, `V3-CROG`, DROG-OFF V2, and aligned CROG V2 profiles without
restarting the camera, detector, or robot-control layer. The same panel exposes
source-pixel grasp height, mask threshold, mask expansion radius, and independent
mask-based grasp point filtering; every initial value comes from the selected
YAML profile.

To enable detection, set `detector.enabled: true`. The GUI then creates a
separate **Object detection** tab. `score_threshold` controls displayed boxes,
`max_detections` caps boxes per frame, and `inference_interval_ms` controls
detector refresh independently from grasp inference. The bundled inference
pipeline accepts live NumPy/RealSense frames and does not require annotation
files.

Camera components are selected with `camera.type` (`camera.backend` remains a
legacy alias):

- `opencv`: integer USB camera index, RTSP URL, or other OpenCV source.
- `video`: repository-relative or absolute video path.
- `image`: repeat one image; useful for a safe end-to-end GUI check.
- `realsense`: direct color stream through `pyrealsense2`.
- `gstreamer`: shared-memory or network pipeline ending in `appsink`. Start from
  `config/deployment/gstreamer.example.yaml` for the old CROG `shmsrc` layout.

The physical lab profile requests a 1280x720 BGR color stream at 30 FPS. The
model's 416/448 input canvas is independent: ToolRGS letterboxes the camera
frame for inference and maps the selected grasp back to 1280x720 source pixels
before robot transmission.

## 4. Dry-run the GUI

Robot permission is denied unless the launcher receives `--allow-robot`, even
though the committed laboratory profile contains the receiver settings. Use the
official GI preview without that flag:

```bash
python deploy_gui_gi.py --config config/deployment/lab.yaml
```

### Linux 无摄像头验证

GUI 可以直接把静态图片当作相机帧，不需要安装 RealSense SDK，也不会连接机械臂。
仓库自带样例图时，直接运行：

```bash
python deploy_gui.py \
  --config config/deployment/lab.yaml \
  --image \
  --prompt "the tool"
```

使用自己的图片：

```bash
python deploy_gui.py \
  --config config/deployment/lab.yaml \
  --image /absolute/path/to/test.jpg \
  --prompt "the sponge"
```

图片模式会自动关闭连续推理。GUI 打开后点击 **Predict now**，即可检查语言抓取、
分割图、质量图、角度图和宽度图；若部署 YAML 已启用目标检测，目标检测页也会使用
同一张图片。不要添加 `--allow-robot`；命令行权限门会阻止 TCP 连接和发送。

Check the segmentation overlay, grasp rectangle, center, angle, and width before
using a physical robot. Object detector, audio, and GelSight controls only
appear when their respective `enabled` settings are true. GelSight follows the
historic checkpoint contract (`arch`, `classes`, and `model`/`state_dict`) and
labels confidence below `confidence_threshold` as `Nothing`.

## 5. Receiver and coordinate contract

The server CROG snapshot does **not** contain the Kinova-side receiver, robot
motion controller, hand-eye calibration, workspace limits, or collision logic.
ToolRGS therefore supplies the compatible sender, not those missing components.
The external receiver must already be running and validated.

The sender emits one ASCII line per command:

```text
{x, y, theta, width, depth}\n
```

`robot.type: legacy_tcp` opens a TCP socket to the configured `host` and
`port` (the current host-network Docker profile uses `127.0.0.1:3000`) and calls
`sendall` with this exact newline-terminated payload. The GUI exposes Connect,
Disconnect, Arm, and Send controls; a socket error closes the connection and
requires an explicit reconnect. With `auto_send: true`, the same guarded socket
path is rate-limited by `auto_send_interval_s`.

- `x`, `y`: grasp center in the configured coordinate space.
- `theta`: image-plane grasp angle in degrees.
- `width`: gripper rectangle width in pixels.
- `depth`: the old demo's semantic tier (`-1`, `0`, or `1`), **not a RealSense
  depth measurement**.

The robot conversion rules are explicit in YAML:

- `width_policy.type: model` sends the predicted grasp width.
- `width_policy.type: mask_span` measures the segmentation span along the
  grasp axis and adds `safety_margin` (30 pixels in the legacy experiment).
  Tape and cable can be excluded without accidentally excluding tape measure.
- `theta_policy` applies the receiver's sign, offset, and normalization. The
  RealSense compatibility profile uses the deployed `theta + 180` convention.
- `depth_policy.class_tiers` overrides individual entries in the 22-class
  semantic-depth table when a particular robot setup uses a different tier.

Set `robot.coordinate_space` to match the receiver calibration:

- `source`: original camera pixels after inverse letterbox; recommended for a
  new calibration.
- `model`: letterboxed model-input pixels; useful for a receiver calibrated in
  ToolRGS input space.

Do not guess this setting. A receiver calibrated against the historic demo's
stretched 416x416 image is not automatically equivalent to either mapping and
must be recalibrated or given an explicit compatibility transform.

`robot.limits` is a second sender-side guard for center, angle, width, and depth.
Update these bounds for the configured coordinate space; an out-of-range command
is rejected before any bytes are sent.

## 6. Enable physical output

After verifying the receiver, calibration, robot limits, emergency stop, and
dry-run prediction:

1. Confirm `robot.enabled: true`, the correct `host`/`port`, and use
   `auto_send: false` for the first manual test.
2. Launch with the additional command-line permission:

   ```bash
   python deploy_gui_gi.py --config config/deployment/lab.yaml --allow-robot
   ```

3. Click **Connect receiver**, then explicitly tick **Arm robot output**.
4. Use **Send current grasp** for the first tests.

Automatic sending additionally requires `auto_send: true` and an armed GUI;
`auto_send_interval_s` rate-limits it. Manual sending is recommended until the
whole calibrated workspace has been tested.

## Ported components

The reusable pieces from the local server snapshot are now modules: grasp GUI,
shared-memory/direct camera access, optional 22-class detector, Whisper input,
GelSight classifier/tab, 22-class semantic depth tiers, mask-derived opening
width, receiver angle conversion, and the legacy TCP sender. All are selected
through YAML and registered components. Absolute `/home/...` paths, fixed CUDA
devices, Qt plugin paths, global `torch.load` monkey patches, and automatic
every-50-frame sends were removed. Robot output still requires both the
`--allow-robot` launch permission and explicit arming in the GUI.

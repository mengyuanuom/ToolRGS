# Real-world ToolRGS deployment

This deployment layer replaces the server CROG demo's hard-coded model, paths,
camera pipeline, IP address, and automatic TCP sends with one YAML file. It
supports every ToolRGS grasp architecture, OpenCV/video, Intel
RealSense, shared-memory GStreamer, an optional MMDetection tab, and optional
Whisper microphone input. The deployed 22-class workflow is also available as
optional components: GelSight classification, segmentation-mask span width,
semantic depth tiers, and receiver-specific angle conversion.

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
weights/epoch_48_13.pth                 # only when detector.enabled=true
weights/gelsight_best.pt                # only when gelsight.enabled=true
```

The checked-in `lab.example.yaml` contains separate CROG and DROG-OFF profiles.
The verified 3090 DROG-OFF release URL will be recorded in that profile and in
the performance summary once the server checkpoint has been recovered and its
SHA-256 has been checked. Do not attach a checkpoint merely because its
filename looks plausible: it must be tied to the reported V2 strict-IoU run.

Paths in deployment YAML are resolved from the ToolRGS repository root, so the
command can be run from any working directory.

## 3. Create a lab config and preflight it

```bash
cp config/deployment/lab.example.yaml config/deployment/lab.yaml
# Edit active_model/model_profiles, camera path/device, and prompt.
python tools/check_deployment.py --config config/deployment/lab.yaml
python tools/check_deployment.py --config config/deployment/lab.yaml \
  --probe-camera --build-model
# When GelSight is enabled and connected:
python tools/check_deployment.py --config config/deployment/lab.yaml \
  --probe-camera --probe-gelsight --build-model
```

The preflight never connects to the robot and never sends a command. In the
GUI, the **Model** selector reloads any entry under `model_profiles` without
restarting the camera or robot-control layer.

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

Keep `robot.enabled: false` and run:

```bash
python deploy_gui.py --config config/deployment/lab.yaml
```

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
`port` (the compatibility profile uses `192.168.38.10:3000`) and calls
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

1. Set `robot.enabled: true`, the correct `host`/`port`, and leave
   `auto_send: false`.
2. Launch with the additional command-line permission:

   ```bash
   python deploy_gui.py --config config/deployment/lab.yaml --allow-robot
   ```

3. Click **Connect receiver**, then explicitly tick **Arm robot output**.
4. Use **Send current grasp** for the first tests.

Automatic sending additionally requires `auto_send: true` and an armed GUI;
`auto_send_interval_s` rate-limits it. Manual sending is recommended until the
whole calibrated workspace has been tested.

## Ported components

The reusable pieces from the local server snapshot are now modules: grasp GUI,
shared-memory/direct camera access, optional 13-class detector, Whisper input,
GelSight classifier/tab, 22-class semantic depth tiers, mask-derived opening
width, receiver angle conversion, and the legacy TCP sender. All are selected
through YAML and registered components. Absolute `/home/...` paths, fixed CUDA
devices, Qt plugin paths, global `torch.load` monkey patches, and automatic
every-50-frame sends were removed. Robot output still requires both the
`--allow-robot` launch permission and explicit arming in the GUI.

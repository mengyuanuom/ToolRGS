# ToolRGS GUI 快速上手（实验室正式版）

这份文档面向直接使用 ToolRGS 的同事。正式仓只保留两个 GUI 入口，默认配置为
`config/deployment/lab.yaml`。

## 1. GUI 的区别

| 入口 | 图像来源 | 机械臂发送 | 用途 |
| --- | --- | --- | --- |
| `deploy_gui_realsense.py` | `pyrealsense2` 直接打开 RealSense，1280×720 | **永久关闭** | 纯相机/模型演示，没有 GI，不控制机械臂 |
| `deploy_gui_gi.py` | GI/GStreamer 共享内存 `fooA`，1280×720 | 只有加 `--allow-robot` 才允许 | 实验室正式机械臂 GUI |

以前的第三个 `deploy_gui_legacy_gi.py` 只是旧布局/阻塞 socket 的测试入口，现已从
正式仓删除。普通使用者不要再运行它。`deploy_gui.py` 是底层通用调试入口，不是
机械臂正式入口。

GI 图像来自：

```text
/home/raico-hri/v1/kinova_rs_grasp/foo/fooA
```

上游进程已占用 RealSense 时，必须使用 GI GUI；这时再用纯 RealSense GUI 可能出现
`RuntimeError: No device connected`。

## 2. 拉取或更新项目

第一次部署：

```bash
cd /home/raico-hri/projects
git clone https://github.com/mengyuanuom/ToolRGS.git
cd ToolRGS
```

已有项目时：

```bash
cd /home/raico-hri/projects/ToolRGS
git status --short
git pull --ff-only origin main
```

如果 `git pull` 提示本地改动会被覆盖，先保存改动再更新：

```bash
git stash push -u -m "local deployment settings"
git pull --ff-only origin main
git stash pop
```

不要直接 `git reset --hard`，否则本地配置可能丢失。

## 3. 一次性安装依赖

在训练/部署所用的 Python 3.9 CUDA 环境中执行：

```bash
cd /home/raico-hri/projects/ToolRGS
pip install -r requirement.txt
pip install -r requirement-deploy.txt
pip install -r requirement-detector.txt
```

纯 RealSense GUI 需要 `pyrealsense2`。GI GUI 还需要系统 GStreamer：

```bash
sudo apt install python3-gi gir1.2-gstreamer-1.0 \
  gstreamer1.0-tools gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad
```

## 4. 检查并自动下载权重

推荐只记住这一条：

```bash
bash tools/gui_quickstart.sh check
```

它等价于：

```bash
python tools/check_deployment.py \
  --config config/deployment/lab.yaml \
  --download-weights
```

该命令不会打开相机、不会连接 TCP、不会发送机械臂指令。它会：

1. 检查 Python/Qt/MMDetection 环境；
2. 从配置的 GitHub Release 下载 DROG-OFF 抓取权重；
3. 下载缺失的 CLIP ViT-B/16 和 DINOv2 预训练权重；
4. 对配置了 Release URL 的权重做 SHA-256 校验；
5. 检查 13 类目标检测权重。

主要文件应位于：

```text
weights/drogoff_grasp_tools_v2_original300_best_j1.pth
weights/epoch_48_13.pth
pretrain/ViT-B-16.pt
pretrain/dinov2_vitb14_reg4_pretrain.pth
```

抓取、CLIP、DINO 权重已配置自动下载。当前 `epoch_48_13.pth` 尚未发布到 ToolRGS
Release，因此 `check` 会在完成其他下载后明确报告 Detector 权重缺失；现阶段需从
实验室可信备份复制到 `weights/epoch_48_13.pth`。它发布到 Release 并在 YAML 填入
`detector.checkpoint_url`/`checkpoint_sha256` 后，同一条 `check` 命令会自动下载它，
无需改脚本。

权重齐全后做一次完整加载验证：

```bash
python tools/check_deployment.py \
  --config config/deployment/lab.yaml \
  --build-model --build-detector
```

看到 `Preflight completed with 0 failure(s).` 才进入正式实验。

## 5. 一键启动

### A. 纯 RealSense 演示（无 GI、绝不发送坐标）

```bash
bash tools/gui_quickstart.sh demo --prompt "the screwdriver"
```

该入口会强制设置 `robot.enabled=false`，也不接受 `--allow-robot`，适合调模型、
目标检测和直接相机画面。

### B. GI 画面安全预览（不发送坐标）

```bash
bash tools/gui_quickstart.sh gi-preview --prompt "the screwdriver"
```

它读取 `fooA`，但没有 `--allow-robot`，因此不会连接机械臂 TCP。正式实验前先用
这个模式确认画面、目标检测、分割、抓取中心、角度和宽度都正确。

### C. GI 正式机械臂模式（会发送坐标）

先在机械臂 Docker 中启动接收程序：

```bash
docker exec -it kinova_rs_grasp bash
cd /workspace/scripts
source /opt/ros/noetic/setup.bash
python test_socket_to_ros1.py
```

接收端应显示：

```text
[+] Listening on port 0.0.0.0 : 3000
```

保持它运行，再在宿主机 ToolRGS 环境执行：

```bash
cd /home/raico-hri/projects/ToolRGS
bash tools/gui_quickstart.sh robot --prompt "the screwdriver"
```

该命令等价于：

```bash
python deploy_gui_gi.py \
  --config config/deployment/lab.yaml \
  --allow-robot \
  --prompt "the screwdriver"
```

实验室 Docker 使用 `network_mode=host`，所以正式配置连接的是：

```text
127.0.0.1:3000
```

不要再写旧地址 `192.168.38.10:3000`。当前配置启用 `auto_connect`、`auto_arm`
和 `auto_send`：GUI 建立连接后会按 2 秒最小间隔发送有效预测。首次物理测试前必须
确认手眼标定、工作空间限制、急停和接收端均正常。

若希望先人工点击发送，将 `config/deployment/lab.yaml` 改为：

```yaml
robot:
  auto_arm: false
  auto_send: false
```

然后在 GUI 中依次点击 **Connect receiver**、勾选 **Arm robot output**，确认命令
预览后点击 **Send current grasp**。

## 6. 发送的数据是什么

TCP 每次发送一行 ASCII：

```text
{x, y, theta, width, depth}\n
```

- `x, y`：映射回 RealSense 原始 1280×720 图像的抓取中心；
- `theta`：按接收端契约执行 `theta + 180`，归一化到 `[0, 360)`；
- `width`：DROG-OFF 模型预测宽度，已经映射到原图像素，不使用分割边缘宽度；
- `depth`：旧协议的语义层级 `-1/0/1`，不是 RealSense 深度值；
- 抓取框短边固定为 20 px，用于可视化/评测矩形，不作为单独 TCP 字段发送。

GUI 的命令预览与 TCP 发送共用同一个 `GraspCommand`，发送前还会检查
`robot.limits`。

## 7. 快速排错

### GI 黑屏或没有画面

```bash
ls -l /home/raico-hri/v1/kinova_rs_grasp/foo/fooA
```

确认上游发布程序在运行，且输出 BGR、1280×720、30 FPS。

### 纯 RealSense 报 `No device connected`

相机未连接、权限不足或已被 GI 上游占用。实验室正式流程请改用：

```bash
bash tools/gui_quickstart.sh gi-preview
```

### GUI 显示 `ROBOT COMMAND OFFLINE` / `DRY RUN`

- `gi-preview` 本来就是安全预览，不连接机器人；
- 正式模式必须运行 `bash tools/gui_quickstart.sh robot`；
- 确认 Docker 接收程序仍监听 `0.0.0.0:3000`；
- 宿主机检查：`ss -ltnp | grep :3000`；
- 确认 `config/deployment/lab.yaml` 中是 `host: 127.0.0.1`。

GI 正式 GUI 的 TCP 超时为 2 秒。接收端没启动时 GUI 会快速报告失败，不会无限
阻塞或黑屏；启动接收端后点击 **Connect receiver** 重连即可。

### Qt `xcb` 错误

ToolRGS 会优先使用 PyQt5 自己的 platform plugins，避免 OpenCV 的
`cv2/qt/plugins` 抢占。若仍失败，运行 `bash tools/gui_quickstart.sh check` 查看缺失
的 Qt/xcb 依赖。

### Detector `weights_only` 错误

`epoch_48_13.pth` 包含旧 MMEngine `HistoryBuffer` 元数据。仓库只对配置中明确标记
为可信的这一个本地路径启用兼容加载；不要对来源不明的 checkpoint 启用
`trusted_checkpoint`。

## 8. 停止 GUI

正常关闭窗口会释放 RealSense/GStreamer 和 TCP socket。异常退出后检查残留：

```bash
pgrep -af "deploy_gui|realsense_object_grasp"
```

确认旧进程退出后再重新启动，避免相机或 TCP 长连接冲突。

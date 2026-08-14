# ToolRGS 机械臂 GUI 启动与故障排查

本文档适用于实验室机械臂服务器上的完整 ToolRGS GUI，包括 DROG-OFF
语言抓取、13 类目标检测、RealSense/GI 图像输入以及 Kinova TCP 坐标发送。

## 1. 三个 GUI 入口

三个入口共享相同的 ToolRGS 模型、目标检测、尺度映射和机械臂发送模块。

| 入口 | 图像来源 | 适用场景 |
| --- | --- | --- |
| `deploy_gui_realsense.py` | `pyrealsense2` 直接打开物理相机 | 没有其他进程占用 RealSense |
| `deploy_gui_gi.py` | GStreamer 共享内存 `fooA` | 兼容原 `..._gi_depthwidth_22.py` 实验系统 |
| `deploy_gui_legacy_gi.py` | GStreamer 共享内存 `fooA` | 旧 GUI 布局和 50 帧发送节奏，仅替换为 ToolRGS 内核 |

原 GI GUI 实际读取的是：

```text
/home/raico-hri/v1/kinova_rs_grasp/foo/fooA
```

它没有再次调用 `rs.pipeline.start()`。当上游程序已经占用 RealSense 时，应使用
GI 入口，否则直连入口可能报告 `RuntimeError: No device connected`。

## 2. 更新与依赖

```bash
cd /home/raico-hri/projects/ToolRGS
git pull origin main
pip install -r requirement.txt
pip install -r requirement-deploy.txt
```

GI/GStreamer 还需要系统包：

```bash
sudo apt install python3-gi gir1.2-gstreamer-1.0 \
  gstreamer1.0-tools gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad
```

## 3. 模型文件与自动下载

完整 GUI 使用两个本地权重：

```text
weights/drogoff_grasp_tools_v2_original300_best_j1.pth
weights/epoch_48_13.pth
```

第一个是抓取模型。若本地不存在，`checkpoint_url` 会从 GitHub Release 原子下载，
并使用 SHA256 校验。当前模型的 SHA256 是：

```text
38b2824b385a293b883bfe03682fec0e2f30c83457e3f86ba08e14d884fd7ecf
```

检查、自动下载并加载模型：

```bash
python tools/check_deployment.py \
  --config config/deployment/lab.yaml \
  --build-model --build-detector
```

目标检测权重 `weights/epoch_48_13.pth` 是实验室 MMEngine checkpoint，需单独放置。
检查文件：

```bash
ls -lh \
  weights/drogoff_grasp_tools_v2_original300_best_j1.pth \
  weights/epoch_48_13.pth
```

## 4. 区分两个 socket

### 视频共享内存 socket

```text
/home/raico-hri/v1/kinova_rs_grasp/foo/fooA
```

它只传输 1280x720 BGR 图像，不发送机械臂坐标。检查：

```bash
ls -l /home/raico-hri/v1/kinova_rs_grasp/foo/fooA
```

若不存在，先启动原实验系统的 RealSense/GStreamer 发布进程。

### 机械臂 TCP socket

```text
192.168.38.10:3000
```

它接收抓取坐标。旧 GI GUI 和 ToolRGS 使用相同地址与协议。

## 5. 安全预览：不发送机械臂坐标

不添加 `--allow-robot` 时，GUI 不会连接接收端，也不会发送坐标。

GI 共享流预览：

```bash
python deploy_gui_gi.py --config config/deployment/lab.yaml
```

旧版兼容布局预览：

```bash
python deploy_gui_legacy_gi.py --config config/deployment/lab.yaml
```

RealSense 直连预览：

```bash
python deploy_gui_realsense.py --config config/deployment/lab.yaml
```

没有摄像头时使用仓库样例图：

```bash
python deploy_gui.py \
  --config config/deployment/lab.yaml \
  --image \
  --prompt "the screwdriver"
```

## 6. 完整机械臂模式

启动前确认机械臂接收端、手眼标定、工作空间限制和急停均正常，并关闭仍在占用
TCP 连接的旧 GUI。

GI 版本：

```bash
python deploy_gui_gi.py \
  --config config/deployment/lab.yaml \
  --allow-robot
```

旧版兼容 GUI（推荐用于复现实验室原工作流）：

```bash
python deploy_gui_legacy_gi.py \
  --config config/deployment/lab.yaml \
  --allow-robot
```

该入口保留旧 GUI 的顶部三模式、两张 640x480 主图、三张 160x120 稠密图、
语言输入、GI `fooA` 视频和每 50 帧发送坐标；抓取网络、检测器、坐标/角度/
宽度映射均来自当前 ToolRGS。

RealSense 直连版本：

```bash
python deploy_gui_realsense.py \
  --config config/deployment/lab.yaml \
  --allow-robot
```

实验室配置中的 `robot.timeout_s: null` 与旧 GI GUI 一致：TCP 连接和发送均
不设置应用层超时。连接在后台线程等待，因此 Docker 接收程序尚未监听时，
GUI 仍可正常绘制和切换页面，不会再出现主线程阻塞导致的黑屏。

`config/deployment/lab.yaml` 当前启用 `auto_connect`、`auto_arm` 和 `auto_send`，
有效预测会按配置的间隔自动发送。若只想手动发送，将 YAML 中以下两项设为
`false`：

```yaml
robot:
  auto_arm: false
  auto_send: false
```

## 7. TCP 连接排查

先检查是否还有旧 GUI 占用连接：

```bash
pgrep -af "realsense_object_grasp|deploy_gui"
```

检查客户端到接收端端口：

```bash
nc -vz -w 5 192.168.38.10 3000
```

使用与 GUI 相同的 Python TCP 调用测试：

```bash
python -c "import socket; s=socket.create_connection(('192.168.38.10',3000),5); print('ROBOT CONNECTED'); s.close()"
```

检查路由与连通性：

```bash
ip route get 192.168.38.10
ping -c 3 192.168.38.10
```

在机械臂接收端检查端口监听：

```bash
ss -ltnp | grep :3000
```

常见结果：

- `Connection refused`：IP 可达，但接收程序没有监听 3000 端口。
- `timed out`：检查网线、网卡地址、防火墙和接收端电源。
- `No route to host`：当前机器没有到 `192.168.38.10` 的有效路由。
- 旧 GUI 正常、新 GUI失败：先完全退出旧 GUI；接收端可能只处理一个长连接。
- GUI 显示 `DRY RUN`：启动命令缺少 `--allow-robot`。

若接收端在 GUI 启动后才上线，可在 GUI 中点击 **Connect receiver** 重新连接。

## 8. 常见图像与 GUI 错误

### `RuntimeError: No device connected`

当前 RealSense 已被上游发布进程占用，或 SDK 没有发现物理设备。使用 GI 入口：

```bash
python deploy_gui_gi.py --config config/deployment/lab.yaml --allow-robot
```

### GI GUI 没有画面

```bash
ls -l /home/raico-hri/v1/kinova_rs_grasp/foo/fooA
```

确认上游发布进程正在运行，并确认流格式为 BGR 1280x720、30 FPS。

### Qt `xcb` 插件错误

确认 PyQt5 和系统 xcb 依赖可用。ToolRGS 启动时会优先选择当前 PyQt5 的
platform plugins，而不是 `cv2/qt/plugins`。

### Detector `weights_only` 错误

`lab.yaml` 已将实验室自己的 `epoch_48_13.pth` 标记为可信 checkpoint。
仅对自己训练或确认可信的权重启用该兼容模式。

## 9. 坐标协议

发送内容为一行 ASCII：

```text
{x, y, theta, width, depth}\n
```

默认契约：

- `x, y`：RealSense 原始 1280x720 图像坐标。
- `theta`：旧接收端使用的 `theta + 180`、`[0, 360)` 角度约定。
- `width`：像素夹爪宽度，当前可使用分割 mask span 策略。
- `depth`：旧系统的语义层级 `-1/0/1`，不是 RealSense 深度值。

GUI 中显示的命令预览与 TCP 实际发送使用同一个 `GraspCommand`，发送前会检查
`robot.limits`。默认边界：

```text
x: 0..1280
y: 0..720
theta: 0..360
width: 1..600
depth: -1..1
```

## 10. 停止与清理

正常关闭 GUI 会停止视频定时器、关闭 TCP socket，并释放直连 RealSense 或
GStreamer pipeline。若窗口异常退出，先检查残留进程：

```bash
pgrep -af "deploy_gui|realsense_object_grasp"
```

确认旧进程完全退出后再启动另一个入口，避免物理相机或 TCP 长连接冲突。

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
  --config config/deployment/lab.yaml
```

该命令不会打开相机、不会连接 TCP、不会发送机械臂指令。它会：

1. 检查 Python/Qt/MMDetection 环境；
2. 从配置的 GitHub Release 下载全部可选抓取权重（四个带统一测试分数的 V3 模型、DROG-OFF V2 与 aligned CROG V2）；
3. 下载缺失的 CLIP ViT-B/16、CLIP RN50、DINOv2 和 MambaVision-T 预训练权重；
4. 对配置了 Release URL 的权重做 SHA-256 校验；
5. 下载并校验 22 类 Grasp-Tools Faster R-CNN 目标检测权重。

主要文件应位于：

```text
weights/drogoff_grasp_tools_v2_original300_best_j1.pth
weights/crog_aligned_grasp_tools_v2_original300_best_j1.pth
weights/v3_drogoff_v2_grasp_tools_15k_original300_best_msr.pth
weights/v3_mambagrasp_grasp_tools_15k_unified_original300_best_j1.pth
weights/v3_crog_grasp_tools_15k_unified_original300_best_j1.pth
weights/v3_maplegrasp_stage2_grasp_tools_15k_unified_original300_best_j1.pth
weights/faster_rcnn_r50_fpn_grasp_tools_v2_best.pth
pretrain/ViT-B-16.pt
pretrain/RN50.pt
pretrain/dinov2_vitb14_reg4_pretrain.pth
pretrain/mambavision_tiny_1k.pth.tar
```

抓取、CLIP、DINO 和 22 类 Faster R-CNN 权重均已配置自动下载。普通 `check`
默认下载缺失文件并做 SHA-256 校验；只想离线检查而不下载时使用
`python tools/check_deployment.py --no-download-weights`。
如果目标路径已有旧 checkpoint，但 SHA-256 与当前 Release 不一致，`check` 会先
完整下载并校验新文件，再原子替换旧文件；下载失败时原文件保持不变。

权重齐全后做一次完整加载验证：

```bash
python tools/check_deployment.py \
  --config config/deployment/lab.yaml \
  --build-model --build-detector
```

看到 `Preflight completed with 0 failure(s).` 才进入正式实验。

### GUI 内切换模型和后处理

顶部设置按钮会随页面变化。进入 **Object Detection** 时显示
**Detection Post-processing**，进入 **Grasping Points Detection** 时显示
**Grasp Model & Post-processing**。抓取模型下拉框默认选中
`config/deployment/lab.yaml` 的 `active_model`，也可以直接切换已配置的
`V3-DROG-OFF-V2 (mSR@1 49.88%, J@1 58.94%)`、
`V3-MambaGrasp (mSR@1 53.62%, J@1 60.97%)`、
`V3-CROG (mSR@1 47.65%, J@1 59.38%)`、
`V3-MapleGrasp-Stage2 (mSR@1 43.44%, J@1 58.38%)`、DROG-OFF V2 与 aligned CROG V2。四个
V3 名称均以前缀 `V3-` 开头并排在列表前面；括号中的指标均来自统一 V3 的 5029 样本 test split。
切换模型只重载抓取网络，不会重启相机、检测器或
机械臂连接。

V3 模型的抓取质量/尺寸激活方式由训练配置决定，不能混用。GUI 不再按
模型名称强制指定激活：四个 V3 GUI profile 的质量与尺寸解码均使用 `auto`，
优先读取权重包中由训练配置写入的激活契约；旧权重缺少对应元数据时，回退到训练配置/模型契约。
DROG-OFF V2 解析为质量 `clamp`、宽度 `sigmoid`；MambaGrasp 和 MapleGrasp
Stage 2 均解析为质量/宽度 `sigmoid`；当前 CROG 解析为质量 `sigmoid`、宽度
`clamp`。因此不会因 GUI 名称或模型类别硬编码而错配训推激活。

推荐按以下顺序切换：

1. 进入 **Grasping Points Detection**，点击 **Grasp Model & Post-processing** 展开面板；
2. 点击 **GRASP MODEL** 右侧的模型框；
3. 在展开的深色列表中**单击一次**目标模型；
4. 状态从绿色 **READY** 变为黄色 **LOADING**，等待进度条消失并恢复
   **READY**；
5. 修改语言指令，然后点击 **Predict now** 或等待连续推理。

模型列表展开期间，GUI 会暂时停止大图和热力图重绘，避免 Linux/Qt 下列表闪烁、
跳动或选不中；列表关闭后画面自动恢复，摄像头/GI 数据源不会被关闭。关闭状态下
鼠标滚轮不会切换模型，只有真正点击列表条目才会触发加载。

加载权重在后台线程执行，期间模型框和 **Predict now** 会暂时锁定，但窗口、相机
与机械臂连接仍保持响应。不要连续重复点击；DINO/CLIP 大权重首次加载可能需要数秒。
加载失败时 GUI 会显示错误并自动回到原模型。GUI 内的选择只在本次进程生效，不会
改写 YAML；重启后仍使用 `active_model`。

Detection 和 Grasp 使用两套独立后处理。Detection 页的参数来自 `detector`：

- **Score threshold**：检测框最低置信度；
- **NMS IoU**：重叠框的 NMS IoU 阈值；
- **Max detections**：单帧最多保留的检测框数量。

这些值会直接更新 MMDetection 测试配置，并从下一帧检测开始生效。Grasp 页的
控件初始值来自当前模型 profile：

- **Gripper height**：抓取框短边，默认原图 `20 px`；
- **Use mask**：是否在后处理和结果叠加中启用分割 mask；
- **Mask threshold**：分割概率二值化阈值，默认 `0.35`；
- **Expand**：二值 mask 向外膨胀的原图像素半径，默认 `0 px`；
- **Filter grasp points**：同时过滤 mask 外的质量峰和 offset 后抓取中心。

关闭 **Use mask** 时，threshold、expand 和抓取点过滤会自动停用；mask 页面仍
保留二值分割结果，便于比较与调试。修改控件后下一次预测立即使用新参数，不会
写回 YAML。

抓取语言框采用自由输入：输入内容只去除首尾空格，然后原样交给模型，不会自动
添加 `Grasp` 或其他模板。空输入按回车不会报错，只会暂停连续推理。

顶部主题下拉框提供四套即时生效的配色：`Midnight Teal`、`Ocean Blue`、
`Violet Night` 和 `Graphite Amber`。本次选择只影响当前 GUI；若要指定启动默认值，
修改 `gui.theme`。

## 5. 一键启动

### A. 纯 RealSense 演示（无 GI、绝不发送坐标）

```bash
bash tools/gui_quickstart.sh demo --prompt "Grasp the screwdriver"
```

该入口会强制设置 `robot.enabled=false`，也不接受 `--allow-robot`，适合调模型、
目标检测和直接相机画面。

### B. GI 画面安全预览（不发送坐标）

```bash
bash tools/gui_quickstart.sh gi-preview --prompt "Grasp the screwdriver"
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
bash tools/gui_quickstart.sh robot --prompt "Grasp the screwdriver"
```

该命令等价于：

```bash
python deploy_gui_gi.py \
  --config config/deployment/lab.yaml \
  --allow-robot \
  --prompt "Grasp the screwdriver"
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

### D. 没有相机时用单张图片验证 GUI

使用仓库自带图片：

```bash
python deploy_gui.py \
  --config config/deployment/lab.yaml \
  --image \
  --prompt "Grasp the screwdriver"
```

使用自己的图片：

```bash
python deploy_gui.py \
  --config config/deployment/lab.yaml \
  --image /absolute/path/to/test.jpg \
  --prompt "Grasp the screwdriver"
```

单图模式自动关闭连续推理；进入 **Grasping Points Detection** 页面后点击
**Predict now**。模型切换、分割 mask、抓取框和目标检测页面仍可正常验证，但不要
添加 `--allow-robot`。

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

### 模型列表闪烁、跳动或选不中

先确认已经完全关闭旧 GUI，再更新并检查提交：

```bash
cd /home/raico-hri/projects/ToolRGS
git pull --ff-only origin main
git rev-parse --short HEAD
pgrep -af "deploy_gui|realsense_object_grasp"
```

包含提交 `91caf20` 及之后的版本会在模型列表展开时暂停预览重绘，并且只响应真实
点击。若仍在运行旧进程，即使代码已经更新，界面行为也不会变化；结束旧进程后重新
启动 GUI。

### 切换模型长时间停在 `LOADING`

- 不要重复点击模型列表，先观察 `nvidia-smi` 是否仍在加载权重；
- 运行 `bash tools/gui_quickstart.sh check` 检查两个 profile 的配置与权重；
- 显存不足时会自动回退到原模型并弹出错误，关闭其他 GPU 任务后重试；
- 如果整个窗口而非只有模型按钮失去响应，先确认当前提交不早于 `e5c7f4d`。

### Detector `weights_only` 错误

Faster R-CNN 权重来自本项目 3090 训练服务器并通过 Release SHA-256 校验。仓库只对
配置中明确标记为可信的这一条本地路径启用 MMEngine/PyTorch 2.6 兼容加载；不要对
来源不明的 checkpoint 启用 `trusted_checkpoint`。

## 8. 停止 GUI

正常关闭窗口会释放 RealSense/GStreamer 和 TCP socket。异常退出后检查残留：

```bash
pgrep -af "deploy_gui|realsense_object_grasp"
```

确认旧进程退出后再重新启动，避免相机或 TCP 长连接冲突。

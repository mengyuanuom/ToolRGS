# VCoT split metadata

The CSV files in this directory are the official VCoT split metadata for the
Grasp-Anything dataset:

- `train.csv`
- `test_seen.csv`
- `test_unseen.csv`

Each row stores `grasp_id, object_name, scene_description`. The images and
annotations are not copied into this repository; set `DATA.root_path` to the
local Grasp-Anything root containing `image/`, `positive_grasp/`, and `mask/`.

Source: [VCoT-Grasp](https://github.com/zhanghr2001/VCoT-Grasp) (MIT License).

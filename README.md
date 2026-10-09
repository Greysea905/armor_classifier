# armor_classifier — 装甲板数字/图案分类器训练

训练装甲板中心"数字/图案"分类模型(**7 类**),产出 ONNX 交给 C++ 主工程 `rmv_autoaim_lxy/detect/` 的 `ArmorClassifier`。

## 1. 类别(7 类)

| label(分类器输出) | target_id | 文件夹名(video/data 目录) |
|---|---|---|
| `1` / `2` / `3` / `4` | 1~4 | `1` / `2` / `3` / `4` |
| `outpost` | 6 | `6outpost` |
| `guard` | 7 | `7guard` |
| `base` | 8 | `8base` |

> 数字 `5` 和负样本 `9neg` 已弃用。无法分类/低置信度由 C++ 侧用置信度阈值处理(→ target_id 0)。
> 文件夹名 = target_id + label(如 `6` + `outpost` = `6outpost`);分类器输出的是 label(`outpost`/`guard`/`base`),不是文件夹名。

## 2. 数据生产(自拍视频 → 拉正灰度小图)

数据用海康相机拍摄。

1. 每类拍若干段视频:车停、人绕圈走,并录不同曝光(4/6/8/10ms),放到 `video/<类>/`;
2. `src/build_dataset.py` 抽帧(跳帧防过拟合)→ YOLO 检测 → 灯条匹配 → 透视拉正 → 裁数字区(灰度)→ 存 `data/<类>/`。

```bash
python3 src/build_dataset.py 7guard --step 20    # 处理单个类
python3 src/build_dataset.py --all --step 20     # 处理 video/ 下所有类
```

> 裁剪尺寸:小装甲板约 26×24、大装甲板约 48×24(和 C++ `ArmorClassifier` 的透视 + 裁切一致,保证训练/推理同分布)。

## 3. 切分 train/val

```bash
python3 src/split.py --src data/
```

按 0.85/0.15 切到 `datasets/train` + `datasets/val`(默认 7 类)。

## 4. 训练(云端 AutoDL)

```bash
python3 src/train.py --device 0
```

`config/train.yaml` 关键参数:

| 参数 | 值 | 说明 |
|---|---|---|
| model | yolo11n-cls.pt | 分类模型(nano) |
| imgsz | 64 | 数字小图 |
| epochs / patience | 50 / 15 | 验证集 15 轮无提升早停 |
| batch | 128 | 小图可大 batch |
| fliplr / flipud | 0 | 数字禁止翻转(6↔9 混淆) |
| hsv_h / hsv_s | 0 | 灰度图无色调 |

> 结果:7 类 top1_acc ≈ **99.8%**,约第 3 轮收敛。

## 5. 交给 C++

把 `runs/digit/weights/best.onnx` 复制为:

```
rmv_autoaim_lxy/detect/models/armor_classifier.onnx
```

## 6. 目录结构

```
armor_classifier/
├── config/train.yaml       # 训练配置
├── src/
│   ├── build_dataset.py    # 视频 -> 数字小图(抽帧+检测+拉正+裁切)
│   ├── split.py            # 切分 train/val
│   ├── train.py            # 训练 + 导出 ONNX
│   └── common.py           # 路径解析
├── video/<类>/             # 原始视频(不入库)
├── data/<类>/              # 生成的数字小图(不入库)
├── datasets/train+val/     # 切分后数据(不入库)
└── runs/                   # 训练输出(不入库)
```

> 大文件(`video/`、`data/`、`datasets/`、`runs/`、`*.pt`、`*.onnx`)已加入 `.gitignore`。

# armor_classifier — 装甲板数字分类器训练

训练数字识别模型(分类),识别装甲板编号 1~5,输出 `target_id`。产出 ONNX 交给 C++ 主工程 `rmv_autoaim_lxy/detect/`。

## 1. 数据来源

队友提供的数字分类数据集(20×28 小图,按类别分文件夹):
- `1` ~ `5` = 装甲板数字编号
- `6outpost` / `7guard` / `8base` = 目标类型(本任务不用)
- `9neg` = 负样本

**本任务只用 `1`~`5` 五个类**(识别 target_id)。

## 2. 目录结构

```
armor_classifier/
├── config/train.yaml   # 训练配置
├── src/
│   ├── common.py       # 路径解析
│   ├── split.py        # 切分 train/val
│   └── train.py        # 分类训练 + 导出 ONNX
├── datasets/           # 切分后的数据(train/val 各含 1-5 子目录)
└── runs/               # 训练输出
```

## 3. 快速开始(云端 AutoDL)

### 3.1 切分数据

```bash
python3 src/split.py --src /path/to/队友数字数据集 --classes 1,2,3,4,5
```

> 把队友数据集(含 1/2/3/4/5 子目录的那个)路径填进 `--src`,切分后生成 `datasets/train/{1..5}` 和 `datasets/val/{1..5}`。

### 3.2 训练

```bash
python3 src/train.py              # 用 train.yaml 默认(device 0)
python3 src/train.py --epochs 80 --batch 256
```

### 3.3 交给 C++

把 `runs/digit/weights/best.onnx` 复制到:
```
rmv_autoaim_lxy/detect/models/digit_cls.onnx
```

## 4. 关键参数

| 参数 | 值 | 说明 |
|---|---|---|
| model | yolo11n-cls.pt | 分类模型 |
| imgsz | 64 | 数字小图 |
| epochs | 50 | 收敛快 |
| batch | 128 | 小图 batch 可大 |

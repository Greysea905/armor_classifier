#!/usr/bin/env python3
"""数字分类器训练:读 config/train.yaml,用 ultralytics 分类模式训练并导出 ONNX。

前置:src/split.py 已把数据切分成 datasets/train 和 datasets/val。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import CONFIG_DIR, DATASETS_DIR, RUNS_DIR, load_yaml, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, default=CONFIG_DIR / "train.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.cfg)
    for key in ("epochs", "imgsz", "batch", "device"):
        v = getattr(args, key)
        if v is not None:
            cfg[key] = v

    model = load_model(cfg["model"])
    kwargs = {
        "data": str(DATASETS_DIR),
        "epochs": cfg["epochs"],
        "imgsz": cfg["imgsz"],
        "batch": cfg["batch"],
        "device": cfg["device"],
        "project": str(RUNS_DIR),
        "name": cfg.get("name", "digit"),
        "exist_ok": False,
    }
    for key in ("save_period", "patience", "fliplr", "flipud", "hsv_h", "hsv_s", "hsv_v", "amp"):
        if key in cfg:
            kwargs[key] = cfg[key]
    model.train(**kwargs)

    # 导出 ONNX(用实际保存目录,兼容 exist_ok 自动加后缀 digit-2 等)
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    load_model(best).export(format="onnx", imgsz=cfg["imgsz"])
    print(f"\n导出完成:{best.with_suffix('.onnx')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""把队友的目标分类数据集(按类别分好的文件夹)切分成 train/val。

用法:
    python3 src/split.py --src data/
默认 9 类(1-5 数字 + 6outpost/7guard/8base 目标类型 + 9neg 负样本),
按 0.85/0.15 切分到 datasets/train 和 datasets/val。
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from common import DATASETS_DIR, IMG_EXTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="队友数据集目录(含 1/2/3/4/5 等子目录)")
    ap.add_argument("--classes", type=str, default="1,2,3,4,6outpost,7guard,8base", help="要用的类别,逗号分隔")
    ap.add_argument("--ratio", type=float, default=0.85)
    ap.add_argument("--max-per-class", type=int, default=None, help="每类最多取多少张(平衡类别)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.split(",")]
    random.seed(args.seed)

    for cls in classes:
        src_dir = args.src / cls
        if not src_dir.is_dir():
            print(f"[跳过] 找不到 {src_dir}")
            continue
        imgs = sorted(p for p in src_dir.glob("*") if p.suffix.lower() in IMG_EXTS)
        random.shuffle(imgs)
        if args.max_per_class and len(imgs) > args.max_per_class:
            imgs = imgs[:args.max_per_class]
        n = len(imgs)
        n_train = int(n * args.ratio)
        for split, lst in (("train", imgs[:n_train]), ("val", imgs[n_train:])):
            dst = DATASETS_DIR / split / cls
            dst.mkdir(parents=True, exist_ok=True)
            for p in lst:
                shutil.copy2(p, dst / p.name)
        print(f"类别 {cls}: {n} 张 -> train {n_train} / val {n - n_train}")

    print(f"\n完成 -> {DATASETS_DIR}/train + val。可运行: python3 src/train.py")


if __name__ == "__main__":
    main()

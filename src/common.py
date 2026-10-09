"""路径解析 + 配置/模型加载"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"
RUNS_DIR = PROJECT_ROOT / "runs"
CONFIG_DIR = PROJECT_ROOT / "config"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_yaml(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(weights):
    from ultralytics import YOLO
    return YOLO(str(weights))

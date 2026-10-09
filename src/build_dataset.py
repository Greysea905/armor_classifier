#!/usr/bin/env python3
"""从视频批量生成数字分类训练数据(灰度 20x28 小图)。

流程:读 video/<类>/ 的视频 -> 抽帧(跳帧防过拟合) -> YOLO 检测装甲板 -> 灯条匹配
     -> 透视拉正 -> 裁中心数字区(灰度) -> 存 data/<类>/<类>_<序号>.png。

处理的是"数字/图案"(颜色无关),所以红蓝装甲板都会裁(检测到哪个颜色就用哪个颜色配对)。

用法:
    python3 src/build_dataset.py 7guard [--step 40] [--limit 99999]
    python3 src/build_dataset.py --all --step 40
"""
import argparse
import glob
import os

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, '..', 'rmv_autoaim_lxy', 'detect', 'models', 'armor_detector.onnx')

# ---- 灯条/配对参数(和 C++ LightBarExtractor / test_match.py 一致) ----
GRAY_THRES = 120
AREA_THRES = 50
RATIO_THRES = 2.2
MAX_TILT = 45
BOX_PAD = 0.10
PARALLEL_THRES = 15.0
LEN_RATIO_THRES = 0.7


def find_lights(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, GRAY_THRES, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lights = []
    for c in contours:
        if len(c) < 5:
            continue
        area = cv2.contourArea(c)
        if area < AREA_THRES:
            continue
        rr = cv2.minAreaRect(c)
        w, h = rr[1]
        if min(w, h) < 1.0:
            continue
        if max(w, h) / min(w, h) < RATIO_THRES:
            continue
        boxpts = cv2.boxPoints(rr)
        mid, lens = [], []
        for k in range(4):
            a, b = boxpts[k], boxpts[(k + 1) % 4]
            mid.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
            lens.append(float(np.hypot(b[0] - a[0], b[1] - a[1])))
        order = sorted(range(4), key=lambda k: lens[k])
        top, bottom = mid[order[0]], mid[order[1]]
        if top[1] > bottom[1]:
            top, bottom = bottom, top
        length = float(np.hypot(bottom[0] - top[0], bottom[1] - top[1]))
        tilt = abs(90 - abs(float(np.degrees(np.arctan2(bottom[1] - top[1], bottom[0] - top[0])))))
        if tilt > MAX_TILT:
            continue
        pts = c.reshape(-1, 2)
        sum_r = int(roi[pts[:, 1], pts[:, 0], 2].sum())
        sum_b = int(roi[pts[:, 1], pts[:, 0], 0].sum())
        color = 1 if sum_r > sum_b * 1.2 else (2 if sum_b > sum_r * 1.2 else 0)
        lights.append(dict(top=top, bottom=bottom,
                           center=((top[0] + bottom[0]) / 2, (top[1] + bottom[1]) / 2),
                           length=length, width=area / max(length, 1e-3),
                           tilt=tilt, color=color))
    return lights


def match_lights(lights, dc):
    target = [l for l in lights if l['color'] == dc]
    if len(target) < 2:
        return None
    target.sort(key=lambda l: l['width'] * l['length'], reverse=True)
    for i in range(len(target)):
        for j in range(i + 1, len(target)):
            a, b = target[i], target[j]
            left, right = (a, b) if a['center'][0] < b['center'][0] else (b, a)
            par = abs(left['tilt'] - right['tilt'])
            lr = min(left['length'], right['length']) / max(left['length'], right['length'])
            if par < PARALLEL_THRES and lr > LEN_RATIO_THRES:
                avg = (left['length'] + right['length']) / 2
                cd = float(np.hypot(left['center'][0] - right['center'][0],
                                    left['center'][1] - right['center'][1])) / avg
                return dict(size=1 if cd >= 3.0 else 0, left=left, right=right,
                            corners=[left['top'], right['top'], right['bottom'], left['bottom']])
    return None


def extract_number(img, corners, size):
    """透视拉正 + 裁中心 20x28 数字区,返回灰度图(和 C++ ArmorClassifier 一致)。"""
    w = 54 if size else 32
    H, L = 28, 12
    ty = (H - L) // 2 - 1
    by = ty + L
    src = np.array([corners[3], corners[0], corners[1], corners[2]], np.float32)
    dst = np.array([[0, by], [0, ty], [w - 1, ty], [w - 1, by]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, H))
    # 裁掉左右灯条(各约 3 列) + 上下各 2 行余量,保留完整中间图案
    roi = warped[2:H - 2, 3:w - 3]
    return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)


def process_frame(det, frame):
    """检测(红/蓝都试) -> 灯条匹配 -> 拉正,返回 20x28 灰度图或 None。"""
    rs = det(frame)
    if len(rs[0].boxes) == 0:
        return None
    for b in rs[0].boxes:
        dc = int(b.cls[0])  # 1=红 2=蓝,用检测到的颜色去配对
        if dc not in (1, 2):
            continue
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        pw, ph = int((x2 - x1) * BOX_PAD), int((y2 - y1) * BOX_PAD)
        x0, y0 = max(0, x1 - pw), max(0, y1 - ph)
        xx, yy = min(frame.shape[1], x2 + pw), min(frame.shape[0], y2 + ph)
        if xx <= x0 or yy <= y0:
            continue
        roi = frame[y0:yy, x0:xx]
        lights = find_lights(roi)
        m = match_lights(lights, dc)
        if m is None:
            continue
        c = [(p[0] + x0, p[1] + y0) for p in m['corners']]
        num = extract_number(frame, c, m['size'])
        return num, m['size']
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cls', nargs='?', help='类名,如 7guard;或用 --all')
    ap.add_argument('--all', action='store_true', help='处理 video/ 下所有类')
    ap.add_argument('--step', type=int, default=40, help='抽帧步长(高fps下跳多帧防过拟合)')
    ap.add_argument('--limit', type=int, default=99999, help='每类最多保存张数')
    args = ap.parse_args()

    if args.all:
        classes = sorted(os.listdir(os.path.join(ROOT, 'video')))
    elif args.cls:
        classes = [args.cls]
    else:
        ap.error('需要类名或 --all')

    det = YOLO(MODEL)
    for cls in classes:
        video_dir = os.path.join(ROOT, 'video', cls)
        out_dir = os.path.join(ROOT, 'data', cls)
        if not os.path.isdir(video_dir):
            print(f'[跳过] {cls}: 无 video/{cls}/')
            continue
        os.makedirs(out_dir, exist_ok=True)

        vids = sorted(glob.glob(os.path.join(video_dir, '*')))
        seq = 0
        saved = 0
        for v in vids:
            cap = cv2.VideoCapture(v)
            if not cap.isOpened():
                print(f'[跳过] 打不开 {v}')
                continue
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            idx = 0
            while True:
                if not cap.grab():
                    break
                if idx % args.step == 0:
                    ret, frame = cap.retrieve()
                    if not ret:
                        break
                    r = process_frame(det, frame)
                    if r is not None:
                        num, size = r
                        out = os.path.join(out_dir, f'{cls}_{seq:06d}.png')
                        cv2.imwrite(out, num)
                        seq += 1
                        saved += 1
                        if saved >= args.limit:
                            cap.release()
                            print(f'{cls}: 达到上限 {args.limit},停止')
                            return
                idx += 1
            cap.release()
            print(f'{cls}: {os.path.basename(v)} 处理完(累计 {saved} 张)')

        print(f'[完成] {cls}: 共 {saved} 张 -> {out_dir}/')


if __name__ == '__main__':
    main()

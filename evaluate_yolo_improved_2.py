"""
Evaluates a YOLO model against CARLA-generated ground-truth labels.

For each image the script computes TP/FP/FN using greedy IoU matching
(highest-confidence predictions matched first) and writes per-image
visualisations and a summary CSV.

Usage:
    python evaluate_yolo_improved_2.py --model yolo11m.pt --dataset dataset_Town01
"""
import os
import csv
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO


# ──────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────

def load_carla_labels(path, img_w, img_h):
    """Loads YOLO-format label file and converts to pixel coordinates."""
    labels = []
    if not os.path.exists(path):
        return np.array([])

    try:
        with open(path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if not parts:
                    continue
                cls, x, y, w, h = map(float, parts)
                xmin = int((x - w / 2.0) * img_w)
                ymin = int((y - h / 2.0) * img_h)
                xmax = int((x + w / 2.0) * img_w)
                ymax = int((y + h / 2.0) * img_h)
                labels.append([int(cls), xmin, ymin, xmax, ymax])
    except Exception as e:
        print(f"  ⚠️  Error reading {path}: {e}")
        return np.array([])

    return np.array(labels) if labels else np.array([])


def calculate_iou(box1, box2):
    """Calculates Intersection over Union between two (xmin, ymin, xmax, ymax) boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area  = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area  = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = float(box1_area + box2_area - inter_area)

    if union_area == 0:
        return 0
    return inter_area / union_area


def draw_evaluation_image(img, pred_boxes, pred_classes, pred_confs, gt_boxes,
                          matched_predictions, matched_gts, class_names, iou_threshold):
    """
    Draws predictions and ground truth on the image with colour coding:
      Green  = True Positive  (matched prediction)
      Red    = False Positive (unmatched prediction)
      Blue   = False Negative (unmatched ground truth)
    """
    draw_img = img.copy()

    for j, g_box in enumerate(gt_boxes):
        g_cls = int(g_box[0])
        xmin, ymin, xmax, ymax = map(int, g_box[1:])
        thickness = 3 if j in matched_gts else 2
        cv2.rectangle(draw_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), thickness)
        cv2.putText(draw_img, f"GT {class_names.get(g_cls, g_cls)}",
                    (xmin, max(0, ymin - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    for i, p_box in enumerate(pred_boxes):
        p_cls = int(pred_classes[i])
        conf  = pred_confs[i]
        xmin, ymin, xmax, ymax = map(int, p_box)
        is_tp     = i in matched_predictions
        color     = (0, 255, 0) if is_tp else (0, 0, 255)
        thickness = 3 if is_tp else 2
        cv2.rectangle(draw_img, (xmin, ymin), (xmax, ymax), color, thickness)
        cv2.putText(draw_img, f"{class_names.get(p_cls, p_cls)}: {conf:.2f}",
                    (xmin, max(0, ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return draw_img


# ──────────────────────────────────────────────────────────────
# MAIN EVALUATION FUNCTION
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO on a CARLA dataset")
    parser.add_argument("--model",   type=str,   default="yolo11m.pt",
                        help="YOLO model file (default: yolo11m.pt)")
    parser.add_argument("--dataset", type=str,   default="dataset_default",
                        help="Dataset folder with pictures/ and labels/ subdirs")
    parser.add_argument("--conf",    type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou",     type=float, default=0.50,
                        help="IoU threshold for TP/FP matching (default: 0.50)")
    parser.add_argument("--classes", type=int, nargs='+', default=None,
                        help="Class IDs to evaluate (e.g. --classes 0 2). Default: all")
    args = parser.parse_args()

    IMAGES_DIR  = os.path.join(args.dataset, 'pictures')
    LABELS_DIR  = os.path.join(args.dataset, 'labels')
    OUTPUT_DIR  = os.path.join(args.dataset, 'yolo_predictions')
    RESULTS_CSV = os.path.join(args.dataset, f'evaluation_results_{Path(args.model).stem}.csv')

    if not os.path.exists(IMAGES_DIR):
        print(f"❌ Error: {IMAGES_DIR} not found")
        return
    if not os.path.exists(LABELS_DIR):
        print(f"⚠️  Warning: {LABELS_DIR} not found, treating all frames as having no labels")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    CLASS_NAMES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

    print(f"🔄 Loading model: {args.model}")
    model = YOLO(args.model)

    image_files = sorted([f for f in os.listdir(IMAGES_DIR)
                          if f.endswith(('.jpg', '.jpeg', '.png'))])

    total_tp = 0
    total_fp = 0
    total_gt = 0
    per_class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'gt': 0})
    csv_rows = []

    print(f"🚀 Evaluating {len(image_files)} images...")
    if args.classes is not None:
        print(f"🔍 Filtering to classes: {args.classes}")
    print(f"📂 Saving visualisations to: {OUTPUT_DIR}")
    print(f"💾 Saving CSV to: {RESULTS_CSV}\n")

    for img_idx, img_name in enumerate(image_files):
        img_path   = os.path.join(IMAGES_DIR, img_name)
        label_path = os.path.join(LABELS_DIR, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        img = cv2.imread(img_path)
        if img is None:
            print(f"  ⚠️  Could not read {img_path}, skipping")
            continue

        img_h, img_w, _ = img.shape
        gt_boxes = load_carla_labels(label_path, img_w, img_h)

        if args.classes is not None and len(gt_boxes) > 0:
            gt_boxes = gt_boxes[np.isin(gt_boxes[:, 0].astype(int), args.classes)]

        img_gt_count = len(gt_boxes)
        total_gt += img_gt_count

        results = model.predict(img, conf=args.conf, classes=args.classes, verbose=False)[0]

        if len(results.boxes) == 0:
            cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), img)
            csv_rows.append([img_name, 0, 0, img_gt_count, img_gt_count, 0.0, 0.0])
            if (img_idx + 1) % 50 == 0:
                print(f"  Processed {img_idx + 1}/{len(image_files)}")
            continue

        pred_boxes   = results.boxes.xyxy.cpu().numpy()
        pred_classes = results.boxes.cls.cpu().numpy()
        pred_confs   = results.boxes.conf.cpu().numpy()

        # Match highest-confidence predictions first to avoid penalising the model
        # for detecting the same object twice when a lower-confidence duplicate arrives.
        sorted_indices = np.argsort(pred_confs)[::-1]
        pred_boxes     = pred_boxes[sorted_indices]
        pred_classes   = pred_classes[sorted_indices]
        pred_confs     = pred_confs[sorted_indices]

        matched_gt_indices   = set()
        matched_pred_indices = set()

        for i, p_box in enumerate(pred_boxes):
            p_cls    = int(pred_classes[i])
            best_iou     = 0
            best_gt_idx  = -1

            for j, g_box in enumerate(gt_boxes):
                if len(g_box) == 0:
                    continue
                if int(g_box[0]) == p_cls:
                    iou = calculate_iou(p_box, g_box[1:])
                    if iou > best_iou:
                        best_iou    = iou
                        best_gt_idx = j

            if best_iou >= args.iou and best_gt_idx not in matched_gt_indices:
                total_tp += 1
                per_class_stats[p_cls]['tp'] += 1
                matched_gt_indices.add(best_gt_idx)
                matched_pred_indices.add(i)
            else:
                total_fp += 1
                per_class_stats[p_cls]['fp'] += 1

        for g_box in gt_boxes:
            per_class_stats[int(g_box[0])]['gt'] += 1

        draw_img = draw_evaluation_image(img, pred_boxes, pred_classes, pred_confs,
                                         gt_boxes, matched_pred_indices, matched_gt_indices,
                                         CLASS_NAMES, args.iou)
        cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), draw_img)

        img_tp        = len(matched_pred_indices)
        img_fp        = len(pred_boxes) - img_tp
        img_fn        = img_gt_count - img_tp
        img_precision = img_tp / (img_tp + img_fp) if (img_tp + img_fp) > 0 else 0
        img_recall    = img_tp / img_gt_count       if img_gt_count > 0       else 0

        csv_rows.append([img_name, img_tp, img_fp, img_gt_count, img_fn,
                         img_precision, img_recall])

        if (img_idx + 1) % 50 == 0:
            print(f"  Processed {img_idx + 1}/{len(image_files)}")

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall    = total_tp / (total_gt + 1e-6)
    f1_score  = 2 * (precision * recall) / (precision + recall + 1e-6)

    print("\n" + "=" * 70)
    print("🎯 EVALUATION RESULTS")
    print("=" * 70)
    print(f"Model:                    {args.model}")
    print(f"Dataset:                  {args.dataset}")
    print(f"Confidence threshold:     {args.conf}")
    print(f"IoU threshold:            {args.iou}")
    if args.classes:
        print(f"Filtered classes:         {args.classes}")
    print("-" * 70)
    print(f"Total ground truth boxes: {total_gt}")
    print(f"True Positives  (TP):     {total_tp}")
    print(f"False Positives (FP):     {total_fp}")
    print(f"False Negatives (FN):     {max(0, total_gt - total_tp)}")
    print("-" * 70)
    print(f"Precision:                {precision * 100:6.2f}%")
    print(f"Recall:                   {recall    * 100:6.2f}%")
    print(f"F1-Score:                 {f1_score:6.4f}")
    print("=" * 70)

    print("\n📊 PER-CLASS METRICS:")
    print("-" * 70)
    print(f"{'Class':<15} {'TP':>6} {'FP':>6} {'GT':>6} {'Precision':>12} {'Recall':>12}")
    print("-" * 70)
    for cls_id in sorted(per_class_stats.keys()):
        s = per_class_stats[cls_id]
        cls_precision = s['tp'] / (s['tp'] + s['fp'] + 1e-6)
        cls_recall    = s['tp'] / (s['gt'] + 1e-6)
        print(f"{CLASS_NAMES.get(cls_id, f'Class {cls_id}'):<15} {s['tp']:>6} {s['fp']:>6} "
              f"{s['gt']:>6} {cls_precision * 100:>11.2f}% {cls_recall * 100:>11.2f}%")
    print("=" * 70)

    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'TP', 'FP', 'GT', 'FN', 'Precision', 'Recall'])
        writer.writerows(csv_rows)

    print(f"\n✅ Results saved to: {RESULTS_CSV}")
    print(f"✅ Visualisations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

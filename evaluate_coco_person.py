"""
evaluate_coco_person_sweep.py
Evaluates pretrained Faster R-CNN (COCO) on COCO val2017 person category.
Reports Precision / Recall / F1 and AP@0.5 across multiple thresholds.
"""

import os
import argparse
import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0.0, (boxA[2] - boxA[0])) * max(0.0, (boxA[3] - boxA[1]))
    boxBArea = max(0.0, (boxB[2] - boxB[0])) * max(0.0, (boxB[3] - boxB[1]))
    denom = boxAArea + boxBArea - interArea
    if denom <= 0:
        return 0.0
    return interArea / denom


def evaluate(images_dir, ann_file, device, max_images=None, iou_thres=0.5, score_thres_list=[0.3,0.5,0.7]):
    #Load COCO
    coco = COCO(ann_file)
    catIds = coco.getCatIds(catNms=["person"])
    imgIds_all = coco.getImgIds(catIds=catIds)
    imgIds = imgIds_all[:max_images] if max_images else imgIds_all

    #Model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval().to(device)

    #For AP calculation
    all_precisions = {}
    all_recalls = {}

    for score_thres in score_thres_list:
        TP, FP, FN = 0, 0, 0
        total_gt, total_preds = 0, 0
        precisions, recalls = [], []

        for imgId in tqdm(imgIds, desc=f"Images (thr={score_thres})"):
            img_info = coco.loadImgs(imgId)[0]
            img_path = os.path.join(images_dir, img_info["file_name"])
            if not os.path.exists(img_path):
                continue
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            annIds = coco.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
            gt_boxes = []
            for a in anns:
                gt = xywh_to_xyxy(a["bbox"])
                if (gt[2] - gt[0]) > 1 and (gt[3] - gt[1]) > 1:
                    gt_boxes.append(gt)
            total_gt += len(gt_boxes)
            gt_matched = [False] * len(gt_boxes)

            tensor = F.to_tensor(img_rgb).to(device)
            with torch.no_grad():
                outputs = model([tensor])[0]

            preds = []
            for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
                if int(label) == 1 and float(score) >= score_thres:
                    preds.append((box.cpu().numpy().tolist(), float(score)))

            preds.sort(key=lambda x: x[1], reverse=True)
            total_preds += len(preds)

            for pred_box, _ in preds:
                best_iou, best_idx = 0.0, -1
                for i, gt in enumerate(gt_boxes):
                    if gt_matched[i]:
                        continue
                    iou = compute_iou(pred_box, gt)
                    if iou > best_iou:
                        best_iou, best_idx = iou, i
                if best_iou >= iou_thres and best_idx >= 0:
                    TP += 1
                    gt_matched[best_idx] = True
                else:
                    FP += 1

            for matched in gt_matched:
                if not matched:
                    FN += 1

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            precisions.append(precision)
            recalls.append(recall)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        ap50 = np.mean([p for p in precisions])  #simplified AP@0.5

        print(f"\n=== Results (score ≥ {score_thres}) ===")
        print(f"Images evaluated: {len(imgIds)}")
        print(f"Total GT: {total_gt}")
        print(f"Total Pred: {total_preds}")
        print(f"TP: {TP}  FP: {FP}  FN: {FN}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(f"AP@0.5:    {ap50:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, default="val2017", help="path to COCO val2017 images folder")
    parser.add_argument("--ann_file", type=str, default="annotations/instances_val2017.json", help="path to COCO instances json")
    parser.add_argument("--max_images", type=int, default=200, help="limit number of images to evaluate (use None for all)")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="IoU threshold to count a match")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate(args.images_dir, args.ann_file, device, max_images=args.max_images, iou_thres=args.iou_thres)

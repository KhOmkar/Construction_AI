"""
Extract worker crops (compliant, no_helmet, no_vest, both_missing)
from scene images and save as compressed .npz files per split.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from src.data_prep.caption_generator import generate_crop_caption


def load_config(config_path="configs/dataset.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def box_iou(b1, b2):
    # b = [x1, y1, x2, y2]
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def make_square_box(x1, y1, x2, y2, img_w, img_h, margin_ratio=0.10):
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    side = max(bw, bh) * (1.0 + margin_ratio)
    half = side / 2.0

    nx1 = max(0, int(cx - half))
    ny1 = max(0, int(cy - half))
    nx2 = min(img_w, int(cx + half))
    ny2 = min(img_h, int(cy + half))

    return nx1, ny1, nx2, ny2


def extract_worker_crops_from_image(img, label_path, min_size=24):
    """
    Extracts worker crops and assigns fine-grained compliance class:
    0: compliant (helmet + vest)
    1: no_helmet violation
    2: no_vest violation
    3: both_missing violation
    """
    w, h = img.size
    if not os.path.exists(label_path):
        return []

    boxes = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])
                x1 = (xc - bw / 2.0) * w
                y1 = (yc - bh / 2.0) * h
                x2 = (xc + bw / 2.0) * w
                y2 = (yc + bh / 2.0) * h
                boxes.append({
                    "cls": cls_id,
                    "box": (x1, y1, x2, y2),
                    "xc": xc * w,
                    "yc": yc * h,
                    "bw": bw * w,
                    "bh": bh * h
                })

    if not boxes:
        return []

    crops = []
    person_boxes = [b for b in boxes if b["cls"] == 4]
    ppe_boxes = [b for b in boxes if b["cls"] != 4]

    if person_boxes:
        for pb in person_boxes:
            px1, py1, px2, py2 = pb["box"]
            if (px2 - px1) < min_size or (py2 - py1) < min_size:
                continue

            # Find PPE boxes inside person
            inside_ppe = []
            for b in ppe_boxes:
                b_cx = b["xc"]
                b_cy = b["yc"]
                # Center is inside person box (with 10% slack)
                slack_x = (px2 - px1) * 0.1
                slack_y = (py2 - py1) * 0.1
                if (px1 - slack_x) <= b_cx <= (px2 + slack_x) and (py1 - slack_y) <= b_cy <= (py2 + slack_y):
                    inside_ppe.append(b["cls"])

            has_helmet = 0 in inside_ppe
            has_no_helmet = 1 in inside_ppe
            has_vest = 2 in inside_ppe
            has_no_vest = 3 in inside_ppe

            # Determine crop class
            if has_no_helmet and has_no_vest:
                cls_tag = 3  # both_missing
            elif has_no_helmet:
                cls_tag = 1  # no_helmet
            elif has_no_vest:
                cls_tag = 2  # no_vest
            elif has_helmet or has_vest:
                cls_tag = 0  # compliant
            else:
                continue  # Worker with no detected equipment/violation

            # Crop person upper body / worker
            # Focus on upper 70% of person box to capture head + torso prominently
            crop_y2 = py1 + (py2 - py1) * 0.75
            sq_x1, sq_y1, sq_x2, sq_y2 = make_square_box(px1, py1, px2, crop_y2, w, h)
            if (sq_x2 - sq_x1) >= min_size and (sq_y2 - sq_y1) >= min_size:
                crop_im = img.crop((sq_x1, sq_y1, sq_x2, sq_y2))
                crops.append({
                    "image": crop_im,
                    "class": cls_tag,
                    "norm_box": (sq_x1 / w, sq_y1 / h, sq_x2 / w, sq_y2 / h)
                })

    else:
        # No person boxes; group vests and helmets
        vest_boxes = [b for b in ppe_boxes if b["cls"] in (2, 3)]
        used_ppe = set()

        for vb in vest_boxes:
            vx1, vy1, vx2, vy2 = vb["box"]
            v_cls = vb["cls"]  # 2: vest, 3: no_vest

            # Look for helmet/no_helmet directly above vest
            head_candidates = [
                b for b in ppe_boxes
                if b["cls"] in (0, 1) and b["yc"] < vb["yc"] and abs(b["xc"] - vb["xc"]) < (vb["bw"] * 0.8)
            ]

            if head_candidates:
                # Closest candidate above
                hb = min(head_candidates, key=lambda b: abs(vb["yc"] - b["yc"]))
                hx1, hy1, hx2, hy2 = hb["box"]
                h_cls = hb["cls"]  # 0: helmet, 1: no_helmet

                used_ppe.add(id(vb))
                used_ppe.add(id(hb))

                if h_cls == 1 and v_cls == 3:
                    cls_tag = 3
                elif h_cls == 1:
                    cls_tag = 1
                elif v_cls == 3:
                    cls_tag = 2
                else:
                    cls_tag = 0

                ux1 = min(vx1, hx1)
                uy1 = min(vy1, hy1)
                ux2 = max(vx2, hx2)
                uy2 = max(vy2, hy2)

                sq_x1, sq_y1, sq_x2, sq_y2 = make_square_box(ux1, uy1, ux2, uy2, w, h)
                if (sq_x2 - sq_x1) >= min_size and (sq_y2 - sq_y1) >= min_size:
                    crop_im = img.crop((sq_x1, sq_y1, sq_x2, sq_y2))
                    crops.append({
                        "image": crop_im,
                        "class": cls_tag,
                        "norm_box": (sq_x1 / w, sq_y1 / h, sq_x2 / w, sq_y2 / h)
                    })

            else:
                # Isolated vest / no_vest: expand upward by 45% of height to include head
                used_ppe.add(id(vb))
                expanded_y1 = max(0, vy1 - vb["bh"] * 0.45)
                cls_tag = 2 if v_cls == 3 else 0

                sq_x1, sq_y1, sq_x2, sq_y2 = make_square_box(vx1, expanded_y1, vx2, vy2, w, h)
                if (sq_x2 - sq_x1) >= min_size and (sq_y2 - sq_y1) >= min_size:
                    crop_im = img.crop((sq_x1, sq_y1, sq_x2, sq_y2))
                    crops.append({
                        "image": crop_im,
                        "class": cls_tag,
                        "norm_box": (sq_x1 / w, sq_y1 / h, sq_x2 / w, sq_y2 / h)
                    })

        # Process remaining unused head/helmet boxes
        head_boxes = [b for b in ppe_boxes if b["cls"] in (0, 1) and id(b) not in used_ppe]
        for hb in head_boxes:
            hx1, hy1, hx2, hy2 = hb["box"]
            h_cls = hb["cls"]
            cls_tag = 1 if h_cls == 1 else 0

            # Expand downward by 120% of head height to include neck/torso
            expanded_y2 = min(h, hy2 + hb["bh"] * 1.2)
            sq_x1, sq_y1, sq_x2, sq_y2 = make_square_box(hx1, hy1, hx2, expanded_y2, w, h)
            if (sq_x2 - sq_x1) >= min_size and (sq_y2 - sq_y1) >= min_size:
                crop_im = img.crop((sq_x1, sq_y1, sq_x2, sq_y2))
                crops.append({
                    "image": crop_im,
                    "class": cls_tag,
                    "norm_box": (sq_x1 / w, sq_y1 / h, sq_x2 / w, sq_y2 / h)
                })

    return crops


def process_split_crops(df_split, split_name, cfg):
    min_size = cfg["processing"].get("min_crop_size", 24)
    res_list = cfg["processing"].get("crop_resolutions", [64])

    print(f"\n--> Extracting crops for split '{split_name}' ({len(df_split)} images)...")

    crops_64 = []
    crops_128 = []
    labels = []
    captions = []
    image_ids = []
    boxes = []

    need_128 = 128 in res_list

    for idx, row in tqdm(df_split.iterrows(), total=len(df_split)):
        img_path = row["final_image_path"]
        lbl_path = row["final_label_path"]
        img_id = row["image_id"]

        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                extracted = extract_worker_crops_from_image(im, lbl_path, min_size=min_size)
                for item in extracted:
                    c_im = item["image"]
                    c_cls = item["class"]
                    n_box = item["norm_box"]

                    # 64x64
                    c64 = c_im.resize((64, 64), Image.Resampling.BICUBIC)
                    crops_64.append(np.array(c64, dtype=np.uint8))

                    if need_128:
                        c128 = c_im.resize((128, 128), Image.Resampling.BICUBIC)
                        crops_128.append(np.array(c128, dtype=np.uint8))

                    labels.append(c_cls)
                    captions.append(generate_crop_caption(c_cls))
                    image_ids.append(img_id)
                    boxes.append(n_box)
        except Exception as e:
            continue

    if not crops_64:
        print(f"Warning: No crops extracted for split {split_name}")
        return

    crops_dir = cfg["paths"]["crops_dir"]
    os.makedirs(crops_dir, exist_ok=True)

    arr_64 = np.stack(crops_64, axis=0)
    arr_labels = np.array(labels, dtype=np.int64)
    arr_captions = np.array(captions, dtype=object)
    arr_ids = np.array(image_ids, dtype=object)
    arr_boxes = np.array(boxes, dtype=np.float32)

    out_file_64 = os.path.join(crops_dir, f"crops_{split_name}_64.npz")
    np.savez_compressed(
        out_file_64,
        images=arr_64,
        labels=arr_labels,
        captions=arr_captions,
        image_ids=arr_ids,
        boxes=arr_boxes
    )
    print(f"Saved {len(arr_64)} crops (64x64) to {out_file_64}")

    if need_128 and crops_128:
        arr_128 = np.stack(crops_128, axis=0)
        out_file_128 = os.path.join(crops_dir, f"crops_{split_name}_128.npz")
        np.savez_compressed(
            out_file_128,
            images=arr_128,
            labels=arr_labels,
            captions=arr_captions,
            image_ids=arr_ids,
            boxes=arr_boxes
        )
        print(f"Saved {len(arr_128)} crops (128x128) to {out_file_128}")


def main():
    cfg = load_config()
    split_manifest = os.path.join(cfg["paths"]["processed_dir"], "split_manifest.csv")
    if not os.path.exists(split_manifest):
        print(f"Error: {split_manifest} not found. Run split_data first.")
        return

    df = pd.read_csv(split_manifest)

    for split in ["train", "val", "test", "final_test"]:
        sub = df[df["split"] == split]
        if len(sub) > 0:
            process_split_crops(sub, split, cfg)


if __name__ == "__main__":
    main()

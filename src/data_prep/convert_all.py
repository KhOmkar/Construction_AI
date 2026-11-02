"""
Convert all raw datasets in data/ into a unified YOLO dataset under processed/scenes/
and build the master manifest.csv.
"""

import os
import sys
import zipfile
import csv
import json
import xml.etree.ElementTree as ET
from PIL import Image
import yaml
import pandas as pd
from tqdm import tqdm


def load_config(config_path="configs/dataset.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_box(x_center, y_center, w, h, img_w, img_h, is_pixel=True):
    if is_pixel:
        xc = x_center / img_w
        yc = y_center / img_h
        nw = w / img_w
        nh = h / img_h
    else:
        xc, yc, nw, nh = x_center, y_center, w, h

    # Clamp to [0, 1]
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))

    if nw <= 0.001 or nh <= 0.001:
        return None
    return xc, yc, nw, nh


def process_ppe_2286y(cfg, out_img_dir, out_lbl_dir):
    source_cfg = cfg["sources"]["ppe_2286y"]
    zip_path = os.path.join(cfg["paths"]["raw_dir"], source_cfg["archive"])
    class_map = {int(k): int(v) for k, v in source_cfg["class_map"].items()}
    records = []

    print(f"--> Processing {source_cfg['archive']}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        # Build mapping of labels
        label_files = {n: n for n in z.namelist() if "/labels/" in n and n.endswith(".txt")}
        for n in z.namelist():
            if "/images/" in n and n.lower().endswith((".jpg", ".png", ".jpeg")):
                parts = n.split("/")
                orig_split = parts[1]  # train or valid
                fname = parts[-1]
                stem = os.path.splitext(fname)[0]
                img_id = f"ppe2286_{orig_split}_{stem}"

                # Read image data
                img_data = z.read(n)
                out_img_path = os.path.join(out_img_dir, f"{img_id}.jpg")
                with open(out_img_path, "wb") as f_out:
                    f_out.write(img_data)

                # Get size
                with Image.open(out_img_path) as im:
                    w, h = im.size

                # Read corresponding label
                lbl_archive_name = n.replace("/images/", "/labels/").replace(os.path.splitext(n)[1], ".txt")
                target_boxes = []
                if lbl_archive_name in label_files:
                    lines = z.read(lbl_archive_name).decode("utf-8", errors="ignore").splitlines()
                    for line in lines:
                        parts_l = line.strip().split()
                        if len(parts_l) >= 5:
                            src_cls = int(parts_l[0])
                            if src_cls in class_map:
                                tgt_cls = class_map[src_cls]
                                xc, yc, bw, bh = map(float, parts_l[1:5])
                                norm = normalize_box(xc, yc, bw, bh, w, h, is_pixel=False)
                                if norm:
                                    target_boxes.append((tgt_cls, *norm))

                out_lbl_path = os.path.join(out_lbl_dir, f"{img_id}.txt")
                with open(out_lbl_path, "w", encoding="utf-8") as f_lbl:
                    for b in target_boxes:
                        f_lbl.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

                records.append({
                    "image_id": img_id,
                    "source_dataset": "ppe_2286y",
                    "original_split": orig_split,
                    "original_path": n,
                    "image_path": out_img_path,
                    "label_path": out_lbl_path,
                    "width": w,
                    "height": h,
                    "num_target_boxes": len(target_boxes),
                    "box_classes": ",".join(str(b[0]) for b in target_boxes)
                })

    return records


def process_safety_vests_csv(cfg, out_img_dir, out_lbl_dir):
    source_cfg = cfg["sources"]["archive_safety_vests"]
    zip_path = os.path.join(cfg["paths"]["raw_dir"], source_cfg["archive"])
    class_map = source_cfg["class_map"]
    records = []

    print(f"--> Processing {source_cfg['archive']} (Safety Vests VOC CSV)...")
    with zipfile.ZipFile(zip_path, "r") as z:
        for split in ["train", "valid", "test"]:
            csv_path = f"{split}/{split}/_annotations.csv"
            if csv_path not in z.namelist():
                continue

            csv_text = z.read(csv_path).decode("utf-8")
            reader = csv.DictReader(csv_text.splitlines())
            boxes_by_image = {}
            for row in reader:
                fn = row["filename"]
                if fn not in boxes_by_image:
                    boxes_by_image[fn] = []
                cls_name = row["class"]
                if cls_name in class_map:
                    tgt_cls = class_map[cls_name]
                    iw = float(row["width"])
                    ih = float(row["height"])
                    xmin = float(row["xmin"])
                    ymin = float(row["ymin"])
                    xmax = float(row["xmax"])
                    ymax = float(row["ymax"])
                    xc = (xmin + xmax) / 2.0
                    yc = (ymin + ymax) / 2.0
                    bw = xmax - xmin
                    bh = ymax - ymin
                    norm = normalize_box(xc, yc, bw, bh, iw, ih, is_pixel=True)
                    if norm:
                        boxes_by_image[fn].append((tgt_cls, *norm))

            # Now save all images in this split
            for fn, boxes in boxes_by_image.items():
                arc_img_path = f"{split}/{split}/{fn}"
                if arc_img_path in z.namelist():
                    stem = os.path.splitext(fn)[0]
                    img_id = f"safetyvest_{split}_{stem}"
                    out_img_path = os.path.join(out_img_dir, f"{img_id}.jpg")
                    with open(out_img_path, "wb") as f_out:
                        f_out.write(z.read(arc_img_path))

                    with Image.open(out_img_path) as im:
                        w, h = im.size

                    out_lbl_path = os.path.join(out_lbl_dir, f"{img_id}.txt")
                    with open(out_lbl_path, "w", encoding="utf-8") as f_lbl:
                        for b in boxes:
                            f_lbl.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

                    records.append({
                        "image_id": img_id,
                        "source_dataset": "safety_vests",
                        "original_split": split,
                        "original_path": arc_img_path,
                        "image_path": out_img_path,
                        "label_path": out_lbl_path,
                        "width": w,
                        "height": h,
                        "num_target_boxes": len(boxes),
                        "box_classes": ",".join(str(b[0]) for b in boxes)
                    })

    return records


def process_css_data(cfg, out_img_dir, out_lbl_dir):
    source_cfg = cfg["sources"]["css_data"]
    zip_path = os.path.join(cfg["paths"]["raw_dir"], source_cfg["archive"])
    class_map = {int(k): int(v) for k, v in source_cfg["class_map"].items()}
    records = []

    print(f"--> Processing {source_cfg['archive']} (Construction Site Safety)...")
    with zipfile.ZipFile(zip_path, "r") as z:
        label_files = {n: n for n in z.namelist() if "/labels/" in n and n.endswith(".txt")}
        for n in z.namelist():
            if "/images/" in n and n.lower().endswith((".jpg", ".png", ".jpeg")):
                parts = n.split("/")
                orig_split = parts[0]  # train, valid, test
                fname = parts[-1]
                stem = os.path.splitext(fname)[0]
                img_id = f"css_{orig_split}_{stem}"

                img_data = z.read(n)
                out_img_path = os.path.join(out_img_dir, f"{img_id}.jpg")
                with open(out_img_path, "wb") as f_out:
                    f_out.write(img_data)

                with Image.open(out_img_path) as im:
                    w, h = im.size

                lbl_archive_name = n.replace("/images/", "/labels/").replace(os.path.splitext(n)[1], ".txt")
                target_boxes = []
                if lbl_archive_name in label_files:
                    lines = z.read(lbl_archive_name).decode("utf-8", errors="ignore").splitlines()
                    for line in lines:
                        parts_l = line.strip().split()
                        if len(parts_l) >= 5:
                            src_cls = int(parts_l[0])
                            if src_cls in class_map:
                                tgt_cls = class_map[src_cls]
                                xc, yc, bw, bh = map(float, parts_l[1:5])
                                norm = normalize_box(xc, yc, bw, bh, w, h, is_pixel=False)
                                if norm:
                                    target_boxes.append((tgt_cls, *norm))

                out_lbl_path = os.path.join(out_lbl_dir, f"{img_id}.txt")
                with open(out_lbl_path, "w", encoding="utf-8") as f_lbl:
                    for b in target_boxes:
                        f_lbl.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

                records.append({
                    "image_id": img_id,
                    "source_dataset": "css_data",
                    "original_split": orig_split,
                    "original_path": n,
                    "image_path": out_img_path,
                    "label_path": out_lbl_path,
                    "width": w,
                    "height": h,
                    "num_target_boxes": len(target_boxes),
                    "box_classes": ",".join(str(b[0]) for b in target_boxes)
                })

    return records


def process_workplace_safety_coco(cfg, out_img_dir, out_lbl_dir):
    source_cfg = cfg["sources"]["workplace_safety_coco"]
    zip_path = os.path.join(cfg["paths"]["raw_dir"], source_cfg["archive"])
    class_map = {int(k): int(v) for k, v in source_cfg["class_map"].items()}
    records = []

    print(f"--> Processing {source_cfg['archive']} (Workplace Safety COCO)...")
    with zipfile.ZipFile(zip_path, "r") as z:
        for split in ["train", "valid", "test"]:
            json_path = f"{split}/_annotations.coco.json"
            if json_path not in z.namelist():
                continue

            coco_data = json.loads(z.read(json_path).decode("utf-8"))
            images_info = {img["id"]: img for img in coco_data.get("images", [])}
            annotations = coco_data.get("annotations", [])

            boxes_by_image = {img_id: [] for img_id in images_info}
            for ann in annotations:
                cat_id = ann.get("category_id")
                img_id = ann.get("image_id")
                if cat_id in class_map and img_id in images_info:
                    tgt_cls = class_map[cat_id]
                    bbox = ann.get("bbox", [])
                    if len(bbox) == 4:
                        bx, by, bw, bh = bbox
                        iw = images_info[img_id]["width"]
                        ih = images_info[img_id]["height"]
                        xc = bx + bw / 2.0
                        yc = by + bh / 2.0
                        norm = normalize_box(xc, yc, bw, bh, iw, ih, is_pixel=True)
                        if norm:
                            boxes_by_image[img_id].append((tgt_cls, *norm))

            for img_id, img_info in images_info.items():
                fn = img_info["file_name"]
                arc_img_path = f"{split}/{fn}"
                if arc_img_path in z.namelist():
                    stem = os.path.splitext(fn)[0]
                    clean_id = f"workplacesafety_{split}_{stem}"
                    out_img_path = os.path.join(out_img_dir, f"{clean_id}.jpg")
                    with open(out_img_path, "wb") as f_out:
                        f_out.write(z.read(arc_img_path))

                    with Image.open(out_img_path) as im:
                        w, h = im.size

                    target_boxes = boxes_by_image.get(img_id, [])
                    out_lbl_path = os.path.join(out_lbl_dir, f"{clean_id}.txt")
                    with open(out_lbl_path, "w", encoding="utf-8") as f_lbl:
                        for b in target_boxes:
                            f_lbl.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

                    records.append({
                        "image_id": clean_id,
                        "source_dataset": "workplace_safety_coco",
                        "original_split": split,
                        "original_path": arc_img_path,
                        "image_path": out_img_path,
                        "label_path": out_lbl_path,
                        "width": w,
                        "height": h,
                        "num_target_boxes": len(target_boxes),
                        "box_classes": ",".join(str(b[0]) for b in target_boxes)
                    })

    return records


def process_ppe_6cls(cfg, out_img_dir, out_lbl_dir):
    source_cfg = cfg["sources"]["ppe_6cls"]
    zip_path = os.path.join(cfg["paths"]["raw_dir"], source_cfg["archive"])
    class_map = {int(k): int(v) for k, v in source_cfg["class_map"].items()}
    records = []

    print(f"--> Processing {source_cfg['archive']} (PPE 6-Class YOLO)...")
    with zipfile.ZipFile(zip_path, "r") as z:
        label_files = {n: n for n in z.namelist() if "/labels/" in n and n.endswith(".txt")}
        for n in z.namelist():
            if "/images/" in n and n.lower().endswith((".jpg", ".png", ".jpeg")):
                parts = n.split("/")
                orig_split = parts[0]  # train, valid, test
                fname = parts[-1]
                stem = os.path.splitext(fname)[0]
                img_id = f"ppe6cls_{orig_split}_{stem}"

                img_data = z.read(n)
                out_img_path = os.path.join(out_img_dir, f"{img_id}.jpg")
                with open(out_img_path, "wb") as f_out:
                    f_out.write(img_data)

                with Image.open(out_img_path) as im:
                    w, h = im.size

                lbl_archive_name = n.replace("/images/", "/labels/").replace(os.path.splitext(n)[1], ".txt")
                target_boxes = []
                if lbl_archive_name in label_files:
                    lines = z.read(lbl_archive_name).decode("utf-8", errors="ignore").splitlines()
                    for line in lines:
                        parts_l = line.strip().split()
                        if len(parts_l) >= 5:
                            src_cls = int(parts_l[0])
                            if src_cls in class_map:
                                tgt_cls = class_map[src_cls]
                                xc, yc, bw, bh = map(float, parts_l[1:5])
                                norm = normalize_box(xc, yc, bw, bh, w, h, is_pixel=False)
                                if norm:
                                    target_boxes.append((tgt_cls, *norm))

                out_lbl_path = os.path.join(out_lbl_dir, f"{img_id}.txt")
                with open(out_lbl_path, "w", encoding="utf-8") as f_lbl:
                    for b in target_boxes:
                        f_lbl.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

                records.append({
                    "image_id": img_id,
                    "source_dataset": "ppe_6cls",
                    "original_split": orig_split,
                    "original_path": n,
                    "image_path": out_img_path,
                    "label_path": out_lbl_path,
                    "width": w,
                    "height": h,
                    "num_target_boxes": len(target_boxes),
                    "box_classes": ",".join(str(b[0]) for b in target_boxes)
                })

    return records


def process_jhboyo_test(cfg, out_img_dir, out_lbl_dir):
    source_cfg = cfg["sources"]["jhboyo_ppe"]
    folder = source_cfg["folder"]
    img_dir = os.path.join(folder, "images")
    lbl_dir = os.path.join(folder, "labels")
    class_map = {int(k): int(v) for k, v in source_cfg["class_map"].items()}
    records = []

    print("--> Processing jhboyo test split (YOLO)...")
    for fname in os.listdir(img_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            stem = os.path.splitext(fname)[0]
            img_id = f"jhboyo_test_{stem}"
            src_img_path = os.path.join(img_dir, fname)
            out_img_path = os.path.join(out_img_dir, f"{img_id}.png")

            with Image.open(src_img_path) as im:
                w, h = im.size
                im.save(out_img_path)

            src_lbl_path = os.path.join(lbl_dir, f"{stem}.txt")
            target_boxes = []
            if os.path.exists(src_lbl_path):
                with open(src_lbl_path, "r", encoding="utf-8") as fp:
                    for line in fp:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            src_cls = int(parts[0])
                            if src_cls in class_map:
                                tgt_cls = class_map[src_cls]
                                xc, yc, bw, bh = map(float, parts[1:5])
                                norm = normalize_box(xc, yc, bw, bh, w, h, is_pixel=False)
                                if norm:
                                    target_boxes.append((tgt_cls, *norm))

            out_lbl_path = os.path.join(out_lbl_dir, f"{img_id}.txt")
            with open(out_lbl_path, "w", encoding="utf-8") as f_lbl:
                for b in target_boxes:
                    f_lbl.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

            records.append({
                "image_id": img_id,
                "source_dataset": "jhboyo_ppe",
                "original_split": "test",
                "original_path": src_img_path,
                "image_path": out_img_path,
                "label_path": out_lbl_path,
                "width": w,
                "height": h,
                "num_target_boxes": len(target_boxes),
                "box_classes": ",".join(str(b[0]) for b in target_boxes)
            })

    return records


def process_cvat_helmet(cfg, out_img_dir, out_lbl_dir):
    source_cfg = cfg["sources"]["cvat_helmet"]
    zip_path = os.path.join(cfg["paths"]["raw_dir"], source_cfg["archive"])
    class_map = source_cfg["class_map"]
    records = []

    print(f"--> Processing {source_cfg['archive']} (CVAT XML)...")
    with zipfile.ZipFile(zip_path, "r") as z:
        if "annotations.xml" in z.namelist():
            xml_data = z.read("annotations.xml").decode("utf-8")
            root = ET.fromstring(xml_data)
            for img_node in root.findall(".//image"):
                img_name = img_node.attrib.get("name")
                if not img_name.startswith("img/"):
                    continue
                w = float(img_node.attrib.get("width"))
                h = float(img_node.attrib.get("height"))
                stem = os.path.splitext(os.path.basename(img_name))[0]
                img_id = f"cvat_helmet_{stem}"

                target_boxes = []
                for box_node in img_node.findall("box"):
                    label = box_node.attrib.get("label")
                    if label in class_map:
                        tgt_cls = class_map[label]
                        xtl = float(box_node.attrib.get("xtl"))
                        ytl = float(box_node.attrib.get("ytl"))
                        xbr = float(box_node.attrib.get("xbr"))
                        ybr = float(box_node.attrib.get("ybr"))
                        xc = (xtl + xbr) / 2.0
                        yc = (ytl + ybr) / 2.0
                        bw = xbr - xtl
                        bh = ybr - ytl
                        norm = normalize_box(xc, yc, bw, bh, w, h, is_pixel=True)
                        if norm:
                            target_boxes.append((tgt_cls, *norm))

                out_img_path = os.path.join(out_img_dir, f"{img_id}.png")
                with open(out_img_path, "wb") as f_out:
                    f_out.write(z.read(img_name))

                out_lbl_path = os.path.join(out_lbl_dir, f"{img_id}.txt")
                with open(out_lbl_path, "w", encoding="utf-8") as f_lbl:
                    for b in target_boxes:
                        f_lbl.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

                records.append({
                    "image_id": img_id,
                    "source_dataset": "cvat_helmet",
                    "original_split": "none",
                    "original_path": img_name,
                    "image_path": out_img_path,
                    "label_path": out_lbl_path,
                    "width": int(w),
                    "height": int(h),
                    "num_target_boxes": len(target_boxes),
                    "box_classes": ",".join(str(b[0]) for b in target_boxes)
                })

    return records


def main():
    cfg = load_config()
    staging_img_dir = os.path.join(cfg["paths"]["processed_dir"], "staging", "images")
    staging_lbl_dir = os.path.join(cfg["paths"]["processed_dir"], "staging", "labels")
    os.makedirs(staging_img_dir, exist_ok=True)
    os.makedirs(staging_lbl_dir, exist_ok=True)

    all_records = []
    all_records.extend(process_ppe_2286y(cfg, staging_img_dir, staging_lbl_dir))
    all_records.extend(process_safety_vests_csv(cfg, staging_img_dir, staging_lbl_dir))
    all_records.extend(process_css_data(cfg, staging_img_dir, staging_lbl_dir))
    all_records.extend(process_workplace_safety_coco(cfg, staging_img_dir, staging_lbl_dir))
    all_records.extend(process_ppe_6cls(cfg, staging_img_dir, staging_lbl_dir))
    all_records.extend(process_jhboyo_test(cfg, staging_img_dir, staging_lbl_dir))
    all_records.extend(process_cvat_helmet(cfg, staging_img_dir, staging_lbl_dir))

    df = pd.DataFrame(all_records)
    staging_manifest = os.path.join(cfg["paths"]["processed_dir"], "staging_manifest.csv")
    df.to_csv(staging_manifest, index=False)
    print(f"\n[DONE] Unified conversion complete: {len(df)} images staged.")
    print(f"Staging manifest saved to: {staging_manifest}")


if __name__ == "__main__":
    main()

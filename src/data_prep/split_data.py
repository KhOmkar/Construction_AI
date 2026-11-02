"""
Perform stratified 80/10/10 split by image (fixed seed = 42) and isolate the real-only final_test benchmark.
Organizes processed/scenes/ and processed/final_test/.
"""

import os
import shutil
import yaml
import pandas as pd
import numpy as np


def load_config(config_path="configs/dataset.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_splits(manifest_df, seed=42, ratios=(0.80, 0.10, 0.10)):
    np.random.seed(seed)

    # Separate duplicates first
    is_dup = manifest_df["is_duplicate"].fillna(False)

    # Keep jhboyo test as the pristine, reserved real-only final_test set
    is_final_test = (manifest_df["source_dataset"] == "jhboyo_ppe") & (~is_dup)

    # All other non-duplicate images form the pool for 80/10/10 split
    pool_indices = manifest_df[~is_dup & ~is_final_test].index.to_numpy().copy()
    np.random.shuffle(pool_indices)

    n_total = len(pool_indices)
    n_train = int(n_total * ratios[0])
    n_val = int(n_total * ratios[1])

    train_indices = pool_indices[:n_train]
    val_indices = pool_indices[n_train : n_train + n_val]
    test_indices = pool_indices[n_train + n_val :]

    split_col = ["duplicate"] * len(manifest_df)
    for idx in train_indices:
        split_col[idx] = "train"
    for idx in val_indices:
        split_col[idx] = "val"
    for idx in test_indices:
        split_col[idx] = "test"
    for idx in manifest_df[is_final_test].index:
        split_col[idx] = "final_test"

    manifest_df["split"] = split_col
    return manifest_df


def organize_filesystem(cfg, manifest_df):
    scenes_dir = cfg["paths"]["scenes_dir"]
    final_test_dir = cfg["paths"]["final_test_dir"]

    # Target directories
    dirs = {
        "train_img": os.path.join(scenes_dir, "images", "train"),
        "train_lbl": os.path.join(scenes_dir, "labels", "train"),
        "val_img": os.path.join(scenes_dir, "images", "val"),
        "val_lbl": os.path.join(scenes_dir, "labels", "val"),
        "test_img": os.path.join(scenes_dir, "images", "test"),
        "test_lbl": os.path.join(scenes_dir, "labels", "test"),
        "final_test_img": os.path.join(final_test_dir, "images"),
        "final_test_lbl": os.path.join(final_test_dir, "labels"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    final_img_paths = []
    final_lbl_paths = []

    print("--> Organizing split files into final directory layout...")
    for idx, row in manifest_df.iterrows():
        split = row["split"]
        src_img = row["image_path"]
        src_lbl = row["label_path"]
        fname = os.path.basename(src_img)
        lbl_fname = os.path.basename(src_lbl)

        if split in ["train", "val", "test"]:
            dst_img = os.path.join(scenes_dir, "images", split, fname)
            dst_lbl = os.path.join(scenes_dir, "labels", split, lbl_fname)
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)
            final_img_paths.append(dst_img)
            final_lbl_paths.append(dst_lbl)
        elif split == "final_test":
            dst_img = os.path.join(final_test_dir, "images", fname)
            dst_lbl = os.path.join(final_test_dir, "labels", lbl_fname)
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)
            final_img_paths.append(dst_img)
            final_lbl_paths.append(dst_lbl)
        else:
            final_img_paths.append(src_img)
            final_lbl_paths.append(src_lbl)

    manifest_df["final_image_path"] = final_img_paths
    manifest_df["final_label_path"] = final_lbl_paths

    # Generate YOLO data.yaml
    yolo_yaml_path = os.path.join(scenes_dir, "dataset.yaml")
    yolo_cfg = {
        "path": os.path.abspath(scenes_dir).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 5,
        "names": {
            0: "helmet",
            1: "no_helmet",
            2: "vest",
            3: "no_vest",
            4: "person",
        },
    }
    with open(yolo_yaml_path, "w", encoding="utf-8") as yf:
        yaml.dump(yolo_cfg, yf, default_flow_style=False)

    print(f"[SPLIT RESULTS] Dataset split counts:\n{manifest_df['split'].value_counts()}")
    return manifest_df


def main():
    cfg = load_config()
    dedup_csv = os.path.join(cfg["paths"]["processed_dir"], "deduped_manifest.csv")
    if not os.path.exists(dedup_csv):
        print(f"Error: {dedup_csv} not found. Run deduplicate first.")
        return

    df = pd.read_csv(dedup_csv)
    df = create_splits(df, seed=cfg.get("seed", 42), ratios=cfg["processing"]["split_ratios"])
    df = organize_filesystem(cfg, df)

    split_manifest = os.path.join(cfg["paths"]["processed_dir"], "split_manifest.csv")
    df.to_csv(split_manifest, index=False)
    print(f"Saved split manifest to: {split_manifest}")


if __name__ == "__main__":
    main()

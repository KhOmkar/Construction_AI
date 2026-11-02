"""
Perceptual deduplication using dhash to identify and prune exact and near-duplicate images across datasets.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm


def load_config(config_path="configs/dataset.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_hashes(manifest_df):
    hashes = []
    print("--> Computing perceptual hashes (dhash 64-bit)...")
    for idx, row in tqdm(manifest_df.iterrows(), total=len(manifest_df)):
        img_path = row["image_path"]
        try:
            with Image.open(img_path) as im:
                h = str(imagehash.dhash(im))
                hashes.append(h)
        except Exception as e:
            hashes.append("error")
    manifest_df["dhash"] = hashes
    return manifest_df


def deduplicate_records(manifest_df, threshold=2):
    print(f"--> Deduplicating images (Hamming distance threshold <= {threshold})...")
    # First sort by num_target_boxes desc, then resolution desc so we keep the richer annotations
    manifest_df["area"] = manifest_df["width"] * manifest_df["height"]
    manifest_df = manifest_df.sort_values(
        by=["num_target_boxes", "area"], ascending=[False, False]
    ).reset_index(drop=True)

    keep_mask = np.ones(len(manifest_df), dtype=bool)
    duplicate_of = [None] * len(manifest_df)

    # Convert hex hashes to integer binary representation for ultra-fast hamming distance
    valid_mask = manifest_df["dhash"] != "error"
    hash_ints = np.array([int(h, 16) if h != "error" else 0 for h in manifest_df["dhash"]], dtype=np.uint64)

    # Fast clustering
    for i in tqdm(range(len(manifest_df)), desc="Clustering hashes"):
        if not keep_mask[i] or not valid_mask.iloc[i]:
            continue
        h_i = hash_ints[i]
        # Compare with remaining candidate images
        candidates = np.where(keep_mask[i + 1 :])[0] + (i + 1)
        if len(candidates) == 0:
            break
        cand_hashes = hash_ints[candidates]

        # Hamming distance using XOR bit count
        xor_result = np.bitwise_xor(h_i, cand_hashes)
        # Vectorized popcount for uint64
        # In python/numpy: convert to int and count bits or use lookup
        dists = [bin(int(x)).count("1") for x in xor_result]
        matches = np.array(dists) <= threshold

        matched_indices = candidates[matches]
        for m_idx in matched_indices:
            keep_mask[m_idx] = False
            duplicate_of[m_idx] = manifest_df.iloc[i]["image_id"]

    manifest_df["is_duplicate"] = ~keep_mask
    manifest_df["duplicate_of"] = duplicate_of

    pruned_count = (~keep_mask).sum()
    kept_count = keep_mask.sum()
    print(f"[DEDUP RESULTS] Total images: {len(manifest_df)} | Kept: {kept_count} | Removed duplicates: {pruned_count}")

    return manifest_df


def main():
    cfg = load_config()
    staging_manifest = os.path.join(cfg["paths"]["processed_dir"], "staging_manifest.csv")
    if not os.path.exists(staging_manifest):
        print(f"Error: {staging_manifest} does not exist. Run convert_all first.")
        return

    df = pd.read_csv(staging_manifest)
    df = compute_hashes(df)
    threshold = cfg["processing"].get("hash_distance_threshold", 2)
    df = deduplicate_records(df, threshold=threshold)

    out_csv = os.path.join(cfg["paths"]["processed_dir"], "deduped_manifest.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved deduplication manifest to: {out_csv}")


if __name__ == "__main__":
    main()

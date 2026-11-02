"""
Master pipeline runner for Phase 1 Data Preparation.
Executes split_data, extract_crops, and generate_stats_and_grids sequentially.
"""

import os
import sys
import time
from src.data_prep.split_data import main as run_split
from src.data_prep.extract_crops import main as run_extract_crops
from src.data_prep.generate_stats_and_grids import main as run_stats_and_grids


def main():
    t0 = time.time()
    print("==================================================")
    print("PHASE 1: STEP 3 - DATA SPLITTING & LAYOUT SETUP")
    print("==================================================")
    run_split()

    print("\n==================================================")
    print("PHASE 1: STEP 4 - WORKER CROP EXTRACTION (64x64)")
    print("==================================================")
    run_extract_crops()

    print("\n==================================================")
    print("PHASE 1: STEP 5 & 6 - STATISTICS & SAMPLE GRIDS")
    print("==================================================")
    run_stats_and_grids()

    print(f"\n[PIPELINE COMPLETE] Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

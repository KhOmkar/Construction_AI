"""
PyTorch Dataset classes for loading worker crops from compressed .npz files.
Supports RAM caching and memory mapping for ultra-fast local Tier B training.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class WorkerCropDataset(Dataset):
    """
    Dataset for 64x64 (or 128x128) worker crops stored in .npz archives.
    
    Classes:
      0: compliant (wearing helmet and vest)
      1: no_helmet violation
      2: no_vest violation
      3: both_missing violation
    """

    def __init__(
        self,
        npz_path,
        transform=None,
        filter_class=None,
        normalize_mode="tanh",  # 'tanh' -> [-1, 1] for GAN/Diffusion; 'sigmoid' -> [0, 1] for AE/VAE
        mmap=False
    ):
        super().__init__()
        self.npz_path = npz_path
        self.normalize_mode = normalize_mode
        self.filter_class = filter_class

        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Crop archive not found: {npz_path}")

        mmap_mode = "r" if mmap else None
        data = np.load(npz_path, mmap_mode=mmap_mode, allow_pickle=True)

        self.images = data["images"]  # shape (N, H, W, 3) uint8
        self.labels = data["labels"]  # shape (N,) int64
        self.captions = data["captions"] if "captions" in data else None
        self.image_ids = data["image_ids"] if "image_ids" in data else None

        if filter_class is not None:
            # E.g. for Autoencoder anomaly training: only compliant (class 0)
            if isinstance(filter_class, int):
                filter_class = [filter_class]
            mask = np.isin(self.labels, filter_class)
            self.indices = np.where(mask)[0]
        else:
            self.indices = np.arange(len(self.labels))

        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_arr = self.images[real_idx]  # H, W, 3 uint8
        label = int(self.labels[real_idx])

        # Convert to Tensor (3, H, W) in float32 [0.0, 1.0]
        tensor = torch.from_numpy(img_arr).permute(2, 0, 1).float() / 255.0

        if self.transform is not None:
            tensor = self.transform(tensor)
        else:
            if self.normalize_mode == "tanh":
                # Normalize [0, 1] -> [-1, 1]
                tensor = (tensor - 0.5) / 0.5
            # if 'sigmoid', stays [0, 1]

        caption = str(self.captions[real_idx]) if self.captions is not None else ""
        img_id = str(self.image_ids[real_idx]) if self.image_ids is not None else ""

        return {
            "image": tensor,
            "label": label,
            "caption": caption,
            "image_id": img_id,
            "index": real_idx
        }


def get_crop_dataloader(
    npz_path,
    batch_size=128,
    shuffle=True,
    num_workers=0,
    filter_class=None,
    normalize_mode="tanh",
    pin_memory=True
):
    dataset = WorkerCropDataset(
        npz_path=npz_path,
        filter_class=filter_class,
        normalize_mode=normalize_mode
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True if shuffle else False
    )
    return loader

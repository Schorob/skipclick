import os
import random
import pickle as pkl
from pathlib import Path
from PIL import Image

import cv2
import numpy as np

from isegm.data.base import ISDataset
from isegm.data.sample import DSample
from isegm.utils.misc import get_labels_with_sizes


"""
@misc{schoen2024wseseg,
      title={WSESeg: Introducing a Dataset for the Segmentation of Winter Sports Equipment with a Baseline for Interactive Segmentation},
      author={Robin Schön and Daniel Kienzle and Rainer Lienhart},
      year={2024},
      eprint={2407.09288},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.09288},
}
"""

CLASS_NAMES = [
    "flickr_bobsleighs", "flickr_curling_stone", "flickr_ski_helmets", "flickr_snow_kites",  "yt_sports",
    "flickr_curling_brooms", "flickr_ski_goggles", "flickr_ski_misc", "flickr_slalom_obstacles", "flickr_snowboards"
]

class WSESeg(ISDataset):
    def __init__(self, dataset_path, class_name, **kwargs):
        super(WSESeg, self).__init__()
        assert class_name in CLASS_NAMES
        self.class_name = class_name
        if class_name == "yt_sports":
            self.dataset_path = Path(dataset_path) / "yt_sports/skijump/annotated_frames/0"
        else:
            self.dataset_path = Path(dataset_path) / class_name

        self.mask_files = list(sorted(self.dataset_path.glob("mask_*.png")))
        self.dataset_samples = []

        for midx, mfile in enumerate(self.mask_files):
            mask = np.array(Image.open(mfile))
            item_idxs = np.unique(mask)[1:]
            for iidx in item_idxs:
                self.dataset_samples.append((midx, iidx))

    def get_sample(self, index) -> DSample:
        # 1. Obtain the mask and create an int32-0/1-mask
        midx, iidx = self.dataset_samples[index]
        mask_path = self.mask_files[midx]
        mask = np.array(Image.open(str(mask_path)))
        mask = np.where(mask == iidx, 1, 0)
        mask = mask.astype(np.int32)

        # 2. Load the image
        image_path = str(mask_path)[:-4] + ".jpg"  # Change the extension
        image_path = image_path.split("/")
        image_path[-1] = image_path[-1][5:]
        image_path = "/".join(image_path)
        image = np.array(Image.open(str(image_path))) / 255.0

        if len(image.shape) == 2:
            image = np.stack([
                image, image, image
            ], axis=2)
        image = image.astype(np.float32)

        return DSample(image, mask, objects_ids=[1], sample_id=index)


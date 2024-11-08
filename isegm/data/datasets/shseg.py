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
This dataset has been randomly extracted amongst images with height and width 
> 150 px in the SkiTB dataset. 
The original images have been published alongside the following paper 
@inproceedings{dunnhofer2024tracking,
  title={Tracking Skiers from the Top to the Bottom},
  author={Dunnhofer, Matteo and Sordi, Luca and Martinel, Niki and Micheloni, Christian},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={8511--8521},
  year={2024}
}
"""

class SHSeg(ISDataset):
    def __init__(self, dataset_path, **kwargs):
        super(SHSeg, self).__init__()
        self.dataset_path = Path(dataset_path)

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


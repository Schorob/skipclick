import cv2
import json
import random
import numpy as np
from pathlib import Path
from isegm.data.base import ISDataset
from isegm.data.sample import DSample

class CocoStuffSingle(ISDataset):
    def __init__(self, dataset_path, split='train',  **kwargs):
        super(CocoStuffSingle, self).__init__(**kwargs)

        assert split in ["train", "val"]
        self.split = split
        self.dataset_path = Path(dataset_path)

        if split == "train":
            image_folder = self.dataset_path / "train2017"
            stuff_mask_folder = self.dataset_path / "cocostuff" / "train2017"
        elif split == "val":
            image_folder = self.dataset_path / "val2017"
            stuff_mask_folder = self.dataset_path / "cocostuff" / "val2017"
        else:
            raise NotImplementedError

        image_path_list = list(sorted(image_folder.glob("*.jpg")))
        stuff_path_list = list(sorted(stuff_mask_folder.glob("*.png")))

        self.dataset_samples = list(zip(image_path_list, stuff_path_list))
        # TODO: Shortened dataset
        self.dataset_samples = self.dataset_samples[:200]

    def get_sample(self, index) -> DSample:
        img_path, mask_path = self.dataset_samples[index]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED).astype(np.int32)

        fg_classes = np.unique(mask)
        if 255 in fg_classes:
            fg_classes = fg_classes[:-1]
        chosen_class = np.random.choice(fg_classes, size=1)
        mask = np.where(mask == chosen_class, 1, 0)

        return DSample(image, mask, objects_ids=[1], sample_id=index)
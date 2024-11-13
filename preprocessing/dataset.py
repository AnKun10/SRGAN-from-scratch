import os
import numpy as np
import utils.config as config
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super(MyImageFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.class_names = os.listdir(root_dir)

        for idx, name in enumerate(self.class_names):
            files = os.listdir(os.path.join(root_dir, name))
            self.data += list(zip(files, [idx] * len(files)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_file, label = self.data[idx]
        root_and_dir = os.path.join(self.root_dir, self.class_names[label])

        image = np.array(Image.open(os.path.join(root_and_dir, img_file)))
        image = config.both_transform(image=image)["image"]
        high_res = config.highres_transform(image=image)["image"]
        low_res = config.lowres_transform(image=image)["image"]
        return low_res, high_res

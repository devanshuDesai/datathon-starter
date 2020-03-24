from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as tf
from glob import glob
import re
import pandas as pd
import os

class KaggleDataset(Dataset):
    def __init__(self, path, transforms=None):
        self.path = path
        self.transforms = transforms
        self.default_transforms = tf.Compose([tf.Resize(224),
                                  tf.RandomCrop(224),
                                  tf.ToTensor(),
                                  tf.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])])
        self.img_dir = f'./image/{path}/'
        try:
            cat_data = pd.read_csv(f'{path}.csv')
            self.fnames = cat_data.fname
            self.targets = cat_data.breedID
        except FileNotFoundError:
            self.fnames = glob(f'{self.img_dir}*')
            self.fnames = list(map(lambda x: re.findall('\d{4}', x)[0], self.fnames))

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f'{self.fnames[idx]}.jpg')
        img = Image.open(img_path)
        if self.transforms:
            img = self.transforms(img)
        else:
            img = self.default_transforms(img)
        if (self.path=='test'):
            fname = self.fnames[idx]
            return fname, img
        else:
            target = self.targets[idx]-1
            return img, target

    def __len__(self):
        return len(self.fnames)


import pandas as pd
from torch.utils import data
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        path = self.root + self.img_name[index] + '.jpeg'
        img = Image.open(path)
        label = self.label[index]

        transform1 = transforms.Compose([
            transforms.ToTensor(),      # (0, 255) -> (0.0, 1.0) & [C, H, W]
            ]
        )
        img = transform1(img)
        
        return img, label

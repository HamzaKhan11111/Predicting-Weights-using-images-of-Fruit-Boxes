from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import torch
import os

class MyDataset(Dataset):
    def __init__(self,csv_path,transform=None):
        super().__init__
        csv=pd.read_csv(csv_path)
        self.img_path=csv.iloc[:,0].tolist()
        self.target=csv.iloc[:,1].tolist()
        self.total_images=len(csv)
        self.transform=transform

    def __len__(self):
        return self.total_images
    
    def __getitem__(self, index):
      if self.transform:
        img = self.transform(Image.open(os.path.join("Data", self.img_path[index])).convert("RGB"))
        target = torch.tensor(self.target[index])
        return img, target
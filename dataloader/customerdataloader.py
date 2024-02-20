import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from scipy.io import loadmat

import glob

class CustomMMFIDataset(Dataset):
    def __init__(self, cfg, transform=None):
        # 使用glob模式匹配来找到所有图片文件
        self.image_paths = glob.glob(os.path.join(cfg.images_dir.Name, '**/rgb/*.png'), recursive=True)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 获取当前图片的路径
        img_path = self.image_paths[idx]
        # 读取图片
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # 根据图片路径构建CSI文件的路径
        csi_path = img_path.replace('/rgb/', '/wifi-csi/').replace('.png', '.mat')
        
        # 读取CSI文件
        csi = loadmat(csi_path)
        CSI_phase = torch.from_numpy(csi['CSIphase']).float()
        CSI_amp = torch.from_numpy(csi['CSIamp']).float()
        CSI = {'phase': CSI_phase, 'amp': CSI_amp}
        
        return {'image': image, 'csi': CSI}
    
def custom_collate_fn(batch):

    dict_list = []

    for data in batch:
        dict_list.append(data)

    return dict_list



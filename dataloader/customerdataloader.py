import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from scipy.io import loadmat
from detectron2.data.detection_utils import read_image
import glob
import detectron2.data.transforms as T
from torchvision.transforms import functional as F
from detectron2.data.transforms import ResizeShortestEdge, PadTransform
import numpy as np
from scipy.signal import medfilt
class CustomMMFIDataset(Dataset):
    def __init__(self, cfg, transform=None):
        self.include_dirs = ['E01', 'E02', 'E03']
        all_image_paths = glob.glob(os.path.join(cfg.images_dir.Name, '**/rgb/*.png'), recursive=True)
        self.image_paths = [path for path in all_image_paths if any(dir_name in path for dir_name in self.include_dirs)]
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST,cfg.INPUT.MAX_SIZE_TEST],cfg.INPUT.MAX_SIZE_TEST
        )
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        
        img_path = self.image_paths[idx]
        origin_image = read_image(img_path, format="BGR")
        image = self.aug.get_transform(origin_image).apply_image(origin_image)

        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        csi_path = img_path.replace('/rgb/', '/wifi-csi/').replace('.png', '.mat')
        csi = loadmat(csi_path)
        csi_phase_np = np.array(csi['CSIphase'])
        csi_phase_unwrapped_np = np.unwrap(csi_phase_np, axis=-1)
        csi_phase_filtered_np = medfilt(csi_phase_unwrapped_np, kernel_size=(1, 1, 3))
        csi_phase_adjusted_np = linear_fit_phase(csi_phase_filtered_np)
        csi_phase_adjusted_tensor = torch.from_numpy(csi_phase_adjusted_np).float()

        csi_amp_np = np.array(csi['CSIamp'])
        csi_amp_filtered_np = medfilt(csi_amp_np, kernel_size=(1, 1, 3))
        csi_amp_adjusted_tensor = torch.from_numpy(csi_amp_filtered_np).float()
        CSI = {'phase': csi_phase_adjusted_tensor, 'amp': csi_amp_adjusted_tensor}
        
        return {'image': image, 'csi': CSI}
    
def custom_collate_fn(batch):

    dict_list = []

    for data in batch:
        dict_list.append(data)

    return dict_list

def linear_fit_phase(phase_data, F=114):
    phase_adjusted = np.zeros_like(phase_data)
    Phi_F = phase_data[:, -1, :] 
    Phi_1 = phase_data[:, 0, :]   
    alpha_1 = (Phi_F - Phi_1) / (2 * np.pi * F)
    alpha_0 = np.mean(phase_data, axis=1) / F
    for f in range(1, F + 1):
        phase_adjusted[:, f - 1, :] = phase_data[:, f - 1, :] - (alpha_1 * f + alpha_0)
    return phase_adjusted


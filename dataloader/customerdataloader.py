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
        CSI_phase = torch.from_numpy(csi['CSIphase']).float()
        CSI_amp = torch.from_numpy(csi['CSIamp']).float()
        CSI = {'phase': CSI_phase, 'amp': CSI_amp}
        
        return {'image': image, 'csi': CSI}
    
def custom_collate_fn(batch):

    dict_list = []

    for data in batch:
        dict_list.append(data)

    return dict_list



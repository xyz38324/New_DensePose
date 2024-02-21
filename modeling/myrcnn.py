import torch
from detectron2.config import configurable
from typing import Dict, List
from modeling.build_model import Teacher_Model_REGISTRY
from detectron2.modeling.meta_arch import GeneralizedRCNN
from densepose.modeling.roi_heads import DensePoseROIHeads
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.structures import ImageList, Instances
from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.structures import Boxes
from copy import deepcopy
@Teacher_Model_REGISTRY.register()
class MyGeneralizedRCNN(GeneralizedRCNN):
    @configurable
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, cfg):
        
        parent_params = super().from_config(cfg)
        return parent_params
    
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)      
        proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(images, features, proposals, None)
        
        return results,features

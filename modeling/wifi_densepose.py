import torch.nn as nn
import torch
from typing import Dict, List, Optional, Tuple
from .build_model import Model_REGISTRY
from detectron2.config import configurable
from detectron2.modeling import  build_proposal_generator,Backbone
from detectron2.modeling.backbone import Backbone,build_backbone
from modeling.build_model import build_teacher_model
from modeling.build_model import build_mtn
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.structures import ImageList
from detectron2.checkpoint import DetectionCheckpointer

__all__ = ["WiFi_DensePose"]

@Model_REGISTRY.register()
class WiFi_DensePose(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        mtn:nn.Module,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        input_format: Optional[str] = None,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        model_weights

    ):  
        super().__init__()
      
        self.mtn =  mtn      
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
       

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
       
      
        
       
    @classmethod
    def from_config(cls,cfg):
        backbone = build_backbone(cfg)
        return {
            "mtn":build_mtn(cfg),
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),            
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "model_weights":cfg.MODEL.WEIGHTS
            
        }
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        csi_phase = [x["csi"]['phase'].to(self.device) for x in batched_inputs]
        csi_phase_tensor = torch.stack(csi_phase)
        csi_amp = [x["csi"]['amp'].to(self.device) for x in batched_inputs]
        csi_amp_tensor = torch.stack(csi_amp)
        mtn_output = self.mtn(csi_amp_tensor,csi_phase_tensor)
        output_list=[]
        for i in range(mtn_output.shape[0]):
            current_tensor = mtn_output[i]
            output_list.append(current_tensor.squeeze(0))

        mtn_image = self.preprocess_mtn(output_list)

        features = self.backbone(mtn_image.tensor)
        proposals, proposal_losses = self.proposal_generator(mtn_image, features, proposals_teacher)
        _, detector_losses = self.roi_heads(mtn_image, features, proposals, proposals_teacher)
         
        proposal_AB = ""
        detector_AB = ""


        losses = {}
        losses.update(proposal_AB)
        losses.update(detector_AB)
        return losses




    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_mtn(self, mtn_output: List[Dict[str, torch.Tensor]]):

        images = ImageList.from_tensors(
            mtn_output,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images
    
    def _calculate_transfer_learning_loss(self,teacher_features, student_features):
        loss = 0.0
        for key in ['p5','p4','p3', 'p2']:
            teacher_feature = teacher_features[key]
            
            student_feature = student_features[key]
            teacher_feature_downsampled = F.interpolate(teacher_feature, size=student_feature.shape[-2:], mode='nearest')
            if torch.isnan(teacher_feature_downsampled).any() or torch.isnan(student_feature).any():
                print("exist Nan")
            loss += F.mse_loss(student_feature, teacher_feature_downsampled)

        return loss

        



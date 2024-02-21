import torch.nn as nn
import torch
from typing import Dict, List, Optional, Tuple
from .build_model import Model_REGISTRY
from detectron2.config import configurable
from detectron2.modeling import  build_proposal_generator,Backbone
from detectron2.modeling.backbone import Backbone,build_backbone
from detectron2.structures import ImageList, Instances
from modeling.build_model import build_mtn
from .build_model import build_student_roihead
from detectron2.structures import Boxes
import torch.nn.functional as F

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
            "roi_heads": build_student_roihead(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "model_weights":cfg.MODEL.WEIGHTS
            
        }
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]],instances,teacher_features):
        new_instances_list=[]
        for instance in instances:
            if hasattr(instance, 'scores') and len(instance.scores) > 0:
                max_score_idx = instance.scores.argmax()
                new_instances = Instances(instance.image_size)
                new_instances.gt_boxes = Boxes(instance.pred_boxes.tensor[max_score_idx].unsqueeze(0)) 
                new_instances.gt_classes = instance.pred_classes[max_score_idx].unsqueeze(0) 
                new_instances.u = instance.pred_densepose.u[max_score_idx].unsqueeze(0)
                new_instances.v = instance.pred_densepose.v[max_score_idx].unsqueeze(0)

                new_instances_list.append(new_instances)

        del instances


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
        proposals, proposal_losses = self.proposal_generator(mtn_image, features, new_instances_list)
        _, detector_losses = self.roi_heads(mtn_image, features, proposals, new_instances_list)
        
        predicted_dp_u = detector_losses['dp_u']
        loss_u = 0.0
        for i, new_instance in enumerate(new_instances_list):
            
            true_u = new_instance.u  
            loss_u += F.mse_loss(predicted_dp_u[i], true_u.squeeze(0))
        loss_u /= len(new_instances_list)

        predicted_dp_v = detector_losses['dp_v']
        loss_v = 0.0
        for i, new_instance in enumerate(new_instances_list):
            
            true_v = new_instance.v 
            loss_v += F.mse_loss(predicted_dp_v[i], true_v.squeeze(0))
        loss_v /= len(new_instances_list)
        loss_densepose = loss_v+loss_u
       

        loss_transfer=0
        for key in teacher_features.keys():      
            teacher_output = teacher_features[key]
            student_output = features[key]
            layer_loss = F.mse_loss(student_output, teacher_output)
            loss_transfer += layer_loss

        losses = {'loss_densepose':loss_densepose*1000,
                  'loss_cls':detector_losses['loss_cls']*100+proposal_losses['loss_rpn_cls']+proposal_losses['loss_rpn_loc'],
                  'loss_box':detector_losses['loss_box_reg']*1000,
                  'loss_transfer':loss_transfer
                  }
      
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

        



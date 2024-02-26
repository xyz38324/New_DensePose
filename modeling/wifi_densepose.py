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
        transfer_only

    ):  
        super().__init__()
      
        self.mtn =  mtn      
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
       
        self.transfer_only = transfer_only
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
            "transfer_only":cfg.transfer.only
            
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

        # del instances


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
        mtn_image.tensor = self.normalize(mtn_image.tensor)
        features = self.backbone(mtn_image.tensor)
        scaled_features = {k: v / 1000 for k, v in features.items()}
        loss_transfer=self.calculate_loss(teacher_features,scaled_features)
        losses = {'loss_transfer':loss_transfer}

        if not self.transfer_only:
            proposals, proposal_losses = self.proposal_generator(mtn_image, scaled_features, new_instances_list)
            _, detector_losses = self.roi_heads(mtn_image, scaled_features, proposals, new_instances_list)
            predicted_dp_u = detector_losses['dp_u']
            loss_u = 0.0
            for i, new_instance in enumerate(new_instances_list):
                
                true_u = new_instance.u  
                loss_u += F.mse_loss(predicted_dp_u[i], true_u.squeeze(0))
            loss_u /= len(instances)
            predicted_dp_v = detector_losses['dp_v']
            loss_v = 0.0
            for i, new_instance in enumerate(new_instances_list):
                
                true_v = new_instance.v 
                loss_v += F.mse_loss(predicted_dp_v[i], true_v.squeeze(0))
            loss_v /= len(instances)
            loss_densepose = loss_v+loss_u
            losses.update({'loss_densepose':loss_densepose,
                  'loss_cls':detector_losses['loss_cls']+proposal_losses['loss_rpn_cls']+proposal_losses['loss_rpn_loc'],
                  'loss_box':detector_losses['loss_box_reg']})
       
        print(f"transfer:{loss_transfer}")
        # print(f"densepose:{loss_densepose}\n cls:{detector_losses['loss_cls']+proposal_losses['loss_rpn_cls']+proposal_losses['loss_rpn_loc']}\n box:{detector_losses['loss_box_reg']}\n ")

        return losses

    def calculate_loss(self, teacher_outputs, student_outputs):
        loss = 0
        keys_to_calculate = ['p2', 'p3', 'p4', 'p5']
        for key in keys_to_calculate:

            # if key in teacher_outputs and key in student_outputs:
            #     teacher_normalized = self.normalize(teacher_outputs[key])
            #     student_normalized = self.normalize(student_outputs[key])/1000
            #     current_loss = F.mse_loss(student_normalized, teacher_normalized)
            #     loss += current_loss
         
                
            current_loss = F.mse_loss(student_outputs[key], teacher_outputs[key])
            loss += current_loss
        return loss

    


    def normalize(self,tensor):
        min_val = tensor.view(tensor.size(0), tensor.size(1), -1).min(dim=2, keepdim=False)[0]
        max_val = tensor.view(tensor.size(0), tensor.size(1), -1).max(dim=2, keepdim=False)[0]

        # 调整min_val和max_val的形状以匹配原始tensor的批次和通道维度
        min_val = min_val.view(tensor.size(0), tensor.size(1), 1, 1)
        max_val = max_val.view(tensor.size(0), tensor.size(1), 1, 1)

        # 归一化到[0, 1]
        normalized_tensor = (tensor - min_val) / (max_val - min_val + 1e-5)
        
        return normalized_tensor


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
    
 
        



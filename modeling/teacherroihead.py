from .build_model import Teacher_ROIHEAD_REGISTRY
from densepose.modeling.roi_heads import DensePoseROIHeads
from typing import Dict, List, Optional
import torch
from detectron2.structures import ImageList, Instances
from detectron2.structures import Boxes

@Teacher_ROIHEAD_REGISTRY.register()
class Teacher_ROIHead(DensePoseROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
  

    # def _forward_densepose(self, features: Dict[str, torch.Tensor], instances: List[Instances]):

    #     features_list = [features[f] for f in self.in_features]
    #     pred_boxes = [x.pred_boxes for x in instances]

    #     # if self.use_decoder:
    #     #     features_list = [self.decoder(features_list)]

    #     features_dp = self.densepose_pooler(features_list, pred_boxes)
    #     if len(features_dp) > 0:
    #         densepose_head_outputs = self.densepose_head(features_dp)
    #         densepose_predictor_outputs = self.densepose_predictor(densepose_head_outputs)
    #         dp_u = densepose_predictor_outputs.u
    #         dp_v = densepose_predictor_outputs.v
            
    #     new_instances_list=[]
    #     for instance in instances:
    #         if hasattr(instance, 'scores') and len(instance.scores) > 0:
    #             max_score_idx = instance.scores.argmax()
    #             new_instances = Instances(instance.image_size)
    #             new_instances.gt_boxes = Boxes(instance.pred_boxes.tensor[max_score_idx].unsqueeze(0)) 
    #             new_instances.gt_classes = instance.pred_classes[max_score_idx].unsqueeze(0) 
    #             new_instances.u = dp_u[max_score_idx].unsqueeze(0)
    #             new_instances.v = dp_v[max_score_idx].unsqueeze(0)

    #             new_instances_list.append(new_instances)

    #     return new_instances_list

    
    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ):
    
        instances = super().forward_with_given_boxes(features, instances)
  
        return instances
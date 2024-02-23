from .build_model import Student_ROIHEAD_REGISTRY
from densepose.modeling.roi_heads import DensePoseROIHeads
from typing import Dict, List, Optional
import torch
from detectron2.structures import ImageList, Instances
from detectron2.modeling.roi_heads import select_foreground_proposals
@Student_ROIHEAD_REGISTRY.register()
class Student_ROIHead(DensePoseROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

    def _forward_densepose(self, features: Dict[str, torch.Tensor], instances: List[Instances]):

        features_list = [features[f] for f in self.in_features]
        
        proposals, _ = select_foreground_proposals(instances, self.num_classes)
        features_list, _ = self.densepose_data_filter(features_list, proposals)
        if len(proposals) > 0:
            proposal_boxes = [x.proposal_boxes for x in proposals]

            if self.use_decoder:
                features_list = [self.decoder(features_list)]

            features_dp = self.densepose_pooler(features_list, proposal_boxes)
            densepose_head_outputs = self.densepose_head(features_dp)
            densepose_predictor_outputs = self.densepose_predictor(densepose_head_outputs)

            densepose_loss_dict = self.densepose_losses(
                    proposals, densepose_predictor_outputs, embedder=self.embedder
                )


            loss={}
            loss['dp_u']=densepose_predictor_outputs.u
            loss['dp_v']=densepose_predictor_outputs.v
            return loss


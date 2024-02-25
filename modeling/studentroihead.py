from .build_model import Student_ROIHEAD_REGISTRY
from densepose.modeling.roi_heads import DensePoseROIHeads
from typing import Dict, List, Optional
import torch
from densepose.modeling.roi_heads import Decoder
from detectron2.structures import ImageList, Instances
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import select_foreground_proposals
from densepose.modeling.build import (
    build_densepose_data_filter,
    build_densepose_embedder,
    build_densepose_head,
    build_densepose_losses,
    build_densepose_predictor,
    
)


@Student_ROIHEAD_REGISTRY.register()
class Student_ROIHead(DensePoseROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)


    def _init_densepose_head(self, cfg, input_shape):
        # fmt: off
        self.densepose_on          = cfg.MODEL.DENSEPOSE_ON
        if not self.densepose_on:
            return
        self.densepose_data_filter = build_densepose_data_filter(cfg)
        dp_pooler_resolution       = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_RESOLUTION
        dp_pooler_sampling_ratio   = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_SAMPLING_RATIO
        dp_pooler_type             = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_TYPE
        self.use_decoder           = True
        # fmt: on
        if self.use_decoder:
            dp_pooler_scales = (1.0 / input_shape[self.in_features[0]].stride,)
        else:
            dp_pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        in_channels = [input_shape[f].channels for f in self.in_features][0]

        if self.use_decoder:
            self.decoder = Decoder(cfg, input_shape, self.in_features)

        self.densepose_pooler = ROIPooler(
            output_size=dp_pooler_resolution,
            scales=dp_pooler_scales,
            sampling_ratio=dp_pooler_sampling_ratio,
            pooler_type=dp_pooler_type,
        )
        self.densepose_head = build_densepose_head(cfg, in_channels)
        self.densepose_predictor = build_densepose_predictor(
            cfg, self.densepose_head.n_out_channels
        )
        self.densepose_losses = build_densepose_losses(cfg)
        self.embedder = build_densepose_embedder(cfg)





    def _forward_densepose(self, features: Dict[str, torch.Tensor], instances: List[Instances]):

        features_list = [features[f] for f in self.in_features]
        
        proposals, _ = select_foreground_proposals(instances, self.num_classes)
        features_list, _= self.densepose_data_filter(features_list, proposals)
        if len(proposals) > 0:
            proposal_boxes = [x.proposal_boxes for x in proposals]
            if self.use_decoder:
                    features_list = [self.decoder(features_list)]

            features_dp = self.densepose_pooler(features_list, proposal_boxes)
            densepose_head_outputs = self.densepose_head(features_dp)
            densepose_predictor_outputs = self.densepose_predictor(densepose_head_outputs)
       
         

            loss={}
            loss['dp_u']=densepose_predictor_outputs.u
            loss['dp_v']=densepose_predictor_outputs.v
            return loss


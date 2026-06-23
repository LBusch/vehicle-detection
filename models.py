import torch
import torch.nn as nn
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights, _utils
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from functools import partial

def load_pretrained_ssdlite(num_classes, detections_per_image, weights_path=None):
    '''Loads an SSDlite model with COCO or specified weights'''

    if weights_path is None:
        weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    else:
        weights = None

    model = ssdlite320_mobilenet_v3_large(weights=weights, detections_per_img=detections_per_image)

    # replace classification head with one that has specified amount of class outputs
    in_channels = _utils.retrieve_out_channels(model.backbone, (320, 320))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDLiteClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
        )
    
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path, weights_only=True, map_location='cpu'))
    model.eval()

    return model

def load_pretrained_fasterrcnn(
        num_classes,
        min_size=800,
        max_size=1333,
        rpn_pre_nms_top_n_train=2000,
        rpn_post_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_test=1000,
        weights_path=None,
        ):
        '''Loads a Faster RCNN model with COCO or specified weights'''

        if weights_path is None:
            weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
        else:
            weights = None

        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=weights,
            min_size=min_size,
            max_size=max_size,
            rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
            rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
            rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
            )
        
        # replace the ROI head with one that has the specified amount of class outputs
        in_channels = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_channels, num_classes)
        
        if weights_path is not None:
            model.load_state_dict(torch.load(weights_path, weights_only=True, map_location='cpu'))
        model.eval()

        return model
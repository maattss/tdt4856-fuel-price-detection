import torch
import torchvision
from torchvision import transforms
# from transforms_custom import Compose, ImageResize, ToTensor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_transform(train):
    trans = []
    trans.append(transforms.ToTensor())

    if train:
        transforms.append(transforms.RandomHorizontalFlip(0.5))
        pass
    return transforms.Compose(trans)

# def get_transform(train):
#    trans = []
# 
#    if train:
#        # trans.append(ImageResize((512, 512)))
#        pass
#
#    trans.append(ToTensor())
#    return Compose(trans)


def collate_fn_custom(batch):
    return tuple(zip(*batch))


def get_model_instance_detection(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model




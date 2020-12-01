#! /usr/bin/env python
import os

import cv2
import rospy
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import random
from object_detection.srv import *
from object_detection.msg import Result
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as SensorImage
from sensor_msgs.msg import RegionOfInterest
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# Global variables
model = None
device = None
threshold = 0.9
rect_th = 3
text_size = 3
text_th = 3
bridge = None
class_names = None

COCO_INSTANCE_CATEGORY_NAMES = [
    'person'
]


def get_prediction(img, threshold):
    global model, device
    transform = transforms.Compose([
        transforms.ToTensor()])
    img = transform(img)  # Apply the transform to the image
    pred = model([img.to(device)])  # Pass the image to the model

    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    # list of indices where element > threshold, pred_score is already sorted in descending order
    # e.g. threshold = 0.3
    # e.g. pred_score = [0.5, 0.4, 0.3, 0.2]
    # e.g. matches = [0, 1, 2]
    matches = [pred_score.index(x) for x in pred_score if x > threshold]

    if len(matches) == 0:
        masks = []
        pred_boxes = []
        class_ids = []
        class_names = []
        return masks, pred_boxes, class_ids, class_names

    # the index of the last element in pred_score list, where element > threshold
    pred_t = matches[-1]
    # pred[0] -> dictionary of the form {boxes: boxes, labels: labels, masks: masks, scores: scores}
    masks = (pred[0]['masks'] > 0.5).squeeze(axis=1).detach().cpu().numpy()

    print(masks)
    print(type(masks))
    # there is only one class -> 'person'
    class_names = [COCO_INSTANCE_CATEGORY_NAMES[0] for i in list(pred[0]['labels'].detach().cpu().numpy())]
    class_ids = pred[0]['labels'].detach().cpu().numpy()
    pred_boxes = [[i[0], i[1], i[2], i[3]] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t + 1]
    masks = np.swapaxes(masks, 0, 2)
    masks = np.swapaxes(masks, 0, 1)
    print(masks.shape)

    pred_boxes = pred_boxes[:pred_t + 1]
    class_ids = class_ids[:pred_t + 1]
    class_names = class_names[:pred_t + 1]
    return masks, pred_boxes, class_names, class_ids


def random_colour_masks(image):
    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
               [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0, 10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def build_result_msg(msg, result):
    result_msg = Result()
    result_msg.header = msg.header
    for i, (x1, y1, x2, y2) in enumerate(result['rois']):
        box = RegionOfInterest()
        box.x_offset = np.asscalar(x1)
        box.y_offset = np.asscalar(y1)
        box.height = np.asscalar(y2 - y1)
        box.width = np.asscalar(x2 - x1)
        result_msg.boxes.append(box)

        class_id = result['class_ids'][i]
        result_msg.class_ids.append(class_id)

        class_name = result['class_names'][i]
        result_msg.class_names.append(class_name)

        mask = result['masks'][:, :, i] * np.uint8(255)
        img_msg = bridge.cv2_to_imgmsg(mask, 'mono8')
        img_msg.header = msg.header
        result_msg.masks.append(img_msg)

    return result_msg

def handle_run_inference(req):
    global model, device, threshold, rect_th, text_size, text_th
    cv_img = bridge.imgmsg_to_cv2(req.img_req, 'bgr8')
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv_img)

    # Run detection
    masks, boxes, class_names, class_ids = get_prediction(img, threshold)
    results = {
        "rois": boxes,
        "class_ids": class_ids,
        "class_names": class_names,
        "masks": masks
    }
    result_msg = build_result_msg(req.img_req, results)

    # visualisation part of result msg
    if len(masks) != 0:
        for i in range(len(masks[0][0])):
            rgb_mask = random_colour_masks(masks[:, :, i])
            cv_img = cv2.addWeighted(cv_img, 1, rgb_mask, 0.5, 0)
            cv2.rectangle(cv_img, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), color=(0, 255, 0), thickness=rect_th)
            cv2.putText(cv_img, class_names[i], (boxes[i][0], boxes[i][1]), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0),
                        thickness=text_th)

    try:
        img_msg = bridge.cv2_to_imgmsg(cv_img, "rgb8")
    except CvBridgeError as e:
        print(e)

    return Mask_RCNNResponse(result_msg, img_msg)


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def init_model():
    global model, device, class_names

    # Now let's instantiate the model and the optimizer
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)
    # move model to the right device
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'maskrcnn.pt')))
    model.eval()
    class_names = rospy.get_param('~class_names', COCO_INSTANCE_CATEGORY_NAMES)
    return model, device, class_names


def run_inference_maskrcnn():
    global model, device, bridge, class_names
    rospy.init_node('run_inference_maskrcnn')

    model, device, class_names = init_model()
    bridge = CvBridge()

    rospy.Service('run_inference_maskrcnn', Mask_RCNN, handle_run_inference)
    rospy.spin()


if __name__ == '__main__':
    run_inference_maskrcnn()

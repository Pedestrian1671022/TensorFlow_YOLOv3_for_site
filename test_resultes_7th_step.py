import cv2
import numpy as np
import tensorflow as tf
import core.utils as utils
from tabulate import tabulate


def load_annotations(annot_path):
    with open(annot_path, 'r') as f:
        lines = f.readlines()
        annotations = [line.strip().split() for line in lines]
    return annotations


def bboxes_iou(boxes1, boxes2):

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file = "yolov3_train_loss.pb"
graph = tf.Graph()
return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

test_annotations = load_annotations("/home/Pedestrian/Documents/TensorFlow_YOLOv3-master/LabelImage_v1.8.1/data/test.txt")
num_classes = 1
input_size = 640

sites = 0
site_loss = 0
site_tp = 0
site_fn = 0
site_alert = 0

with tf.Session(graph=graph) as sess:
    for test_annotation in test_annotations:
        original_image = cv2.imread(test_annotation[0])
        original_bboxes = np.array([list(map(int, box.split(','))) for box in test_annotation[1:]])
        original_image_size = original_image.shape[:2]
        image_data = utils.image_preporcess(original_image.copy(), [input_size, input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = sess.run([return_tensors[1], return_tensors[2], return_tensors[3]], feed_dict={ return_tensors[0]: image_data})
        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)), np.reshape(pred_mbbox, (-1, 5 + num_classes)), np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
        bboxes = utils.nms(bboxes, 0.45, method='nms')

        for i in range(len(original_bboxes)):
            if original_bboxes[i][4] == 0:
                sites += 1

            if len(bboxes) == 0:
                site_loss += 1
                continue
            else:
                ious = bboxes_iou(np.copy(original_bboxes[i][:4]), np.copy(np.array(bboxes)[:, :4]))
                print(ious)
                index = np.argmax(ious)
                iou = ious[index]
                if iou>0.5:
                    if bboxes[index][5]==original_bboxes[i][4]:
                        if bboxes[index][5] == 0:
                            site_tp += 1
                        del bboxes[index]
                    else:
                        if bboxes[index][5] == 0:
                            site_fn += 1
                        del bboxes[index]
                else:
                    if original_bboxes[i][4] == 0:
                        site_loss += 1

        for i in range(len(bboxes)):
            if bboxes[i][5] == 0:
                site_alert += 1


    table = []
    table.append(("site", site_tp, site_fn, site_loss, site_alert, sites))
    x = tabulate(table, headers=["category", "tp", "fn", "loss", "alert", "total"])
    print(x)
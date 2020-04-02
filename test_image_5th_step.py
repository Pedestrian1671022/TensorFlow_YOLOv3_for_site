import cv2
import numpy as np
import tensorflow as tf
import core.utils as utils


def load_annotations(annot_path):
    with open(annot_path, 'r') as f:
        lines = f.readlines()
        annotations = [line.strip().split() for line in lines]
    return annotations


return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file = "yolov3_train_loss.pb"
graph = tf.Graph()
return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

test_annotations = load_annotations("/home/Pedestrian/Documents/TensorFlow_YOLOv3-master/LabelImage_v1.8.1/data/test.txt")
num_classes = 1
input_size = 640


with tf.Session(graph=graph) as sess:
    cv2.namedWindow("test", cv2.WINDOW_AUTOSIZE)
    for test_annotation in test_annotations:
        original_image = cv2.imread(test_annotation[0])
        original_image_size = original_image.shape[:2]
        image_data = utils.image_preporcess(original_image.copy(), [input_size, input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = sess.run([return_tensors[1], return_tensors[2], return_tensors[3]], feed_dict={ return_tensors[0]: image_data})
        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)), np.reshape(pred_mbbox, (-1, 5 + num_classes)), np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
        bboxes = utils.nms(bboxes, 0.45, method='nms')
        image = utils.draw_bbox(original_image, bboxes)
        cv2.imshow("test", image)
        if cv2.waitKey() == 27:
            break
    cv2.destroyAllWindows()
import cv2
import numpy as np
import tensorflow as tf
import core.utils as utils


return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file = "yolov3_train_loss.pb"
graph = tf.Graph()
return_tensors  = utils.read_pb_return_tensors(graph, pb_file, return_elements)

video_path = ".mp4"
input_size = 640
num_classes = 1

with tf.Session(graph=graph) as sess:
    vid = cv2.VideoCapture(video_path)
    cv2.namedWindow("test", cv2.WINDOW_AUTOSIZE)
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print("No image!")
            break
        frame_size = frame.shape[:2]
        image_data = utils.image_preporcess(frame.copy(), [input_size, input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = sess.run([return_tensors[1], return_tensors[2], return_tensors[3]], feed_dict={ return_tensors[0]: image_data})
        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)), np.reshape(pred_mbbox, (-1, 5 + num_classes)), np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
        bboxes = utils.nms(bboxes, 0.45, method='nms')
        image = utils.draw_bbox(frame, bboxes)

        cv2.imshow("test", image)
        if cv2.waitKey(40) == 27:
            break
    cv2.destroyAllWindows()
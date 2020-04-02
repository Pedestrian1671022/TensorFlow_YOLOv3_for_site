import tensorflow as tf
from core.yolov3 import YOLOV3


with tf.name_scope('input'):
    input_data = tf.placeholder(dtype=tf.float32, name='input_data')

YOLOV3(input_data, trainable=False)

sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
ckpt = tf.train.get_checkpoint_state("ckpt")
if ckpt:
    tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
else:
    raise ValueError("The ckpt file is None.")

converted_graph_def = tf.graph_util.convert_variables_to_constants(sess, input_graph_def  = sess.graph.as_graph_def(),
                            output_node_names = ["input/input_data", "pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2"])

with tf.gfile.GFile("yolov3_train_loss.pb", "wb") as f:
    f.write(converted_graph_def.SerializeToString())
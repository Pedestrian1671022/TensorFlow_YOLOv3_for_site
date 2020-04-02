import os
import time
import shutil
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from core import utils
from core.dataset import Dataset
from core.yolov3 import YOLOV3


class YoloTrain(object):
    def __init__(self):
        self.anchor_per_scale    = 3
        self.classes             = utils.read_class_names("/home/Pedestrian/Documents/TensorFlow_YOLOv3-master/LabelImage_v1.8.1/data/predefined_classes.txt")
        self.num_classes         = len(self.classes)
        self.train_epochs        = 160
        self.max_bbox_per_scale  = 150
        self.trainset            = Dataset()
        self.steps_per_period    = len(self.trainset)
        self.sess                = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        with tf.name_scope('define_input'):
            self.input_data   = tf.placeholder(dtype=tf.float32, name='input_data')
            self.label_sbbox  = tf.placeholder(dtype=tf.float32, name='label_sbbox')
            self.label_mbbox  = tf.placeholder(dtype=tf.float32, name='label_mbbox')
            self.label_lbbox  = tf.placeholder(dtype=tf.float32, name='label_lbbox')
            self.true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
            self.true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
            self.true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
            self.trainable    = tf.placeholder(dtype=tf.bool, name='training')

        with tf.name_scope("define_loss"):
            self.model = YOLOV3(self.input_data, self.trainable)
            self.net_var = tf.global_variables()
            self.giou_loss, self.conf_loss, self.prob_loss = self.model.compute_loss(
                                                    self.label_sbbox,  self.label_mbbox,  self.label_lbbox,
                                                    self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)
            self.loss = self.giou_loss + self.conf_loss + self.prob_loss

        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float32, trainable=False, name='global_step')  # step 1: 1.0  ==>   num * self.steps_per_period

            self.update_global_step = tf.assign_add(self.global_step, 1.0)

            self.learn_rate = tf.Variable(1e-4, dtype=tf.float32, trainable=False, name='learn_rate')

            self.update_learn_rate = tf.assign(self.learn_rate, self.learn_rate * 0.8)


        with tf.name_scope("train"):
            self.optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                      var_list=tf.trainable_variables())

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([self.optimizer, self.update_global_step]):
                    self.train_op_with_trainable_variables = tf.no_op()


        with tf.name_scope('summary'):
            tf.summary.scalar("learn_rate", self.learn_rate)
            tf.summary.scalar("giou_loss",  self.giou_loss)
            tf.summary.scalar("conf_loss",  self.conf_loss)
            tf.summary.scalar("prob_loss",  self.prob_loss)
            tf.summary.scalar("total_loss", self.loss)

            logdir = "log"
            if os.path.exists(logdir): shutil.rmtree(logdir)
            os.mkdir(logdir)
            self.write_op = tf.summary.merge_all()
            self.summary_writer  = tf.summary.FileWriter(logdir, graph=self.sess.graph)


    def train(self):
        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state("ckpt")
        if ckpt:
            print('=> Restoring weights from: %s ... ' % ckpt.model_checkpoint_path)
            tf.train.Saver(var_list=tf.trainable_variables()).restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print('=> Now it starts to train YOLOV3 from scratch ...')

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=40)
        for epoch in range(self.train_epochs):
            train_op = self.train_op_with_trainable_variables

            pbar = tqdm(self.trainset)
            train_epoch_loss = []

            # self.sess.run(self.update_learn_rate)

            if epoch % 2 == 0 and epoch != 0:
                self.sess.run(self.update_learn_rate)

            print("learn_rate:", self.sess.run(self.learn_rate))

            for train_data in pbar:
                _, summary, train_step_loss, global_step_val = self.sess.run(
                    [train_op, self.write_op, self.loss, self.global_step],feed_dict={
                                                self.input_data:   train_data[0],
                                                self.label_sbbox:  train_data[1],
                                                self.label_mbbox:  train_data[2],
                                                self.label_lbbox:  train_data[3],
                                                self.true_sbboxes: train_data[4],
                                                self.true_mbboxes: train_data[5],
                                                self.true_lbboxes: train_data[6],
                                                self.trainable:    True,
                })

                train_epoch_loss.append(train_step_loss)
                self.summary_writer.add_summary(summary, global_step_val)
                pbar.set_description("train loss: %.2f" %train_step_loss)

            train_epoch_loss = np.mean(train_epoch_loss)
            ckpt_file = "./ckpt/yolov3_train_loss=%.4f.ckpt" % train_epoch_loss
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Train loss: %.2f Saving %s ..."
                            %(epoch+1, log_time, train_epoch_loss, ckpt_file))
            saver.save(self.sess, ckpt_file, global_step=epoch+1)


if __name__ == '__main__': YoloTrain().train()
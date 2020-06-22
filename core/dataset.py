import os
import cv2
import math
import random
import numpy as np
import tensorflow as tf
from core import utils as utils


class Dataset(object):
    """implement Dataset here"""

    def __init__(self):
        self.annot_path = "/home/Pedestrian/Documents/TensorFlow_YOLOv3-master/LabelImage_v1.8.1/data/train.txt"
        self.batch_size = 2
        self.data_aug = True  # 防止过拟合

        # self.train_input_sizes = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]  #防止过拟合
        self.train_input_sizes = [1024]
        self.strides = np.array([8, 16, 32])
        self.classes = utils.read_class_names(
            "/home/Pedestrian/Documents/TensorFlow_YOLOv3-master/LabelImage_v1.8.1/data/predefined_classes.txt")
        self.num_classes = len(self.classes)
        self.anchors = np.array(
            utils.get_anchors(
                "/home/Pedestrian/Documents/TensorFlow_YOLOv3-master/LabelImage_v1.8.1/data/basline_anchors.txt"))
        self.anchor_per_scale = 3
        self.max_bbox_per_scale = 150

        self.annotations = self.load_annotations()
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def load_annotations(self):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        return annotations

    def __iter__(self):
        return self

    def __next__(self):

        with tf.device('/cpu:0'):
            self.train_input_size = random.choice(self.train_input_sizes)
            self.train_output_sizes = self.train_input_size // self.strides

            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3))

            batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                          self.anchor_per_scale, 5 + self.num_classes))
            batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                          self.anchor_per_scale, 5 + self.num_classes))
            batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                          self.anchor_per_scale, 5 + self.num_classes))

            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))

            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples: index -= self.num_samples
                    annotation = self.annotations[index]
                    image, bboxes = self.parse_annotation(annotation)
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(
                        bboxes)

                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1
                self.batch_count += 1
                return batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
                       batch_sbboxes, batch_mbboxes, batch_lbboxes
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def random_crop(self, image_path, image, bboxes):
        width = 1024
        height = 1024
        padding = 5
        h, w, _ = image.shape
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = max_bbox[2]
        max_d_trans = max_bbox[3]

        if min(max_l_trans - padding, w - width) < max(0, max_l_trans - (
                width - ((max_r_trans - max_l_trans) + padding))) or min(max_u_trans - padding, h - height) < max(0,
                                                                                                                  max_u_trans - (
                                                                                                                          height - (
                                                                                                                          (
                                                                                                                                  max_d_trans - max_u_trans) + padding))):
            # raise Exception('image error:', image_path)
            print(image_path)

        crop_x = int(random.uniform(max(0, max_l_trans - (width - ((max_r_trans - max_l_trans) + padding))),
                                    min(max_l_trans - padding, w - width)))
        crop_y = int(random.uniform(max(0, max_u_trans - (height - ((max_d_trans - max_u_trans) + padding))),
                                    min(max_u_trans - padding, h - height)))

        image = image[crop_y: crop_y + height, crop_x: crop_x + width]

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_x
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_y

        return image, bboxes

    def random_horizontal_flip(self, image, bboxes):
        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1]
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
        return image, bboxes

    def random_erasing(self, image, bboxes, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        if random.random() < 0.5:
            for attempt in range(100):
                area = image.shape[0] * image.shape[1]

                target_area = random.uniform(sl, sh) * area
                aspect_ratio = random.uniform(r1, 1 / r1)

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < image.shape[1] and h < image.shape[0]:
                    x1 = random.randint(0, image.shape[0] - h)
                    y1 = random.randint(0, image.shape[1] - w)
                    if image.shape[2] == 3:
                        image[x1:x1 + h, y1:y1 + w, 0] = mean[0]
                        image[x1:x1 + h, y1:y1 + w, 1] = mean[1]
                        image[x1:x1 + h, y1:y1 + w, 2] = mean[2]
                    else:
                        image[x1:x1 + h, y1:y1 + w, 0] = mean[0]
                    return image, bboxes
        return image, bboxes

    def hide_patch(self, image, bboxes):
        if random.random() < 0.5:
            # get width and height of the image
            h = image.shape[0]
            w = image.shape[1]

            # possible grid size, 0 means no hiding
            grid_sizes = [0, 16, 32, 44, 56]

            # hiding probability
            hide_prob = 0.5

            # randomly choose one grid size
            grid_size = grid_sizes[random.randint(0, len(grid_sizes) - 1)]

            # hide the patches
            if (grid_size != 0):
                for y in range(0, h, grid_size):
                    for x in range(0, w, grid_size):
                        y_end = min(h, y + grid_size)
                        x_end = min(w, x + grid_size)
                        if (random.random() <= hide_prob):
                            image[y:y_end, x:x_end, :] = 0

        return image, bboxes

    def grid_mask(self, image, bboxes):
        if random.random() < 0.5:
            dy = 20
            dx = 20
            ry = 0.5
            rx = 0.5
            starty = random.randint(0, dy - dy * ry - 1)
            startx = random.randint(0, dx - dx * rx - 1)

            # get width and height of the image
            h = image.shape[0]
            w = image.shape[1]

            for y in range(0, h - dx, dy):
                for x in range(0, w - dy, dx):
                    y_start = y + starty
                    x_start = x + startx
                    y_end = int(min(h, y_start + dy * ry))
                    x_end = int(min(w, x_start + dx * rx))
                    image[y_start:y_end, x_start:x_end, :] = 0

        return image, bboxes

    def affine_translate(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    def perspective_translate(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            pts2 = np.float32([[0 + w * 0.1, h * 0.1], [w - w * 0.1, h * 0.1], [0, h], [w, h]])
            # pts3 = np.float32([[0, 0], [w, 0], [w*0.1, h-h*0.1], [w-w*0.1, h-h*0.1]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            # M = cv2.getPerspectiveTransform(pts1, pts3)
            image = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            bboxes1, bboxes2 = bboxes[:, :4], bboxes[:, 4:]
            new_bboxes = cv2.perspectiveTransform(bboxes1.reshape(1, 2 * len(bboxes1), 2).astype(np.float), M)
            new_bboxes = new_bboxes.reshape(len(bboxes1), 4).astype(np.int)
            bboxes = np.hstack((new_bboxes, bboxes2))
        return image, bboxes

    def parse_annotation(self, annotation):

        line = annotation.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)
        image = np.array(cv2.imread(image_path))
        bboxes = np.array([list(map(int, box.split(','))) for box in line[1:]])

        if self.data_aug:
            image, bboxes = self.random_crop(image_path, np.copy(image), np.copy(bboxes))
            # image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            # image, bboxes = self.random_erasing(np.copy(image), np.copy(bboxes))
            # image, bboxes = self.hide_patch(np.copy(image), np.copy(bboxes))
            # image, bboxes = self.grid_mask(np.copy(image), np.copy(bboxes))
            # image, bboxes = self.affine_translate(np.copy(image), np.copy(bboxes))
            image, bboxes = self.perspective_translate(np.copy(image), np.copy(bboxes))

        image, bboxes = utils.image_preporcess(np.copy(image), [self.train_input_size, self.train_input_size],
                                               np.copy(bboxes))
        return image, bboxes

    def bbox_iou(self, boxes1, boxes2):

        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                 boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                 boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / union_area

    def preprocess_true_boxes(self, bboxes):

        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batchs

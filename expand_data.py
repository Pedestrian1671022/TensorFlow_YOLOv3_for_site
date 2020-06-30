import os
import cv2
import math
import random
import colorsys
import numpy as np


def random_crop(width, height, padding, image_path, image, bboxes):
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


def random_horizontal_flip(image, bboxes):
    if random.random() < 0.5:
        _, w, _ = image.shape
        image = image[:, ::-1]
        bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
    return image, bboxes


def random_erasing(image, bboxes, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
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


def hide_patch(image, bboxes):
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


def grid_mask(image, bboxes):
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


def affine_translate(image, bboxes):
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


def perspective_translate(image, bboxes):
    if random.random() < 0.5:
        h, w, _= image.shape
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([[0 + w*0.1, h*0.1], [w-w*0.1, h*0.1], [0, h], [w, h]])
        # pts3 = np.float32([[0, 0], [w, 0], [w*0.1, h-h*0.1], [w-w*0.1, h-h*0.1]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        # M = cv2.getPerspectiveTransform(pts1, pts3)
        image = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        bboxes1, bboxes2 = bboxes[:, :4], bboxes[:, 4:]
        new_bboxes = cv2.perspectiveTransform(bboxes1.reshape(1, 2 * len(bboxes1), 2).astype(np.float), M)
        new_bboxes = new_bboxes.reshape(len(bboxes1), 4).astype(np.int)
        bboxes = np.hstack((new_bboxes, bboxes2))
    return image, bboxes


def mixup(image, bboxes):
    if random.random() < 0.5:
        annot_path = "/home/Pedestrian/Documents/TensorFlow_YOLOv3_for_site/LabelImage_v1.8.1/data/train.txt"
        annotations = load_annotations(annot_path)
        image1, bboxes1 = image, bboxes
        index = random.randint(0, len(annotations)-1)
        line = annotations[index].split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)
        image2 = np.array(cv2.imread(image_path))
        bboxes2 = np.array([list(map(int, box.split(','))) for box in line[1:]])
        image2, bboxes2 = random_crop(1024, 1024, 5, image_path, np.copy(image2), np.copy(bboxes2))
        image2, bboxes2 = random_horizontal_flip(np.copy(image2), np.copy(bboxes2))
        alpha = random.uniform(0.3, 0.7)
        beta = random.uniform(0.3, 0.7)
        image = cv2.addWeighted(image1, alpha, image2, beta, 0)
        bboxes = np.vstack((bboxes1, bboxes2))
    return image, bboxes

def mosaic(image, bboxes):
    if random.random() < 0.5:
        annot_path = "/home/Pedestrian/Documents/TensorFlow_YOLOv3_for_site/LabelImage_v1.8.1/data/train.txt"
        annotations = load_annotations(annot_path)
        image1, bboxes1 = random_crop(512, 512, 0, "mosaic", image, bboxes)

        index2 = random.randint(0, len(annotations)-1)
        line2 = annotations[index2].split()
        image_path2 = line2[0]
        if not os.path.exists(image_path2):
            raise KeyError("%s does not exist ... " % image_path2)
        image2 = np.array(cv2.imread(image_path2))
        bboxes2 = np.array([list(map(int, box.split(','))) for box in line2[1:]])
        image2, bboxes2 = random_crop(512, 512, 0, image_path2, np.copy(image2), np.copy(bboxes2))
        image2, bboxes2 = random_horizontal_flip(np.copy(image2), np.copy(bboxes2))
        bboxes2[:, [1, 3]] = bboxes2[:, [1, 3]] + 512

        index3 = random.randint(0, len(annotations) - 1)
        line3 = annotations[index3].split()
        image_path3 = line3[0]
        if not os.path.exists(image_path3):
            raise KeyError("%s does not exist ... " % image_path3)
        image3 = np.array(cv2.imread(image_path3))
        bboxes3 = np.array([list(map(int, box.split(','))) for box in line3[1:]])
        image3, bboxes3 = random_crop(512, 512, 0, image_path3, np.copy(image3), np.copy(bboxes3))
        image3, bboxes3 = random_horizontal_flip(np.copy(image3), np.copy(bboxes3))
        bboxes3[:, [0, 2]] = bboxes3[:, [0, 2]] + 512

        index4 = random.randint(0, len(annotations) - 1)
        line4 = annotations[index4].split()
        image_path4 = line4[0]
        if not os.path.exists(image_path4):
            raise KeyError("%s does not exist ... " % image_path4)
        image4 = np.array(cv2.imread(image_path4))
        bboxes4 = np.array([list(map(int, box.split(','))) for box in line4[1:]])
        image4, bboxes4 = random_crop(512, 512, 0, image_path4, np.copy(image4), np.copy(bboxes4))
        image4, bboxes4 = random_horizontal_flip(np.copy(image4), np.copy(bboxes4))
        bboxes4[:, [0, 1, 2, 3]] = bboxes4[:, [0, 1, 2, 3]] + 512

        image_1 = np.concatenate((image1, image2), axis=0)
        image_2 = np.concatenate((image3, image4), axis=0)
        image = np.concatenate((image_1, image_2), axis=1)
        bboxes = np.vstack((bboxes1, bboxes2, bboxes3, bboxes4))
    return image, bboxes


def load_annotations(annot_path):
    with open(annot_path, 'r') as f:
        txt = f.readlines()
        annotations = [line.strip() for line in txt if len(line.strip().split()[2:]) != 0]
    # np.random.shuffle(annotations)
    return annotations


def parse_annotation(annotation):

    line = annotation.split()
    image_path = line[0]
    if not os.path.exists(image_path):
        raise KeyError("%s does not exist ... " % image_path)
    image = np.array(cv2.imread(image_path))
    bboxes = np.array([list(map(int, box.split(','))) for box in line[1:]])

    image, bboxes = random_crop(1024, 1024, 5, image_path, np.copy(image), np.copy(bboxes))
    image, bboxes = random_horizontal_flip(np.copy(image), np.copy(bboxes))
    # image, bboxes = random_erasing(np.copy(image), np.copy(bboxes))
    # image, bboxes = hide_patch(np.copy(image), np.copy(bboxes))
    # image, bboxes = grid_mask(np.copy(image), np.copy(bboxes))
    # image, bboxes = affine_translate(np.copy(image), np.copy(bboxes))
    # image, bboxes = perspective_translate(np.copy(image), np.copy(bboxes))
    # image, bboxes = mixup(np.copy(image), np.copy(bboxes))
    image, bboxes = mosaic(np.copy(image), np.copy(bboxes))
    return image, bboxes


def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def draw_bbox(image, bboxes, classes=read_class_names(
    "/home/Pedestrian/Documents/TensorFlow_YOLOv3_for_site/LabelImage_v1.8.1/data/predefined_classes.txt"),
              show_label=True):

    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        class_ind = int(bbox[4])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        image = cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s' % (classes[class_ind])
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            image = cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            image = cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

    return image


if __name__ == '__main__':
    annot_path = "/home/Pedestrian/Documents/TensorFlow_YOLOv3_for_site/LabelImage_v1.8.1/data/train.txt"
    annotations = load_annotations(annot_path)
    for annotation in annotations:
        image, bboxes = parse_annotation(annotation)
        new_image = draw_bbox(image, bboxes)
        cv2.imshow("test", new_image)
        if cv2.waitKey() == 27:
            break
    cv2.destroyAllWindows()

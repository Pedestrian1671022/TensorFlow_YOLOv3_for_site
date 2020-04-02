import os
import cv2
import random
import argparse
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom.minidom import Document


width = 640
height = 512
padding = 20

def random_crop(image_path, image, bboxes):
    print(image_path)
    h, w, _ = image.shape
    max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

    max_l_trans = max_bbox[0]
    max_u_trans = max_bbox[1]
    max_r_trans = max_bbox[2]
    max_d_trans = max_bbox[3]

    if min(max_l_trans-padding, w-width)<max(0, max_l_trans - (width - ((max_r_trans - max_l_trans) + padding))) or min(max_u_trans-padding, h-height)<max(0, max_u_trans - (height - ((max_d_trans - max_u_trans) + padding))):
        raise Exception('image error:', image_path)

    crop_x = int(random.uniform(max(0, max_l_trans - (width - ((max_r_trans - max_l_trans) + padding))), min(max_l_trans-padding, w-width)))
    crop_y = int(random.uniform(max(0, max_u_trans - (height - ((max_d_trans - max_u_trans) + padding))), min(max_u_trans-padding, h-height)))

    image = image[crop_y: crop_y + height, crop_x: crop_x + width]

    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_x
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_y

    return image, bboxes


def expand_dataset(data_path, original_xmls, original_pictures, xmls, pictures):

    image_inds = [xml[0:-4] for xml in os.listdir(os.path.join(data_path, original_xmls))]

    for image_ind in image_inds:
        label_path = os.path.join(data_path, original_xmls, image_ind + '.xml')
        print(image_ind)
        root = ET.parse(label_path).getroot()
        objects = root.findall('object')
        bboxes = []
        class_inds = []
        for obj in objects:
            bbox = obj.find('bndbox')
            class_ind = obj.find('name').text.lower().strip()
            xmin = bbox.find('xmin').text.strip()
            xmax = bbox.find('xmax').text.strip()
            ymin = bbox.find('ymin').text.strip()
            ymax = bbox.find('ymax').text.strip()
            bboxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
            class_inds.append(class_ind)
        image_path = os.path.join(data_path, original_pictures, image_ind + '.jpg')
        for i in range(4):
            original_image = cv2.imread(image_path)
            new_image, new_bboxes = random_crop(image_path, np.copy(original_image), np.copy(bboxes))

            doc = Document()

            annotation = doc.createElement('annotation')

            filename = doc.createElement('filename')
            filename_name = doc.createTextNode(image_ind + '_' + str(i) + ".jpg")
            filename.appendChild(filename_name)
            annotation.appendChild(filename)

            size = doc.createElement('size')
            width = doc.createElement('width')
            width.appendChild(doc.createTextNode(str(width)))
            height = doc.createElement('height')
            height.appendChild(doc.createTextNode(str(height)))
            depth = doc.createElement('depth')
            depth.appendChild(doc.createTextNode(str(3)))
            size.appendChild(width)
            size.appendChild(height)
            size.appendChild(depth)
            annotation.appendChild(size)
            
            for j in range(len(class_inds)):
                object = doc.createElement('object')

                name = doc.createElement('name')
                name.appendChild(doc.createTextNode(class_ind))
                object.appendChild(name)
                
                bndbox = doc.createElement('bndbox')
                xmin = doc.createElement('xmin')
                xmin.appendChild(doc.createTextNode(str(new_bboxes[j][0])))
                bndbox.appendChild(xmin)
                ymin = doc.createElement('ymin')
                ymin.appendChild(doc.createTextNode(str(new_bboxes[j][1])))
                bndbox.appendChild(ymin)
                xmax = doc.createElement('xmax')
                xmax.appendChild(doc.createTextNode(str(new_bboxes[j][2])))
                bndbox.appendChild(xmax)
                ymax = doc.createElement('ymax')
                ymax.appendChild(doc.createTextNode(str(new_bboxes[j][3])))
                bndbox.appendChild(ymax)

                object.appendChild(bndbox)
                annotation.appendChild(object)

            doc.appendChild(annotation)
            xml_file = open(os.path.join(data_path, xmls, image_ind + '_' + str(i) + '.xml'), 'w')
            xml_file.write(doc.toprettyxml())
            xml_file.close()
            cv2.imwrite(os.path.join(data_path, pictures, image_ind + '_' + str(i) + '.jpg') ,new_image)
    return len(image_inds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/home/Pedestrian/Documents/TensorFlow_YOLOv3-master/LabelImage_v1.8.1/data")
    flags = parser.parse_args()

    num = expand_dataset(flags.data_path, 'original_xmls', 'original_pictures', 'xmls', 'pictures')
    print('=> The number of image for train is: %d' %num)
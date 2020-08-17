import os
import cv2
import argparse
import xml.etree.ElementTree as ET


def xml_to_text(data_path, xmls, jpgs, txts):

    classes = ['none_of_the_above', 'car', 'airport', 'ship', 'storagetank', 'bridge', 'plane']
    image_inds = [xml[0:-4] for xml in os.listdir(os.path.join(data_path, xmls)) if xml.endswith('.xml')]

    for image_ind in image_inds:
        image_path = os.path.join(data_path, jpgs, image_ind + '.jpg')
        image = cv2.imread(image_path)
        h, w, _ = image.shape

        xml_path = os.path.join(data_path, xmls, image_ind + '.xml')
        root = ET.parse(xml_path).getroot()
        objects = root.findall('object')

        text_path = os.path.join(data_path, txts, image_ind + '.txt')
        with open(text_path, 'a') as f:
            for obj in objects:
                bbox = obj.find('bndbox')
                class_ind = classes.index(obj.find('name').text.lower().strip())
                xmin = bbox.find('xmin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymin = bbox.find('ymin').text.strip()
                ymax = bbox.find('ymax').text.strip()
                txt = ' '.join([str(class_ind), str((float(xmax) + float(xmin))/2/w), str((float(ymax) + float(ymin))/2/h), str((float(xmax) - float(xmin))/w), str((float(ymax) - float(ymin))/h)])
                f.write(txt + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/home/pc405/Documents/TensorFlow_YOLOv3_for_site/data")
    flags = parser.parse_args()

    xml_to_text(flags.data_path, 'Annotations', 'JPEGImages', 'Labels')

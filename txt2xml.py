import os
import cv2
import xml.etree.ElementTree as ET



if __name__ == '__main__':
    data_path = "/home/Pedestrian/Documents/TensorFlow_YOLOv3_for_site/LabelImage_v1.8.1/data"

    jpgs = 'jpgs'
    txts = 'txts'
    xmls = 'xmls'

    image_inds = [jpg[0:-4] for jpg in os.listdir(os.path.join(data_path, jpgs))]

    for image_ind in image_inds:
        image_path = os.path.join(data_path, jpgs, image_ind + '.jpg')
        image = cv2.imread(image_path)
        h, w, d = image.shape
        root = ET.Element('annotation')
        filename = ET.SubElement(root, 'filename')
        filename.text = image_ind + '.jpg'
        size = ET.SubElement(root, 'size')
        height = ET.SubElement(size, 'height')
        height.text = str(h)
        width = ET.SubElement(size, 'width')
        width.text = str(w)
        depth = ET.SubElement(size, 'depth')
        depth.text = str(d)
        text_path = os.path.join(data_path, txts, image_ind + '.txt')
        with open(text_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                annotation = list(map(float, line.split()[1:]))
                object_name = str(int(line.split()[0]))
                object_xmin = str(int((annotation[0] - annotation[2] / 2) * w))
                object_ymin = str(int((annotation[1] - annotation[3] / 2) * h))
                object_xmax = str(int((annotation[0] + annotation[2] / 2) * w))
                object_ymax = str(int((annotation[1] + annotation[3] / 2) * h))
                object = ET.SubElement(root, 'object')
                name = ET.SubElement(object, 'name')
                name.text = object_name
                bndbox = ET.SubElement(object, 'bndbox')
                xmin = ET.SubElement(bndbox, 'xmin')
                xmin.text = object_xmin
                ymin = ET.SubElement(bndbox, 'ymin')
                ymin.text = object_ymin
                xmax = ET.SubElement(bndbox, 'xmax')
                xmax.text = object_xmax
                ymax = ET.SubElement(bndbox, 'ymax')
                ymax.text = object_ymax
        tree = ET.ElementTree(root)
        tree.write(os.path.join(data_path, xmls, image_ind + '.xml'), encoding='utf-8', xml_declaration=True)
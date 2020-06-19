import os
import cv2
import random
import argparse
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom.minidom import Document



def expand_dataset(data_path, pictures):

    image_inds = [path for path in os.listdir(os.path.join(data_path, pictures))]
    txt = ''
    for image_ind in image_inds:
        txt = txt + "build/darknet/x64/data/obj/" + image_ind + '\n'
    xml_file = open('train.txt', 'w')
    xml_file.write(txt)
    xml_file.close()
    return len(image_inds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/home/Pedestrian/Documents/TensorFlow_YOLOv3-master/LabelImage_v1.8.1/data")
    flags = parser.parse_args()

    num = expand_dataset(flags.data_path, 'pictures')
    print('=> The number of image for train is: %d' %num)
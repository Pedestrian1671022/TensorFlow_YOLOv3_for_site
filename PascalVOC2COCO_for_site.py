# -*- coding:utf-8 -*-
# !/usr/bin/env python
import sys
import glob
import json
import xml.etree.ElementTree as ET


class PascalVOC2coco(object):
    def __init__(self, xmls, json_file):
        '''
        :param xmls: 所有Pascal VOC的xml文件路径组成的列表
        :param save_json_path: json保存位置
        '''
        self.xmls = xmls
        self.json_file = json_file
        self.images = []
        self.categories = []
        self.annotations = []
        self.label = []
        self.height = 0
        self.width = 0
        self.annID = 0

        self.save_json()

    def data_transfer(self):
        for num, xml in enumerate(self.xmls):

            # 进度输出
            sys.stdout.write('\r>> Converting image %d/%d' % (
                num + 1, len(self.xmls)))
            sys.stdout.flush()

            self.num = num

            root = ET.parse(xml).getroot()

            self.file_name =root.find('filename')

            size = root.find('size')
            self.width = int(size.find('width').text)
            self.height = int(size.find('height').text)

            self.images.append(self.image())

            objects = root.findall('object')
            for obj in objects:
                self.supercategory = obj.find('name').text
                if self.supercategory not in self.label:
                    self.categories.append(self.category())
                    self.label.append(self.supercategory)

                bndbox = obj.find('bndbox')
                x1 = int(bndbox.find('xmin').text)
                y1 = int(bndbox.find('ymin').text)
                x2 = int(bndbox.find('xmax').text)
                y2 = int(bndbox.find('xmax').text)

                self.bbox = [x1, y1, x2 - x1, y2 - y1]  # COCO 对应格式[x,y,w,h]

                self.annID += 1
                self.annotations.append(self.annotation())

        sys.stdout.write('\n')
        sys.stdout.flush()

    def image(self):
        image = {}
        image['height'] = self.height
        image['width'] = self.width
        image['id'] = self.num + 1
        image['file_name'] = self.file_name.text
        return image

    def category(self):
        category = {}
        category['supercategory'] = self.supercategory
        category['id'] = len(self.label) + 1  # 0 默认为背景
        category['name'] = self.supercategory
        return category

    def annotation(self):
        annotation = {}
        annotation['iscrowd'] = 0
        annotation['image_id'] = self.num + 1
        annotation['bbox'] = self.bbox
        annotation['category_id'] = self.getcatid(self.supercategory)
        annotation['id'] = self.annID
        return annotation

    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie['name']:
                return categorie['id']
        return -1

    # '''
    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        # 保存json文件
        json.dump(self.data_coco, open(self.json_file, 'w'), indent=4)  # indent=4 更加美观显示


xmls = glob.glob('/home/Pedestrian/Documents/TensorFlow_YOLOv3-master/LabelImage_v1.8.1/data/xmls/*.xml')

PascalVOC2coco(xmls, './sites.json')

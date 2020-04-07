# -*- coding:utf-8 -*-
# !/usr/bin/env python
import os
import sys
import glob
import json


class PascalVOC2coco(object):
    def __init__(self, xmls, pictures, json_file):
        '''
        :param xmls: 所有Pascal VOC的xml文件路径组成的列表
        :param save_json_path: json保存位置
        '''
        self.xmls = xmls
        self.pictures = pictures
        self.json_file = json_file
        self.images = []
        self.categories = []
        self.annotations = []
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0

        self.save_json()

    def data_transfer(self):
        for num, xml in enumerate(self.xmls):

            # 进度输出
            sys.stdout.write('\r>> Converting image %d/%d' % (
                num + 1, len(self.xmls)))
            sys.stdout.flush()

            self.num = num
            path = os.path.dirname(xml)
            path = os.path.dirname(path)

            with open(xml, 'r') as fp:
                for p in fp:
                    if 'filename' in p:
                        self.file_name = p.split('>')[1].split('<')[0]

                        self.picture = os.path.join(path, 'pictures', self.file_name.split('.')[0] + '.jpg')
                        if self.picture not in self.pictures:
                            break


                    if 'width' in p:
                        self.width = int(p.split('>')[1].split('<')[0])
                    if 'height' in p:
                        self.height = int(p.split('>')[1].split('<')[0])

                        self.images.append(self.image())

                    if '<object>' in p:
                        # 类别
                        d = [next(fp).split('>')[1].split('<')[0] for _ in range(9)]
                        # self.supercategory = d[0]
                        self.supercategory = 'site'
                        if self.supercategory not in self.label:
                            self.categories.append(self.category())
                            self.label.append(self.supercategory)

                        # 边界框
                        x1 = int(d[-4])
                        y1 = int(d[-3])
                        x2 = int(d[-2])
                        y2 = int(d[-1])
                        self.rectangle = [x1, y1, x2, y2]
                        self.bbox = [x1, y1, x2 - x1, y2 - y1]  # COCO 对应格式[x,y,w,h]

                        self.annotations.append(self.annotation())
                        self.annID += 1

        sys.stdout.write('\n')
        sys.stdout.flush()

    def image(self):
        image = {}
        image['height'] = self.height
        image['width'] = self.width
        image['id'] = self.num + 1
        image['file_name'] = self.file_name
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
pictures = glob.glob('/home/Pedestrian/Documents/TensorFlow_YOLOv3-master/LabelImage_v1.8.1/data/pictures/*.jpg')

PascalVOC2coco(xmls, pictures, './sites.json')

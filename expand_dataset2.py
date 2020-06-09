import os
import cv2
import random
import argparse
import xml.etree.ElementTree as ET


width = 1024
height = 1024


def prettyXml(element, indent, newline, level = 0): # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
    if element:  # 判断element是否有子元素
        if element.text == None or element.text.isspace(): # 如果element的text没有内容
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
    #else:  # 此处两行如果把注释去掉，Element的text也会另起一行
        #element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element) # 将elemnt转成list
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1): # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        prettyXml(subelement, indent, newline, level = level + 1) # 对子元素进行递归操作

def expand_dataset(data_path, original_xmls, original_pictures, original_txts, xmls, pictures, txts):

    image_inds = [xml[0:-4] for xml in os.listdir(os.path.join(data_path, original_xmls))]

    for image_ind in image_inds:
        xmins = []
        ymins = []
        xmaxs = []
        ymaxs = []
        label_path = os.path.join(data_path, original_xmls, image_ind + '.xml')
        new_xml = ET.parse(label_path)
        root = new_xml.getroot()
        objects = root.findall('object')
        for obj in objects:
            bbox = obj.find('bndbox')
            xmin = bbox.find('xmin').text.strip()
            xmax = bbox.find('xmax').text.strip()
            ymin = bbox.find('ymin').text.strip()
            ymax = bbox.find('ymax').text.strip()
            xmins.append(int(xmin))
            ymins.append(int(ymin))
            xmaxs.append(int(xmax))
            ymaxs.append(int(ymax))

        original_image = cv2.imread(os.path.join(data_path, original_pictures, image_ind + '.jpg'))

        txt = open(os.path.join(data_path, original_txts, image_ind + '.txt')).read()

        for i in range(len(objects)):
            for _ in range(20):
                transfer_size = [150, 100, 50, -50, -100, -150]
                new_xmin = xmins[i] + random.choice(transfer_size)
                new_ymin = ymins[i] + random.choice(transfer_size)
                new_xmax = new_xmin + xmaxs[i] - xmins[i]
                new_ymax = new_ymin + ymaxs[i] - ymins[i]
                if new_xmax > width or new_ymax > height or new_xmin < 0 or new_ymin < 0:
                    continue
                is_contine = True
                for m in range(len(objects)):
                    if (xmins[m] < new_xmin < xmaxs[m] and ymins[m] < new_ymin < ymaxs[m]) or (xmins[m] < new_xmax < xmaxs[m] and ymins[m] < new_ymin < ymaxs[m]) or (xmins[m] < new_xmin < xmaxs[m] and ymins[m] < new_ymax < ymaxs[m]) or (xmins[m] < new_xmax < xmaxs[m] and ymins[m] < new_ymax < ymaxs[m]):
                        is_contine = False
                if not is_contine:
                    continue
                if new_xmax < xmins[i] or new_xmin > xmaxs[i] or new_ymax < ymins[i] or new_ymin > ymaxs[i]:
                    original_image[new_ymin:new_ymax, new_xmin:new_xmax, :] = original_image[ymins[i]:ymaxs[i], xmins[i]:xmaxs[i], :]

                    object = ET.Element('object')

                    name = ET.SubElement(object, 'name')
                    name.text = str(1)
                    txt = txt + str(0)

                    bndbox = ET.SubElement(object, 'bndbox')
                    xmin = ET.SubElement(bndbox, 'xmin')
                    xmin.text = str(new_xmin)
                    txt = txt + ' ' + str((int(new_xmin + new_xmax)) / 2 / 1024)
                    ymin = ET.SubElement(bndbox, 'ymin')
                    ymin.text = str(new_ymin)
                    txt = txt + ' ' + str((int(new_ymin + new_ymax)) / 2 / 1024)
                    xmax = ET.SubElement(bndbox, 'xmax')
                    xmax.text = str(new_xmax)
                    txt = txt + ' ' + str(int(new_xmax - new_xmin) / 1024)
                    ymax = ET.SubElement(bndbox, 'ymax')
                    ymax.text = str(new_ymax)
                    txt = txt + ' ' + str((int(new_ymax - new_ymin)) / 1024)

                    root.append(object)
                    prettyXml(root, '    ', '\n')
                    txt = txt + '\n'
                    break

        cv2.imwrite(os.path.join(data_path, pictures, image_ind + str(i) + '.jpg'), original_image)
        txt_file = open(os.path.join(data_path, txts, image_ind + str(i) + '.txt'), 'w')
        txt_file.write(txt)
        xml_file = open(os.path.join(data_path, xmls, image_ind + str(i) + '.xml'), 'wb')
        new_xml.write(xml_file, encoding="utf-8", xml_declaration=True)
    return len(image_inds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/home/Pedestrian/Documents/TensorFlow_YOLOv3-master/LabelImage_v1.8.1/data")
    flags = parser.parse_args()

    num = expand_dataset(flags.data_path, '_xmls', '_pictures', '_txts', '__xmls', '__pictures', '__txts')
    print('=> The number of image for train is: %d' %num)
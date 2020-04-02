import os
import argparse
import xml.etree.ElementTree as ET


def xmls_to_text(data_path, xmls, pictures, train_annotations, test_annotations):

    classes = ['site']
    image_inds = [xml[0:-4] for xml in os.listdir(os.path.join(data_path, xmls))]

    split_num = int(len(image_inds) * 0.0)

    image_inds_train = image_inds[:split_num]

    image_inds_test = image_inds[split_num:]

    with open(train_annotations, 'a') as f:
        for image_ind in image_inds_train:
            image_path = os.path.join(data_path, pictures, image_ind + '.jpg')
            annotation = image_path
            print(annotation)
            label_path = os.path.join(data_path, xmls, image_ind + '.xml')
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            for obj in objects:
                bbox = obj.find('bndbox')
                class_ind = classes.index(obj.find('name').text.lower().strip())
                xmin = bbox.find('xmin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymin = bbox.find('ymin').text.strip()
                ymax = bbox.find('ymax').text.strip()
                annotation += ' ' + ','.join([str(int(float(xmin))), str(int(float(ymin))), str(int(float(xmax))), str(int(float(ymax))), str(class_ind)])
            f.write(annotation + "\n")
    with open(test_annotations, 'a') as f:
        for image_ind in image_inds_test:
            image_path = os.path.join(data_path, pictures, image_ind + '.jpg')
            annotation = image_path
            label_path = os.path.join(data_path, xmls, image_ind + '.xml')
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            for obj in objects:
                bbox = obj.find('bndbox')
                class_ind = classes.index(obj.find('name').text.lower().strip())
                xmin = bbox.find('xmin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymin = bbox.find('ymin').text.strip()
                ymax = bbox.find('ymax').text.strip()
                annotation += ' ' + ','.join([str(int(float(xmin))), str(int(float(ymin))), str(int(float(xmax))), str(int(float(ymax))), str(class_ind)])
            print(annotation)
            f.write(annotation + "\n")
    return len(image_inds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/home/Pedestrian/Documents/TensorFlow_YOLOv3-master/LabelImage_v1.8.1/data")
    parser.add_argument("--train_annotations", default="/home/Pedestrian/Documents/TensorFlow_YOLOv3-master/LabelImage_v1.8.1/data/train.txt")
    parser.add_argument("--test_annotations", default="/home/Pedestrian/Documents/TensorFlow_YOLOv3-master/LabelImage_v1.8.1/data/test.txt")
    flags = parser.parse_args()

    if os.path.exists(flags.train_annotations):os.remove(flags.train_annotations)
    if os.path.exists(flags.test_annotations): os.remove(flags.test_annotations)

    num = xmls_to_text(flags.data_path, 'xmls', 'pictures', flags.train_annotations, flags.test_annotations)
    print('=> The number of image is: %d' %num)
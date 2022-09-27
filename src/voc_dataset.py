#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Faster-RCNN 
@File    ：voc_dataset.py
@IDE     ：PyCharm 
@Author  ：离开流沙河的坚定大土豆
@Date    ：2022/9/19 20:26 
'''


from __future__ import division

import os
import numpy as np
from numpy import random

# import mmcv
import mindspore as ms
from mindspore import Tensor
import mindspore.dataset as ds
from config import config
import cv2
import albumentations as A
from lxml import etree
import json
from PIL import Image

class VocDateSet:
    def __init__(self, transforms, is_training=True):
        # 设置数据集路径
        self.voc_root = config.voc_root
        self.img_root = os.path.join(self.voc_root, "JPEGImages")
        self.annotations_root = os.path.join(self.voc_root, "Annotations")
        # 读取train.txt or val.txt
        if is_training:
            txt_path = os.path.join(self.voc_root, "ImageSets", "Main", "train.txt")
        else:
            txt_path = os.path.join(self.voc_root, "ImageSets", "Main", "val.txt")
        # 根据路径读取txt文件，并将结果保存在 xml_list 中
        with open(txt_path) as read:
            xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                        for line in read.readlines() if len(line.strip()) > 0]

        # 检查xml_list文件，排除不合适的数据，将过滤的结果保存在self.xml_list
        self.xml_list = []
        for xml_path in xml_list:
            if os.path.exists(xml_path) is False:
                print(f"Warning: not found '{xml_path}', skip this annotation file.")
                continue
            # 如果存在这个xml文件
            with open(xml_path) as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = self.parse_xml_to_dict(xml)["annotation"]
            # 如果xml文件中不包含object信息，就排除这个数据
            if "object" not in data:
                print(f"INFO: no objects in {xml_path}, skip this annotation file.")
                continue
            self.xml_list.append(xml_path)

        # 读取 json 文件，class_indict
        json_file_path = config.json_file_path
        assert os.path.exists(json_file_path), "{} file not exist.".format(json_file_path)
        with open(json_file_path, 'r') as f:
            self.class_dict = json.load(f)

        # 保存数据增强
        self.transforms = transforms

    def __getitem__(self, idx):
        # 读取第 idx 个xml文件
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        img_path = os.path.join(self.img_root, data["filename"])
        image = cv2.imread(img_path)
        image = Image.open(img_path).convert('RGB')
        # if image.format != "JPEG":
        #     raise ValueError("Image '{}' format not JPEG".format(img_path))

        boxes = []
        labels = []
        iscrowd = []
        assert "object" in data, "{} lack of object information.".format(xml_path)
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # convert everything into a torch.Tensor
        boxes = Tensor(boxes, ms.float32)
        labels = Tensor(labels, ms.int64)
        iscrowd = Tensor(iscrowd, ms.int64)
        image_id = Tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image

    def __len__(self):
        return len(self.xml_list)

    @property
    def column_names(self):
        column_names = ['image']
        return column_names

    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree
        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

def create_dataset(batch_size, transforms, is_training, shuffle):
    vocDataSet = VocDateSet(transforms=transforms, is_training=is_training)
    print("数据个数：", len(vocDataSet))
    dataset = ds.GeneratorDataset(vocDataSet, vocDataSet.column_names, shuffle=shuffle, python_multiprocessing=False)
    dataset = dataset.batch(batch_size, num_parallel_workers=1)
    return dataset

transforms = None
res = create_dataset(batch_size=1, transforms=transforms, is_training=True, shuffle=True)
for image in res:
    print(image)
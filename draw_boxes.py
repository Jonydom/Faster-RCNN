#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Faster-RCNN 
@File    ：draw_boxes.py
@IDE     ：PyCharm 
@Author  ：离开流沙河的坚定大土豆
@Date    ：2022/9/22 19:10 
'''

import os
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

json_path = './minicoco2017/annotations/train2017.json'
img_path = './minicoco2017/train2017'
# 创建coco类读取json数据
coco = COCO(annotation_file=json_path)
# 获得所有图片的index信息
ids = list(sorted(coco.imgs.keys()))
print("number of images: {}".format(len(ids)))

coco_classes = dict([(v["id"], v["name"]) for k, v in coco.cats.items()])

for img_id in ids[:3]:
    ann_ids = coco.getAnnIds(imgIds=img_id)
    targets = coco.loadAnns(ann_ids)
    path = coco.loadImgs(img_id)[0]['file_name']

    img = Image.open(os.path.join(img_path, path)).convert('RGB')
    draw = ImageDraw.Draw(img)

    for target in targets:
        x, y, w, h = target["bbox"]
        x1, y1, x2, y2 = x, y, int(x+w), int(y+h)
        draw.rectangle((x1, y1, x2, y2),outline ='red',width =1)
        draw.text((x1, y1), coco_classes[target["category_id"]])

    plt.imshow(img)
    plt.show()
    # img.save('./{}.jpg'.format(img_id))
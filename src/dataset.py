"""FasterRcnn dataset"""
from __future__ import division

import os
import numpy as np
from numpy import random

# import mmcv
import mindspore.dataset as de
import mindspore.dataset.vision.c_transforms as C
from mindspore.mindrecord import FileWriter
from config import config
import cv2


def create_coco_label(is_training):
    """Get image path and annotation from COCO."""
    from pycocotools.coco import COCO

    coco_root = config.coco_root   # coco_root: "./coco2017/"
    data_type = config.val_data_type  # data_type: "val2017"
    if is_training:
        data_type = config.train_data_type  # data_type: "train2017"

    # Classes need to train or test.
    train_cls = config.coco_classes  # train_cls: ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    train_cls_dict = {}
    for i, cls in enumerate(train_cls):
        train_cls_dict[cls] = i   # train_cls_dict: {'background': 0, 'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 'traffic light': 10, 'fire hydrant': 11, 'stop sign': 12, 'parking meter': 13, 'bench': 14, 'bird': 15, 'cat': 16, 'dog': 17, 'horse': 18, 'sheep': 19, 'cow': 20, 'elephant': 21, 'bear': 22, 'zebra': 23, 'giraffe': 24, 'backpack': 25, 'umbrella': 26, 'handbag': 27, 'tie': 28, 'suitcase': 29, 'frisbee': 30, 'skis': 31, 'snowboard': 32, 'sports ball': 33, 'kite': 34, 'baseball bat': 35, 'baseball glove': 36, 'skateboard': 37, 'surfboard': 38, 'tennis racket': 39, 'bottle': 40, 'wine glass': 41, 'cup': 42, 'fork': 43, 'knife': 44, 'spoon': 45, 'bowl': 46, 'banana': 47, 'apple': 48, 'sandwich': 49, 'orange': 50, 'broccoli': 51, 'carrot': 52, 'hot dog': 53, 'pizza': 54, 'donut': 55, 'cake': 56, 'chair': 57, 'couch': 58, 'potted plant': 59, 'bed': 60, 'dining table': 61, 'toilet': 62, 'tv': 63, 'laptop': 64, 'mouse': 65, 'remote': 66, 'keyboard': 67, 'cell phone': 68, 'microwave': 69, 'oven': 70, 'toaster': 71, 'sink': 72, 'refrigerator': 73, 'book': 74, 'clock': 75, 'vase': 76, 'scissors': 77, 'teddy bear': 78, 'hair drier': 79, 'toothbrush': 80}

    anno_json = os.path.join(coco_root, config.instance_set.format(data_type))  # anno_json: './coco2017/annotations/instances_train2017.json'
    
    # 根据annotations json文件创建COCO类
    coco = COCO(anno_json)
    classes_dict = {}
    cat_ids = coco.loadCats(coco.getCatIds())  # 获取所有的类别信息，loadCats()需要传入 需加载的类别id序列
    for cat in cat_ids:
        classes_dict[cat["id"]] = cat["name"]  # classes_dict: {1:'person', 2:'bicycle'...}

    image_ids = coco.getImgIds()  # 获取所有 标记所对应的原图id  image_ids: [391895, 522418...]

    # 创建要返回的变量
    image_files = []
    image_anno_dict = {}

    for img_id in image_ids:
        image_info = coco.loadImgs(img_id)  # coco.loadImgs返回一个list，list[0]包含图片的具体信息
        file_name = image_info[0]["file_name"]
        anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)  # 获取该图片的所有标注的id（指定的标注）
        anno = coco.loadAnns(anno_ids)  # 返回list，记录了每个segmentation(分割)在图片中的坐标位置以及其他信息
        image_path = os.path.join(coco_root, data_type, file_name)
        annos = []
        for label in anno:
            bbox = label["bbox"]
            class_name = classes_dict[label["category_id"]]
            if class_name in train_cls:
                x1, x2 = bbox[0], bbox[0] + bbox[2]
                y1, y2 = bbox[1], bbox[1] + bbox[3]
                annos.append([x1, y1, x2, y2] + [train_cls_dict[class_name]] + [int(label["iscrowd"])])

        image_files.append(image_path)
        if annos:
            image_anno_dict[image_path] = np.array(annos)
        else:
            image_anno_dict[image_path] = np.array([0, 0, 0, 0, 0, 1])

        # image_files: ['./coco2017/train2017/000000391895.jpg'...]
        # image_anno_dict: {'./coco2017/train2017/000000391895.jpg': np.array[x1,y1,x2,y2,class_id,iscrowd]}
    return image_files, image_anno_dict


def data_to_mindrecord_byte_image(dataset="coco", is_training=True, prefix="fasterrcnn.mindrecord", file_num=8):
    pass

create_coco_label(True)
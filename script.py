#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Faster-RCNN 
@File    ：script.py
@IDE     ：PyCharm 
@Author  ：离开流沙河的坚定大土豆
@Date    ：2022/9/21 10:22 
'''


# Fast script for the creation of a sub-set of the coco dataset in the form of a data folder.

import json
from pycocotools.coco import COCO
import wget
import numpy as np
from random import sample
from pathlib import Path
from joblib import delayed, Parallel

ANNOTATIONS = {"info": {
    "description": "minicoco2017"
}
}

def myImages(images: list, train: int, val: int) -> tuple:
    myImagesTrain = images[:train]
    myImagesVal = images[train:train+val]
    return myImagesTrain, myImagesVal


def cocoJson(images: list) -> dict:
    arrayIds = np.array([k["id"] for k in images])
    annIds = coco.getAnnIds(imgIds=arrayIds, catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    for k in anns:
        k["category_id"] = catIds.index(k["category_id"])+1
    catS = [{'id': int(value), 'name': key}
            for key, value in categories.items()]
    ANNOTATIONS["images"] = images
    ANNOTATIONS["annotations"] = anns
    ANNOTATIONS["categories"] = catS

    return ANNOTATIONS


def createJson(JsonFile: json, train: bool) -> None:
    name = "train"
    if not train:
        name = "val"
    Path("minicoco2017/annotations").mkdir(parents=True, exist_ok=True)
    with open(f"minicoco2017/annotations/{name}2017.json", "w") as outfile:
        json.dump(JsonFile, outfile)


def downloadImagesToTrain(img: dict) -> None:
    link = (img['coco_url'])
    Path("minicoco2017/train2017").mkdir(parents=True, exist_ok=True)
    wget.download(link, f"{'minicoco2017/train2017/' + img['file_name']}")

def downloadImagesToVal(img: dict) -> None:
    link = (img['coco_url'])
    Path("minicoco2017/val2017").mkdir(parents=True, exist_ok=True)
    wget.download(link, f"{'minicoco2017/val2017/' + img['file_name']}")

# Instantiate COCO specifying the annotations json path; download here: https://cocodataset.org/#download
coco = COCO('./coco2017/annotations/instances_train2017.json')

# Specify a list of category names of interest
catNms = ['car', 'airplane', 'person']

catIds = coco.getCatIds(catNms)  # catIds: [1, 3, 5]

dictCOCO = {k: coco.getCatIds(k)[0] for k in catNms}  # dictCOCO: {'car': 3, 'airplane': 5, 'person': 1}
dictCOCOSorted = dict(sorted(dictCOCO.items(), key=lambda x: x[1]))  # dictCOCOSorted: {'person': 1, 'car': 3, 'airplane': 5}

IdCategories = list(range(1, len(catNms)+1))  # IdCategories: [1, 2, 3]
categories = dict(zip(list(dictCOCOSorted), IdCategories))  # categories: {'person': 1, 'car': 2, 'airplane': 3}

# getCatIds return a sorted list of id.
# For the creation of the json file in coco format, the list of ids must be successive 1, 2, 3..
# So we reorganize the ids. In the cocoJson method we modify the values of the category_id parameter.

# Get the corresponding image ids and images using loadImgs
imgIds = coco.getImgIds(catIds=catIds)  # 根据物体类别得id号，得到训练集中对应img的id，这里一共173张
imgOriginals = coco.loadImgs(imgIds)  # 返回list数组，数组中包含173个字典

# The images are selected randomly
imgShuffled = sample(imgOriginals, len(imgOriginals))  # 进行图片顺序打乱

# Choose the number of images for the training and validation set. default 30-10
myImagesTrain, myImagesVal = myImages(imgShuffled, 30, 10)  # imgShuffled前30个图片作为训练集，31-40作为验证集

trainSet = cocoJson(myImagesTrain)
createJson(trainSet, train=True)

valSet = cocoJson(myImagesVal)
createJson(valSet, train=False)

Parallel(
    n_jobs=-1, prefer="threads")([delayed(downloadImagesToTrain)(img) for img in myImagesTrain])

Parallel(
    n_jobs=-1, prefer="threads")([delayed(downloadImagesToVal)(img) for img in myImagesVal])

print("\nfinish.")
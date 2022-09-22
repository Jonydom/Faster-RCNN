# 基于MindSpore框架的Faster-RCNN案例实现

## 1 模型简介

Faster-RCNN模型于2016年在论文《Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks》中被提出，它的提出是为了改进Fast-RCNN区域候选算法运行时间长的问题。

Faster-RCNN模型是在Fast-RCNN模型的基础上建立的，由于目标检测网络依靠区域候选算法（如Selective Search）来假设目标的位置，运算时间开销很大，于是Faster-RCNN提出了一个可以共享卷积特征图的深度全卷积网络RPN来代替区域候选算法，使用RPN网络产生的候选区域进行分类与边框回归计算，从而大大加快了运行速度。

### 1.1 模型结构

Faster R-CNN是个两阶段的目标检测方法，主要由提取候选区域的全卷积网络RPN与Fast R-CNN检测器组成，整个检测过程通过一个网络完成。RPN和Fast R-CNN的配合作用可以理解为一种注意力机制，先大致确定目标在视野中的位置，然后再锁定目标仔细观察，确定目标的类别和更加准确的位置。 图1为论文中给出的Faster-RCNN基本结构图。

![image text](https://github.com/514forever/IMG/blob/main/%E5%9B%BE%E7%89%871.png)

Faster R-CNN检测部分主要可以分为以下四个模块：

（1）Conv layers特征提取网络，采用VGG16、ResNet等常用结构作为特征提取的模块，用于提取输入图像特征。然后将提取得到的图像特征feature maps用于后续的RPN层生成一系列可能的候选框。

（2）RPN候选检测框生成网络，该网络替代了之前Fast R-CNN版本的Selective Search，用于生成候选框proposal ，输出为一系列候选框以及每个候选框框中目标的概率值。

（3）RoI Pooling兴趣域池化，以RPN网络输出的兴趣区域和Conv layers输出的图像特征为输入，将两者进行综合后得到固定大小的区域特征图，后续将其送入全连接层继续做目标分类和坐标回归。

（4）Classification and Regression分类与回归。利用上一层得到的区域特征图通过softmax对图像进行分类，并通过边框回归修正物体的精确位置，输出兴趣区域中物体所属的类别以及物体在图像中精确的位置。

### 1.2 模型特点

使用RPN来生成候选区域，完全使用CNN解决目标检测任务，将特征提取、候选框选取、边框回归和分类都整合到一个网络中。

## 2 案例实现

### 2.1 环境准备与数据读取
本案例基于MindSpore-CPU版本实现，在CPU上完成模型训练。

案例实现所使用的数据来自MS coco2017数据集，由于coco2017数据集数据量太大，故经过采样脚本对其进行裁剪，生成minicoco2017数据集，其包括3个文件夹，分别对应标签、训练集样本、验证集样本，文件路径结构如下：

```
.minicoco2017/
├── annotations
│   ├── train2017.json
│   ├── val2017.json
├── train2017
│   ├── 000000001311.jpg
│   ├── 000000030345.jpg
│   └── ......
└── val2017
    ├── 000000078469.jpg
    ├── 000000099598.jpg
    └── ......
```

其中，annotations文件中有两个json文件，分为对应训练集和验证集的标签数据；train2017文件夹中包含30张训练图片，val2017文件夹中包含10张验证图片。minicoco2017数据集从coco2017数据集的80个分类中选择了3个分类：person、airplane、car。

具体裁剪的实现方式：首先读取coco2017中目标检测标注文件instances_train2017.json，选择指定的三个分类；其次，根据这三个分类的id选择与其相关的所有图片，再对这些图片进行随机采样，选择30张作为训练集，选择10张作为验证集；最后，根据40张图片的id找出它们对应的标注信息。将上述图片和标注信息按照coco数据集文件的排列方式存储在本地。

```python3
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
```
![](https://cdn.jsdelivr.net/gh/Jonydom/myPic/img/1311.jpg)
![](https://cdn.jsdelivr.net/gh/Jonydom/myPic/img/30345.jpg)
图2-3 训练集样本及其对应标签

### 2.2 数据集创建

### 2.3 模型构建


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
在进行上述coco数据集裁剪后，就完成了数据集准备工作。接下来数据集创建的工作分为两部分：

1. 读取minicoco2017数据集中训练集的标签信息，将图片信息和标注信息转换为华为mindrecord数据格式，生成文件`./MindRecord_COCO_TRAIN/FasterRcnn.mindrecord`和`./MindRecord_COCO_TRAIN/FasterRcnn.mindrecord.db`;

​	具体实现方法：使用`pycocotools`工具读取minicoco2017数据集中训练集的标注文件`/minicoco2017/annotations/train2017.json`，获取所有的类别信息，存为字典`classes_dict`，获取所有图片的id，根据每个图片的id值，查找该图片的地址和所有标注的id，然后检索这个图片中所有的标注框信息（x, y, w, h, iscrowd）。根据`x, y, w, h`计算标注框的左上角和右下角的位置坐标`(x1, y1), (x2, y2)`。最后根据图片的文件路径，将对应的图片二进制信息、标注信息（x1, y1, x2, y2, class_id, iscrowd）存入mindrecord文件中。

2. 使用`mindspore.dataset.MindDataset`创建数据集，读取生成的mindrecord数据文件，对数据进行数据增强操作。

​	具体实现方法：使用`mindspore.dataset.MindDataset`创建数据集，读取生成的mindrecord数据文件，先将`image`二进制图片进行`Decode()`解码，转换为RGB模式，然后对数据再进行expand、rescale、imnormalize、flip、transpose等操作。

```python3
def data_to_mindrecord_byte_image(config, dataset="coco", is_training=True, prefix="fasterrcnn.mindrecord", file_num=1):
    """Create MindRecord file."""
    mindrecord_dir = config.mindrecord_dir  # mindrecord_dir: "./MindRecord_COCO_TRAIN"
    mindrecord_path = os.path.join(mindrecord_dir, prefix)  # mindrecord_file: "/MindRecord_COCO_TRAIN/FasterRcnn.mindrecord0"
    writer = FileWriter(mindrecord_path, file_num)
    if dataset == "coco":
        image_files, image_anno_dict = create_coco_label(is_training, config=config)
    else:
        image_files, image_anno_dict = create_train_data_from_txt(config.image_dir, config.anno_path)

    fasterrcnn_json = {
        "image": {"type": "bytes"},
        "annotation": {"type": "int32", "shape": [-1, 6]},
    }
    writer.add_schema(fasterrcnn_json, "fasterrcnn_json")

    for image_name in image_files:
        with open(image_name, 'rb') as f:
            img = f.read()
        annos = np.array(image_anno_dict[image_name], dtype=np.int32)
        row = {"image": img, "annotation": annos}
        writer.write_raw_data([row])
    writer.commit()


def create_fasterrcnn_dataset(config, mindrecord_file, batch_size=2, device_num=1, rank_id=0, is_training=True,
                              num_parallel_workers=8, python_multiprocessing=False):
    """Create FasterRcnn dataset with MindDataset."""
    cv2.setNumThreads(0)
    de.config.set_prefetch_size(1)
    ds = de.MindDataset(mindrecord_file, columns_list=["image", "annotation"], num_shards=device_num, shard_id=rank_id,
                        num_parallel_workers=4, shuffle=is_training)
    decode = ms.dataset.vision.Decode()
    ds = ds.map(input_columns=["image"], operations=decode)
    compose_map_func = (lambda image, annotation: preprocess_fn(image, annotation, is_training, config=config))

    if is_training:
        ds = ds.map(input_columns=["image", "annotation"],
                    output_columns=["image", "image_shape", "box", "label", "valid_num"],
                    column_order=["image", "image_shape", "box", "label", "valid_num"],
                    operations=compose_map_func, python_multiprocessing=python_multiprocessing,
                    num_parallel_workers=num_parallel_workers)
        ds = ds.batch(batch_size, drop_remainder=True)
    else:
        ds = ds.map(input_columns=["image", "annotation"],
                    output_columns=["image", "image_shape", "box", "label", "valid_num"],
                    column_order=["image", "image_shape", "box", "label", "valid_num"],
                    operations=compose_map_func,
                    num_parallel_workers=num_parallel_workers)
        ds = ds.batch(batch_size, drop_remainder=True)
    return ds

```


### 2.3 模型构建


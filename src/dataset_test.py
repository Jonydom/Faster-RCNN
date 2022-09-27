"""FasterRcnn dataset"""
from __future__ import division

import os
import numpy as np
from numpy import random

# import mmcv
import mindspore as ms
import mindspore.dataset as de
from mindspore.mindrecord import FileWriter
from config import config
import cv2

class Expand:
    """expand image"""
    def __init__(self, mean=(0, 0, 0), to_rgb=True, ratio_range=(1, 4)):
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range

    def __call__(self, img, boxes, labels):
        if random.randint(2):
            return img, boxes, labels

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img
        img = expand_img
        boxes += np.tile((left, top), 2)
        return img, boxes, labels

def expand_column(img, img_shape, gt_bboxes, gt_label, gt_num):
    """expand operation for image"""
    expand = Expand()
    img, gt_bboxes, gt_label = expand(img, gt_bboxes, gt_label)

    return (img, img_shape, gt_bboxes, gt_label, gt_num)

def rescale_column(img, img_shape, gt_bboxes, gt_label, gt_num):
    """rescale operation for image"""
    img_data, scale_factor = rescale_with_tuple(img, (config.img_width, config.img_height))
    if img_data.shape[0] > config.img_height:
        img_data, scale_factor2 = rescale_with_tuple(img_data, (config.img_height, config.img_height))
        scale_factor = scale_factor * scale_factor2

    gt_bboxes = gt_bboxes * scale_factor
    gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_data.shape[1] - 1)
    gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_data.shape[0] - 1)

    pad_h = config.img_height - img_data.shape[0]
    pad_w = config.img_width - img_data.shape[1]
    assert ((pad_h >= 0) and (pad_w >= 0))

    pad_img_data = np.zeros((config.img_height, config.img_width, 3)).astype(img_data.dtype)
    pad_img_data[0:img_data.shape[0], 0:img_data.shape[1], :] = img_data

    img_shape = (config.img_height, config.img_width, 1.0)
    img_shape = np.asarray(img_shape, dtype=np.float32)

    return (pad_img_data, img_shape, gt_bboxes, gt_label, gt_num)

def rescale_with_tuple(img, scale):
    h, w = img.shape[:2]
    scale_factor = min(max(scale) / max(h, w), min(scale) / min(h, w))
    new_size = int(w * float(scale_factor) + 0.5), int(h * float(scale_factor) + 0.5)
    rescaled_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

    return rescaled_img, scale_factor
def resize_column(img, img_shape, gt_bboxes, gt_label, gt_num):
    """resize operation for image"""
    img_data = img
    h, w = img_data.shape[:2]
    img_data = cv2.resize(
        img_data, (config.img_width, config.img_height), interpolation=cv2.INTER_LINEAR)
    h_scale = config.img_height / h
    w_scale = config.img_width / w

    scale_factor = np.array(
        [w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
    img_shape = (config.img_height, config.img_width, 1.0)
    img_shape = np.asarray(img_shape, dtype=np.float32)

    gt_bboxes = gt_bboxes * scale_factor

    gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1] - 1)
    gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0] - 1)

    return (img_data, img_shape, gt_bboxes, gt_label, gt_num)

def imnormalize_column(img, img_shape, gt_bboxes, gt_label, gt_num):
    """imnormalize operation for image"""
    # Computed from random subset of ImageNet training images
    mean = np.asarray([123.675, 116.28, 103.53])
    std = np.asarray([58.395, 57.12, 57.375])
    img_data = img.copy().astype(np.float32)
    cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB, img_data)  # inplace
    cv2.subtract(img_data, np.float64(mean.reshape(1, -1)), img_data)  # inplace
    cv2.multiply(img_data, 1 / np.float64(std.reshape(1, -1)), img_data)  # inplace

    img_data = img_data.astype(np.float32)
    return (img_data, img_shape, gt_bboxes, gt_label, gt_num)

def flip_column(img, img_shape, gt_bboxes, gt_label, gt_num):
    """flip operation for image"""
    img_data = img
    img_data = np.flip(img_data, axis=1)
    flipped = gt_bboxes.copy()
    _, w, _ = img_data.shape

    flipped[..., 0::4] = w - gt_bboxes[..., 2::4] - 1
    flipped[..., 2::4] = w - gt_bboxes[..., 0::4] - 1

    return (img_data, img_shape, flipped, gt_label, gt_num)

def transpose_column(img, img_shape, gt_bboxes, gt_label, gt_num):
    """transpose operation for image"""
    img_data = img.transpose(2, 0, 1).copy()
    img_data = img_data.astype(np.float32)
    img_shape = img_shape.astype(np.float32)
    gt_bboxes = gt_bboxes.astype(np.float32)
    gt_label = gt_label.astype(np.int32)
    gt_num = gt_num.astype(np.bool)

    return (img_data, img_shape, gt_bboxes, gt_label, gt_num)

def resize_column_test(img, img_shape, gt_bboxes, gt_label, gt_num):
    """resize operation for image of eval"""
    img_data = img
    h, w = img_data.shape[:2]
    img_data = cv2.resize(
        img_data, (config.img_width, config.img_height), interpolation=cv2.INTER_LINEAR)
    h_scale = config.img_height / h
    w_scale = config.img_width / w

    scale_factor = np.array(
        [w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
    img_shape = np.append(img_shape, (h_scale, w_scale))
    img_shape = np.asarray(img_shape, dtype=np.float32)

    gt_bboxes = gt_bboxes * scale_factor

    gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1] - 1)
    gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0] - 1)

    return (img_data, img_shape, gt_bboxes, gt_label, gt_num)


def preprocess_fn(image, box, is_training):
    """Preprocess function for dataset."""
    def _infer_data(image_bgr, image_shape, gt_box_new, gt_label_new, gt_iscrowd_new_revert):
        image_shape = image_shape[:2]
        input_data = image_bgr, image_shape, gt_box_new, gt_label_new, gt_iscrowd_new_revert

        if config.keep_ratio:
            input_data = rescale_column(*input_data)
        else:
            input_data = resize_column_test(*input_data)
        input_data = imnormalize_column(*input_data)

        output_data = transpose_column(*input_data)
        return output_data

    def _data_aug(image, box, is_training):
        """Data augmentation function."""
        image_bgr = image.copy()
        image_bgr[:, :, 0] = image[:, :, 2]
        image_bgr[:, :, 1] = image[:, :, 1]
        image_bgr[:, :, 2] = image[:, :, 0]
        image_shape = image_bgr.shape[:2]
        gt_box = box[:, :4]
        gt_label = box[:, 4]
        gt_iscrowd = box[:, 5]

        pad_max_number = config.num_gts  # num_gts: 128
        gt_box_new = np.pad(gt_box, ((0, pad_max_number - box.shape[0]), (0, 0)), mode="constant", constant_values=0)
        gt_label_new = np.pad(gt_label, (0, pad_max_number - box.shape[0]), mode="constant", constant_values=-1)
        gt_iscrowd_new = np.pad(gt_iscrowd, (0, pad_max_number - box.shape[0]), mode="constant", constant_values=1)
        gt_iscrowd_new_revert = (~(gt_iscrowd_new.astype(np.bool))).astype(np.int32)

        if not is_training:
            return _infer_data(image_bgr, image_shape, gt_box_new, gt_label_new, gt_iscrowd_new_revert)

        flip = (np.random.rand() < config.flip_ratio)
        expand = (np.random.rand() < config.expand_ratio)
        input_data = image_bgr, image_shape, gt_box_new, gt_label_new, gt_iscrowd_new_revert

        if expand:
            input_data = expand_column(*input_data)
        if config.keep_ratio:
            input_data = rescale_column(*input_data)
        else:
            input_data = resize_column(*input_data)
        input_data = imnormalize_column(*input_data)
        if flip:
            input_data = flip_column(*input_data)

        output_data = transpose_column(*input_data)
        return output_data

    return _data_aug(image, box, is_training)

def create_coco_label(is_training):
    """Get image path and annotation from COCO."""
    from pycocotools.coco import COCO

    coco_root = config.coco_root   # coco_root: "../coco2017/"
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

        # image_files: ['../coco2017/train2017/000000391895.jpg'...]
        # image_anno_dict: {'../coco2017/train2017/000000391895.jpg': np.array[x1,y1,x2,y2,class_id,iscrowd]}
    return image_files, image_anno_dict


def data_to_mindrecord_byte_image(config, dataset="coco", is_training=True, prefix="fasterrcnn.mindrecord", file_num=8):
    "Create MindRecord file"
    mindrecord_dir = config.mindrecord_dir  # mindrecord_dir: "../MindRecord_COCO_TRAIN"
    mindrecord_path = os.path.join(mindrecord_dir, prefix)
    writer = FileWriter(mindrecord_path, file_num)  # mindspore下的函数：将用户自定义的数据转为MindRecord格式数据集的类（文件路径,生成MindRecord的文件个数）
    if dataset == "coco":
        image_files, image_anno_dict = create_coco_label(is_training)
    # else:
    #     image_files, image_anno_dict = filter_valid_data(config.IMAGE_DIR, config.ANNO_PATH)

    fasterrcnn_json = {
        "image": {"type": "bytes"},
        "annotation": {"type": "int32", "shape": [-1, 6]},
    }  # fasterrcnn_json是定义的存储的格式
    writer.add_schema(fasterrcnn_json, "fasterrcnn_json")

    for image_name in image_files:
        with open(image_name, 'rb') as f:
            img = f.read()
        annos = np.array(image_anno_dict[image_name], dtype=np.int32)
        row = {"image": img, "annotation": annos}
        writer.write_raw_data([row])
    writer.commit()

def create_fasterrcnn_dataset(mindrecord_file, batch_size=2, device_num=1, rank_id=0, is_training=True, num_parallel_workers=8):
    """Create FasterRcnn dataset with MindDataset."""
    cv2.setNumThreads(0)
    de.config.set_prefetch_size(8)  # 设置预取数据size or 设置管道中线程的队列容量。
    ds = de.MindDataset(mindrecord_file, columns_list=["image", "annotation"], num_shards=device_num, shard_id=rank_id,
                        num_parallel_workers=4, shuffle=is_training)
    decode = ms.dataset.vision.Decode()  # Decode()类，将输入的压缩图像解码为RGB格式

    ds = ds.map(input_columns=["image"], operations=decode)  # Apply each operation in operations to this dataset.
    compose_map_func = (lambda image, annotation: preprocess_fn(image, annotation, is_training))

    if is_training:
        ds = ds.map(input_columns=["image", "annotation"],
                    output_columns=["image", "image_shape", "box", "label", "valid_num"],
                    column_order=["image", "image_shape", "box", "label", "valid_num"],
                    operations=compose_map_func, python_multiprocessing=False,
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
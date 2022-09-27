"""train FasterRcnn and get checkpoint files."""

import os
import time
from pprint import pprint
import numpy as np
import mindspore as ms
from mindspore import Tensor, Parameter
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.nn import SGD, Adam
from mindspore.common import set_seed
from mindspore.train.callback import SummaryCollector
from src.FasterRcnn.faster_rcnn import Faster_Rcnn
from src.network_define import LossCallBack, WithLossCell, TrainOneStepCell, LossNet
from src.dataset import data_to_mindrecord_byte_image, create_fasterrcnn_dataset
from src.lr_schedule import dynamic_lr, multistep_lr
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id


def train_fasterrcnn_():
    """ train_fasterrcnn_ """
    print("Start create dataset!")

    # It will generate mindrecord file in config.mindrecord_dir,
    # and the file name is FasterRcnn.mindrecord0, 1, ... file_num.
    prefix = "FasterRcnn.mindrecord"  # mindrecord文件前缀
    mindrecord_dir = config.mindrecord_dir  # mindrecord_dir: "/MindRecord_COCO_TRAIN"
    mindrecord_file = os.path.join(mindrecord_dir, prefix)  # mindrecord_file: "/MindRecord_COCO_TRAIN/FasterRcnn.mindrecord0"
    print("CHECKING MINDRECORD FILES ...")

    if rank == 0 and not os.path.exists(mindrecord_file + ".db"):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if config.dataset == "coco":
            print(config.coco_root)
            if os.path.isdir(config.coco_root):
                if not os.path.exists(config.coco_root):
                    print("Please make sure config:coco_root is valid.")
                    raise ValueError(config.coco_root)
                print("Create Mindrecord. It may take some time.")
                data_to_mindrecord_byte_image(config, "coco", True, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("coco_root not exits.")
        else:
            if os.path.isdir(config.image_dir) and os.path.exists(config.anno_path):
                if not os.path.exists(config.image_dir):
                    print("Please make sure config:image_dir is valid.")
                    raise ValueError(config.image_dir)
                print("Create Mindrecord. It may take some time.")
                data_to_mindrecord_byte_image(config, "other", True, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("image_dir or anno_path not exits.")


    while not os.path.exists(mindrecord_file + ".db"):
        time.sleep(5)

    print("CHECKING MINDRECORD FILES DONE!")

    # When create MindDatacreate_fasterrcnn_datasetset, using the fitst mindrecord file, such as FasterRcnn.mindrecord0.
    dataset = create_fasterrcnn_dataset(config, mindrecord_file, batch_size=config.batch_size, device_num=device_num, rank_id=rank, num_parallel_workers=config.num_parallel_workers, python_multiprocessing=config.python_multiprocessing)

    dataset_size = dataset.get_dataset_size()
    print("Create dataset done!")

    return dataset_size, dataset


def train_fasterrcnn():
    """ train_fasterrcnn """
    print(f"\n[{rank}] - rank id of process")
    dataset_size, dataset = train_fasterrcnn_()  # 主要是对coco数据进行处理，生成dataset

    print(f"\n[{rank}]", "===> Creating network...")
    net = Faster_Rcnn(config=config)
    net = net.set_train()


if __name__ == '__main__':
    set_seed(1)  # Set global seed.used by numpy.random, mindspore.common.Initializer, mindspore.ops.composite.random_ops and mindspore.nn.probability.distribution.
    # 设置运行环境的context，在GRAPH_MODE模式下运行，device_target支持CPU，device_id为0
    ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target, device_id=get_device_id())

    local_path = '/'.join(os.path.realpath(__file__).split('\\')[:-1])  # local_path: 'D:/project/Faster-RCNN'

    ## 取消了源代码中 GPU 和 分布式 的代码
    rank = 0
    device_num = 1

    pprint(config)  # 打印config
    print(f"\n[{rank}] Please check the above information for the configurations\n\n", flush=True)

    # 开始训练
    train_fasterrcnn()
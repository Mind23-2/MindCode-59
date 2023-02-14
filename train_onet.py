import argparse
import sys
import os
import ast
import math
from mindspore import context
sys.path.append(os.getcwd())
from mindspore.train.model import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
import mindspore.dataset as ds
from mindspore.nn import SGD, Adam
import mindspore.common.initializer as weight_init
import mindspore.nn as nn
import numpy as np
from mindspore import Tensor

from src.imagedb import ImageDB, data_to_mindrecord_byte_image, create_mtcnn_dataset
from src.config import config
from network.mtcnn import ONetWithLoss
#from src.image_reader import TrainImageReader
#from mindspore.nn import learning_rate_schedule as lr_schedules


def parse_args():
    parser = argparse.ArgumentParser(description='Train PNet')

    parser.add_argument('--anno_file', help='training data annotation file',
                        default="/root/xidian_wks/mtcnn/anno_strore/pos_48_copy.txt", type=str)
    parser.add_argument('--model_path', help='training model store directory',
                        default='/root/xidian_wks/mtcnn/model_store/sgd/', type=str)
    parser.add_argument('--end_epoch', help='end epoch of training',
                        default=config.END_EPOCH, type=int)
    parser.add_argument('--mindrecord_dir', help='the path of saving mindrecord',
                        default="/root/xidian_wks/mtcnn/data_set", type=str)
    parser.add_argument('--frequent', help='frequency of logging',
                        default=200, type=int)
    parser.add_argument('--lr', dest='lr', help='learning rate',
                        default=config.TRAIN_LR, type=float)
    parser.add_argument("--run_distribute", type=ast.literal_eval, default=False,
                        help="Run distribute, default is false.")
    parser.add_argument('--batch_size', help='train batch size', default=config.TRAIN_BATCH_SIZE, type=int)
    parser.add_argument("--device_id", type=int, default=1, help="Device id, default is 0.")
    parser.add_argument('--device_target', choices=("Ascend", "GPU", "CPU"), default='Ascend')
    parser.add_argument("--device_num", type=int, default=0, help="device_num, default is 1.")
    parser.add_argument('--prefix_path', help='training data annotation images prefix root path', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    """
    imagedb = ImageDB(args.anno_file)
    gt_imdb = imagedb.load_imdb()
    gt_imdb = imagedb.append_flipped_images(gt_imdb)
    print("len(gt_imdb) : ", len(gt_imdb))
    train_data = data_to_mindrecord_byte_image(gt_imdb, prefix="img48.mindrecord", file_num=1, mindrecord_dir=args.mindrecord_dir)
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target, device_id=args.device_id)
    print("Start train for mtcnn!")

    rank = 0
    device_num = 1

    mindrecord_file = "/root/xidian_wks/mtcnn/data_set/img48.mindrecord"

    print("Start create dataset!")

    dataset = create_mtcnn_dataset(mindrecord_file, batch_size=128, device_num=device_num,
                                   rank_id=rank)
    step_per_epoch = dataset.get_dataset_size()
    print("--------size-------", step_per_epoch)
    net_with_loss = ONetWithLoss(is_train=True)

    learning_rate = 0.01

    #init weight
    for _, cell in net_with_loss.cells_and_names():
        if isinstance(cell, nn.Conv2d) or isinstance(cell, nn.Dense):
            cell.weight.default_input = weight_init.initializer(weight_init.XavierUniform(gain=1 / math.sqrt(3)),
                                                                cell.weight.shape,
                                                                cell.weight.dtype)
            cell.bias.default_input = weight_init.initializer(weight_init.Constant(0.1), cell.bias.shape, cell.bias.dtype)

    opt = SGD(params=net_with_loss.trainable_params(),
               learning_rate=learning_rate)

    model = Model(net_with_loss, optimizer=opt)

    time_cb = TimeMonitor(data_size=step_per_epoch)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]

    config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_per_epoch,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    ckpt_cb = ModelCheckpoint(prefix='mtcnn_onet',
                              directory=args.model_path,
                              config=config_ck)
    cb += [ckpt_cb]

    epoch_size = 16
    model.train(epoch_size, dataset, callbacks=cb)
    print("Train success!")
    
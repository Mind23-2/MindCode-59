import sys
sys.path.append("../img_preprocess/")

#from utils import IoU
import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as np
import numpy
import mindspore.common.initializer as init
import mindspore.ops as ops
from mindspore.common import dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore import context

class ClsLoss(nn.Cell):
    def __init__(self, cls_factor=1):
        super(ClsLoss, self).__init__()
        self.cls_factor = cls_factor
        self.squeeze = ops.Squeeze()
        self.select = ops.Select()
        self.great_equal = ops.GreaterEqual()
        self.loss_cls = nn.BCELoss()
        self.zeros = Tensor(np.zeros([128]), ms.float32)
        self.six = Tensor(np.full((128), 0.6, dtype=ms.float32) )
        self.five = Tensor(np.full((128), 0.5, dtype=ms.float32) )
        self.reducesum = ops.ReduceSum()
        self.cast = ops.Cast()
        self.div = ops.Div()
        self.eq = ops.Equal()

    def construct(self, gt_label, pred_label):
        gt_label = self.squeeze(gt_label)
        gt_label = self.cast(gt_label, mstype.float32)
        pred_label = self.squeeze(pred_label)
        pred_label = self.cast(pred_label, mstype.float32)

        mask = self.great_equal(gt_label, self.zeros)
        #print("mask: ", mask)
        chose_index = ops.count_nonzero(self.cast(mask, mstype.int32))#which labels can use
        cls_loss = self.loss_cls(pred_label, gt_label)
        cls_loss = self.select(mask, cls_loss, self.zeros)
        cls_loss = self.reducesum(cls_loss)
        cls_loss = (cls_loss / chose_index) * self.cls_factor
        
        #cal accurancy
        valid_gt_cls = self.select(mask, gt_label, self.five)
        #print("valid_gt_cls:　", valid_gt_cls)
        valid_prob_cls = self.select(mask, pred_label, self.zeros)
        #print("valid_prob_cls", valid_prob_cls)
        prob_ones = self.great_equal(valid_prob_cls, self.six)
        prob_ones = self.cast(prob_ones, mstype.float32)
        #print("prob_ones: ", prob_ones)
        right_ones = self.eq(prob_ones,valid_gt_cls)
        #print("right_ones: ", right_ones)
        right_value = self.cast(right_ones, mstype.float32)
        #print("right_value: ", right_value)
        value = self.reducesum(right_value)
        #print("value : ", value)
        accurancy = self.div(value, chose_index)
        print("accurancy: ", accurancy)
        
        return cls_loss


class BoxLoss(nn.Cell):
    def __init__(self, box_factor=1):
        super(BoxLoss, self).__init__()
        self.squeeze = ops.Squeeze()
        self.equal = ops.Equal()
        self.loss_box = nn.MSELoss()
        self.box_factor = box_factor
        self.cast = ops.Cast()

    def construct(self, gt_label, gt_offset, pred_offset):
        #print("gt_offset: ", gt_offset)
        #print("pred_offset: ", pred_offset)
        pred_offset = self.squeeze(pred_offset)
        gt_offset = self.squeeze(gt_offset)
        gt_label = self.squeeze(gt_label)
        unmask = self.equal(gt_label, 0)
        mask = self.equal(unmask, 0)
        chose_index = ops.count_nonzero(self.cast(mask, mstype.int32))
        chose_index = self.squeeze(chose_index)
        valid_gt_offset = gt_offset[chose_index, :]
        #print("bbox -----valid_gt_offset dtype -------", valid_gt_offset.dtype)
        valid_pred_offset = pred_offset[chose_index, :]
        #print("bbox -----valid_pred_offset dtype -------", valid_pred_offset.dtype)
        return self.loss_box(valid_pred_offset, valid_gt_offset) * self.box_factor


class LandMarkLoss(nn.Cell):
    def __init__(self, landmark_factor = 1):
        super(LandMarkLoss, self).__init__()
        self.squeeze = ops.Squeeze()
        self.equal = ops.Equal()
        self.cast = ops.Cast()
        self.land_factor = landmark_factor
        self.loss_landmark = nn.MSELoss()
        self.zeros = Tensor(np.zeros([128, 10]), ms.float32)
        self.select = ops.Select()

    def construct(self, gt_label, gt_landmark, pred_landmark):
        pred_landmark = self.squeeze(pred_landmark)
        gt_landmark = self.squeeze(gt_landmark)
        gt_label = self.squeeze(gt_label)
        mask = self.equal(gt_label, -2)
        print("mask: ", mask.shape)
        print("mask: ", mask)
        print("gt_landmark: ", gt_landmark.shape)

        #chose_index = ops.count_nonzero(self.cast(mask, mstype.int32), keep_dims=True)
        #chose_index = self.squeeze(chose_index)
        
        #print("chose_index: ", chose_index)
        valid_gt_landmark = self.select(mask, gt_landmark, self.zeros)
        valid_pred_landmark = self.select(mask, pred_landmark, self.zeros)
        #valid_gt_landmark = gt_landmark[chose_index, :]
        #valid_pred_landmark = pred_landmark[chose_index, :]
        #print("valid_gt_landmark: ", valid_gt_landmark)
        #print("valid_gt_landmark shape: ", valid_gt_landmark.shape)
        print("valid_pred_landmark: ", valid_pred_landmark)
        print("valid_pred_landmark shape: ", valid_pred_landmark.shape)
        return self.loss_landmark(valid_pred_landmark, valid_gt_landmark) * self.land_factor


class PNetWithLoss(nn.Cell):
    def __init__(self, is_train=True):
        super(PNetWithLoss, self).__init__()
        self.net = PNet(is_train=is_train)
        self.cls_loss = ClsLoss()
        self.box_loss = BoxLoss()
        self.add = ops.Add()

    def construct(self, x, cls_label, bbox_target, gt_landmark):
        label_pre, box_pre = self.net(x)
        cls_offset_loss = self.cls_loss(cls_label, label_pre)
        #print("cls_offset_loss shape: ", cls_offset_loss.shape)
        #print("cls_offset_loss: ", cls_offset_loss)
        box_offset_loss = self.box_loss(cls_label, bbox_target, box_pre)
        #print("box_offset_loss shape: ", box_offset_loss.shape)
        #print("box_offset_loss: ", box_offset_loss)
        all_loss = self.add(cls_offset_loss * 1.0,  box_offset_loss * 0.5)
        #print("all_loss shape", all_loss.shape)
        #print("all_loss ", all_loss)
        return all_loss




class PNet(nn.Cell):
    ''' PNet '''

    def __init__(self, is_train=True):
        super(PNet, self).__init__()
        # backend
        
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1, has_bias=True, pad_mode="pad")  # conv1
        self.prelu1 = nn.PReLU()  # PReLU1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")  # pool1
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3, stride=1, has_bias=True, pad_mode="pad")  # conv2
        self.prelu2 = nn.PReLU()  # PReLU2
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, has_bias=True, pad_mode="pad")  # conv3
        self.prelu3 = nn.PReLU()  # PReLU3
        """
        self.pre_layer = nn.SequentialCell(
            [
                nn.Conv2d(3, 10, kernel_size=3, stride=1, has_bias=True, pad_mode="pad", weight_init="XavierUniform"),  # conv1
                nn.PReLU(),  # PReLU1
                nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same"),  # pool1
                nn.Conv2d(10, 16, kernel_size=3, stride=1, has_bias=True, pad_mode="pad", weight_init="XavierUniform"),  # conv2
                nn.PReLU(),  # PReLU2
                nn.Conv2d(16, 32, kernel_size=3, stride=1, has_bias=True, pad_mode="pad", weight_init="XavierUniform"),  # conv3
                nn.PReLU()  # PReLU3
            ]
        )
        """
        # detection
        self.conv4_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1, has_bias=True, pad_mode="pad")
        # bounding box regresion
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1, has_bias=True, pad_mode="pad")
        # landmark localization
        self.conv4_3 = nn.Conv2d(32, 10, kernel_size=1, stride=1, has_bias=True, pad_mode="pad")
        self.sigmoid = nn.Sigmoid()
        self.cast = ops.Cast()
        self.type_dst = ms.float32
        self.is_train = is_train
        self.transpose = ops.Transpose()
        self.expand_dims = ops.ExpandDims()
        self.squeeze = ops.Squeeze(0)

    def construct(self, x):
        if self.is_train:
            x = self.cast(x, self.type_dst)
            x = self.conv1(x)
            x = x[:, :, :, :]
            x = self.prelu1(x)
            x = x[:, :, :, :]
            x = self.pool1(x)
            x = self.conv2(x)
            x = x[:, :, :, :]
            x = self.prelu2(x)
            x = x[:, :, :, :]
            x = self.conv3(x)
            x = x[:, :, :, :]
            x = self.prelu3(x)
            x = x[:, :, :, :]
            #print("------input shape-----", x.shape)
            #x = self.pre_layer(x)
            #print("---prelayer_shape----", x.shape)
            y = self.conv4_1(x)
            #print("---conv4_1_shape----", y.shape)
            label = self.sigmoid(y)
            #print("---label_shape----", label.shape)
            offset = self.conv4_2(x)
            #print("---offset_shape----", offset.shape)
            return label, offset
        else:
            #print("input shape: ", x.shape)
            x = self.expand_dims(x, 0)
            # print("expand dims: ", x.shape)
            x = self.transpose(x, (0, 1, 3, 2))
            # print("transpose dims: ", x.shape)
            x = self.transpose(x, (0, 2, 1, 3))
            x = self.cast(x, self.type_dst)
            # print("------train shape-----", x.shape)
            #x = self.pre_layer(x)
            x = self.conv1(x)
            x = self.prelu1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.prelu2(x)
            x = self.conv3(x)
            x = self.prelu3(x)
            # print("---prelayer_shape----", x.shape)
            y = self.conv4_1(x)
            # print("---conv4_1_shape----", y.shape)
            label = self.sigmoid(y)
            label = self.transpose(label, (0, 2, 1, 3))
            label = self.transpose(label, (0, 1, 3, 2))
            # label = self.squeeze(label)
            # print("---label_shape----", label.shape)
            offset = self.conv4_2(x)
            offset = self.transpose(offset, (0, 2, 1, 3))
            offset = self.transpose(offset, (0, 1, 3, 2))
            # offset = self.squeeze(offset)
            # print("---offset_shape----", offset.shape)
            return label, offset

class RNetWithLoss(nn.Cell):
    def __init__(self, is_train=True):
        super(RNetWithLoss, self).__init__()
        self.net = RNet(is_train=is_train)
        self.cls_loss = ClsLoss()
        self.box_loss = BoxLoss()
        self.add = ops.Add()

    def construct(self, x, cls_label, bbox_target, gt_landmark):
        label_pre, box_pre = self.net(x)
        cls_offset_loss = self.cls_loss(cls_label, label_pre)
        #print("cls_offset_loss: ", cls_offset_loss.shape)
        box_offset_loss = self.box_loss(cls_label, bbox_target, box_pre)
        #print("box_offset_loss shape: ", box_offset_loss.shape)
        all_loss = self.add(cls_offset_loss * 1.0, box_offset_loss * 0.5)
        return all_loss

class RNet(nn.Cell):
    ''' RNet '''

    def __init__(self, is_train=True):
        super(RNet, self).__init__()
        # backend
        
        self.conv1 = nn.Conv2d(3, 28, kernel_size=3, stride=1, has_bias=True, pad_mode="pad", weight_init="XavierUniform")   # conv1
        self.prelu1 = nn.PReLU()  # prelu1
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)  # pool1
        self.conv2 = nn.Conv2d(28, 48, kernel_size=3, stride=1, has_bias=True, pad_mode="pad", weight_init="XavierUniform")  # conv2
        self.prelu2 = nn.PReLU()  # prelu2
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)  # pool2
        self.conv3 = nn.Conv2d(48, 64, kernel_size=2, stride=1, has_bias=True, pad_mode="pad", weight_init="XavierUniform")   # conv3
        self.prelu3 = nn.PReLU()  # prelu3
        """
        self.pre_layer = nn.SequentialCell(
            nn.Conv2d(3, 28, kernel_size=3, stride=1, has_bias=True, pad_mode="pad", weight_init="XavierUniform"),   # conv1
            nn.PReLU(),  # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(28, 48, kernel_size=3, stride=1, has_bias=True, pad_mode="pad", weight_init="XavierUniform"), # conv2
            nn.PReLU(),  # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(48, 64, kernel_size=2, stride=1, has_bias=True, pad_mode="pad", weight_init="XavierUniform"),   # conv3
            nn.PReLU()  # prelu3

        )
        """
        self.conv4 = nn.Dense(64 * 2 * 2, 128)  # conv4
        self.prelu4 = nn.PReLU()  # prelu4
        # detection
        self.conv5_1 = nn.Dense(128, 1)
        # bounding box regression
        self.conv5_2 = nn.Dense(128, 4)
        # lanbmark localization
        self.conv5_3 = nn.Dense(128, 10)
        self.reshape = ops.Reshape()
        self.sigmoid = nn.Sigmoid()
        self.is_train = is_train
        self.cast = ops.Cast()
        self.type_dst = ms.float32


    def construct(self, x):
        # backend
        if self.is_train:
            x = self.cast(x, self.type_dst)
            x = self.conv1(x)
            x = x[:, :, :, :]
            x = self.prelu1(x)
           # x = x[:, :, :, :]
            x = self.pool1(x)
            x = self.conv2(x)
            x = x[:, :, :, :]
            x = self.prelu2(x)
           # x = x[:, :, :, :]
            x = self.pool2(x)
            x = self.conv3(x)
            x = x[:, :, :, :]
            x = self.prelu3(x)
          #  x = x[:, :, :, :]
            #x = self.pre_layer(x)
            x = self.reshape(x, (x.shape[0], -1))
            x = self.conv4(x)
            x = self.prelu4(x)
            # detection
            det = self.sigmoid(self.conv5_1(x))
            box = self.conv5_2(x)

            return det, box
        else:
            x = self.conv1(x)
            x = self.prelu1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.prelu2(x)
            x = self.pool2(x)
            x = self.conv3(x)
            x = self.prelu3(x)
            # x = self.pre_layer(x)
            x = self.reshape(x, (x.shape[0], -1))
            x = self.conv4(x)
            x = self.prelu4(x)
            # detection
            det = self.sigmoid(self.conv5_1(x))
            box = self.conv5_2(x)

            return det, box


class ONetWithLoss(nn.Cell):
    def __init__(self, is_train=True):
        super(ONetWithLoss, self).__init__()
        self.net = ONet(is_train=is_train)
        self.cls_loss = ClsLoss()
        self.box_loss = BoxLoss()
        self.landmark_loss = LandMarkLoss()

    def construct(self, x, cls_label, bbox_target, gt_landmark):
        label_pre, box_pre, landmark_pre = self.net(x)
        cls_offset_loss = self.cls_loss(cls_label, label_pre)
        #print("cls_offset_loss: ", cls_offset_loss.shape)
        box_offset_loss = self.box_loss(cls_label, bbox_target, box_pre)
        #print("box_offset_loss shape: ", box_offset_loss.shape)
        landmark_loss = self.landmark_loss(cls_label, gt_landmark, landmark_pre)
        all_loss = cls_offset_loss*0.8+box_offset_loss*0.6+landmark_loss*1.5
        return all_loss


class ONet(nn.Cell):
    ''' ONet '''

    def __init__(self, is_train=True):
        super(ONet, self).__init__()
        # backend
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, has_bias=True, pad_mode="pad", weight_init="XavierUniform")  # conv1
        self.prelu1 = nn.PReLU()  # prelu1
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)  # pool1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, has_bias=True, pad_mode="pad", weight_init="XavierUniform")  # conv2
        self.prelu2 = nn.PReLU()  # prelu2
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)  # pool2
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, has_bias=True, pad_mode="pad", weight_init="XavierUniform")  # conv3
        self.prelu3 = nn.PReLU()  # prelu3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # pool3
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2, stride=1, has_bias=True, pad_mode="pad", weight_init="XavierUniform")  # conv4
        self.prelu4 = nn.PReLU()  # prelu4
        """
        self.pre_layer = nn.SequentialCell(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, has_bias=True, pad_mode="pad", weight_init="XavierUniform"),  # conv1
            nn.PReLU(),  # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(32, 64, kernel_size=3, stride=1, has_bias=True, pad_mode="pad", weight_init="XavierUniform"),  # conv2
            nn.PReLU(),  # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(64, 64, kernel_size=3, stride=1, has_bias=True, pad_mode="pad", weight_init="XavierUniform"),  # conv3
            nn.PReLU(),  # prelu3
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool3
            nn.Conv2d(64, 128, kernel_size=2, stride=1, has_bias=True, pad_mode="pad", weight_init="XavierUniform"),  # conv4
            nn.PReLU()  # prelu4
        )
        """
        self.conv5 = nn.Dense(128 * 2 * 2, 256)  # conv5
        self.prelu5 = nn.PReLU()  # prelu5
        # detection
        self.conv6_1 = nn.Dense(256, 1)
        # bounding box regression
        self.conv6_2 = nn.Dense(256, 4)
        # lanbmark localization
        self.conv6_3 = nn.Dense(256, 10)
        self.reshape = ops.Reshape()
        self.sigmoid = nn.Sigmoid()
        self.is_train = is_train
        self.cast = ops.Cast()
        self.type_dst = ms.float32

    def construct(self, x):
        # backend
        if self.is_train is True:
            x = self.cast(x, self.type_dst)
            x = self.conv1(x)
            x = x[:, :, :, :]
            x = self.prelu1(x)
            x = x[:, :, :, :]
            x = self.pool1(x)
            x = self.conv2(x)
            x = x[:, :, :, :]
            x = self.prelu2(x)
            x = x[:, :, :, :]
            x = self.pool2(x)
            x = self.conv3(x)
            x = x[:, :, :, :]
            x = self.prelu3(x)
            x = x[:, :, :, :]
            x = self.pool3(x)
            x = self.conv4(x)
            x = x[:, :, :, :]
            x = self.prelu4(x)
            x = x[:, :, :, :]
            #x = self.pre_layer(x)
            x = self.reshape(x, (x.shape[0], -1))
            x = self.conv5(x)
            x = x[:, :]
            x = self.prelu5(x)
            x = x[:, :]
            # detection
            y = self.conv6_1(x)
            det = self.sigmoid(y)
            box = self.conv6_2(x)
            landmark = self.conv6_3(x)
            return det, box, landmark
        else:
            x = self.conv1(x)
            x = self.prelu1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.prelu2(x)
            x = self.pool2(x)
            x = self.conv3(x)
            x = self.prelu3(x)
            x = self.pool3(x)
            x = self.conv4(x)
            x = self.prelu4(x)
            x = self.reshape(x, (x.shape[0], -1))
            x = self.conv5(x)
            x = self.prelu5(x)
            # detection
            y = self.conv6_1(x)
            det = self.sigmoid(y)
            box = self.conv6_2(x)
            landmark = self.conv6_3(x)
            # landmard = self.conv5_3(x)
            return det, box, landmark



if __name__ == '__main__':
    inputs = Tensor(np.ones([64, 3, 48, 48]), ms.float32)
    x1 = Tensor(np.ones([64]), ms.int32)
    x2 = Tensor(np.ones([64, 4]), ms.int32)
    print("inpue_shape", inputs.shape)
    net = ONet()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", save_graphs=False,
                        device_id=3)
    net.set_train()
    out = net(inputs)
    print("______________", out)



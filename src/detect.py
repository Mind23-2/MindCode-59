import mmcv
import sys
import time
import numpy as np
import os
from mindspore import load_checkpoint, load_param_into_net
from mindspore import ops
from mindspore.common.tensor import Tensor
import mindspore as ms
import cv2

from network.mtcnn import PNet, RNet, ONet
import src.image_tools as image_tools
from src.data_utils import nms as nms


def create_mtcnn_net(p_model_path=None, r_model_path=None, o_model_path=None):

    pnet, rnet, onet = None, None, None

    if p_model_path is not None:
        pnet = PNet(is_train=False)
        param_dict = load_checkpoint(p_model_path, filter_prefix="pnet")
        load_param_into_net(pnet, param_dict)
        pnet.set_train(False)

    if r_model_path is not None:
        rnet = RNet()
        param_dict = load_checkpoint(r_model_path, filter_prefix="rnet")
        param_not_load = load_param_into_net(rnet, param_dict)
        rnet.set_train(False)

    if o_model_path is not None:
        onet = ONet()
        param_dict = load_checkpoint(o_model_path, filter_prefix="onet")
        param_not_load = load_param_into_net(onet, param_dict)
        onet.set_train(False)

    return pnet,rnet,onet




class MtcnnDetector(object):
    """
        P,R,O net face detection and landmarks align
    """
    def  __init__(self,
                 pnet = None,
                 rnet = None,
                 onet = None,
                 min_face_size=12,
                 stride=2,
                 threshold=[0.6, 0.7, 0.7],
                 scale_factor=0.709,
                 ):

        self.pnet_detector = pnet
        self.rnet_detector = rnet
        self.onet_detector = onet
        self.min_face_size = min_face_size
        self.stride=stride
        self.thresh = threshold
        self.scale_factor = scale_factor


    def unique_image_format(self,im):
        if not isinstance(im,np.ndarray):
            if im.mode == 'I':
                im = np.array(im, np.int32, copy=False)
            elif im.mode == 'I;16':
                im = np.array(im, np.int16, copy=False)
            else:
                im = np.asarray(im)
        return im

    def square_bbox(self, bbox):
        """
            convert bbox to square
        Parameters:
        ----------
            bbox: numpy array , shape n x m
                input bbox
        Returns:
        -------
            a square bbox
        """
        square_bbox = bbox.copy()

        # x2 - x1
        # y2 - y1
        h = bbox[:, 2] - bbox[:, 0]
        w = bbox[:, 3] - bbox[:, 1]
        l = np.maximum(h,w)
        # x1 = x1 + w*0.5 - l*0.5
        # y1 = y1 + h*0.5 - l*0.5
        square_bbox[:, 0] = bbox[:, 0] + w*0.5 - l*0.5
        square_bbox[:, 1] = bbox[:, 1] + h*0.5 - l*0.5

        # x2 = x1 + l - 1
        # y2 = y1 + l - 1
        square_bbox[:, 2] = square_bbox[:, 0] + l - 1
        square_bbox[:, 3] = square_bbox[:, 1] + l - 1
        return square_bbox


    def generate_bounding_box(self, map, reg, scale, threshold):
        """
            generate bbox from feature map
        Parameters:
        ----------
            map: numpy array , n x m x 1
                detect score for each position
            reg: numpy array , n x m x 4
                bbox
            scale: float number
                scale of this detection
            threshold: float number
                detect threshold
        Returns:
        -------
            bbox array
        """
        stride = 2
        cellsize = 12 # receptive field

        t_index = np.where(map > threshold)
        #print('shape of t_index:{0}'.format(len(t_index)))
        # print('t_index{0}'.format(t_index))
        # time.sleep(5)

        # find nothing
        if t_index[0].size == 0:
            #print("find nothing")
            return np.array([])

        # reg = (1, n, m, 4)
        # choose bounding box whose socre are larger than threshold
        dx1, dy1, dx2, dy2 = [reg[0, t_index[0], t_index[1], i] for i in range(4)]
        # print(dx1.shape)
        # time.sleep(5)
        reg = np.array([dx1, dy1, dx2, dy2])
        # print('shape of reg{0}'.format(reg.shape))

        # lefteye_dx, lefteye_dy, righteye_dx, righteye_dy, nose_dx, nose_dy, \
        # leftmouth_dx, leftmouth_dy, rightmouth_dx, rightmouth_dy = [landmarks[0, t_index[0], t_index[1], i] for i in range(10)]
        #
        # landmarks = np.array([lefteye_dx, lefteye_dy, righteye_dx, righteye_dy, nose_dx, nose_dy, leftmouth_dx, leftmouth_dy, rightmouth_dx, rightmouth_dy])

        # abtain score of classification which larger than threshold
        # t_index[0]: choose the first column of t_index
        # t_index[1]: choose the second column of t_index
        score = map[t_index[0], t_index[1], 0]

        # hence t_index[1] means column, t_index[1] is the value of x
        # hence t_index[0] means row, t_index[0] is the value of y
        boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),            # x1 of prediction box in original image
                                 np.round((stride * t_index[0]) / scale),            # y1 of prediction box in original image
                                 np.round((stride * t_index[1] + cellsize) / scale), # x2 of prediction box in original image
                                 np.round((stride * t_index[0] + cellsize) / scale), # y2 of prediction box in original image
                                                                                     # reconstruct the box in original image
                                 score,
                                 reg,
                                 # landmarks
                                 ])

        return boundingbox.T


    def resize_image(self, img, scale):
        """
            resize image and transform dimention to [batchsize, channel, height, width]
        Parameters:
        ----------
            img: numpy array , height x width x channel
                input image, channels in BGR order here
            scale: float number
                scale factor of resize operation
        Returns:
        -------
            transformed image tensor , 1 x channel x height x width
        """
        height, width, channels = img.shape
        new_height = int(height * scale)     # resized new height
        new_width = int(width * scale)       # resized new width
        new_dim = (new_width, new_height)
        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)      # resized image
        return img_resized

    def pad1(self,bboxes, width, height):
        """Crop boxes that are too big and get coordinates
        with respect to cutouts.
    
        Arguments:
            bboxes: a float numpy array of shape [n, 5],
                where each row is (xmin, ymin, xmax, ymax, score).
            width: a float number.
            height: a float number.
    
        Returns:
            dy, dx, edy, edx: a int numpy arrays of shape [n],
                coordinates of the boxes with respect to the cutouts.
            y, x, ey, ex: a int numpy arrays of shape [n],
                corrected ymin, xmin, ymax, xmax.
            h, w: a int numpy arrays of shape [n],
                just heights and widths of boxes.
    
            in the following order:
                [dy, edy, dx, edx, y, ey, x, ex, w, h].
        """
    
        x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
        w, h = x2 - x1 + 1.0,  y2 - y1 + 1.0
        num_boxes = bboxes.shape[0]
    
        # 'e' stands for end
        # (x, y) -> (ex, ey)
        x, y, ex, ey = x1, y1, x2, y2
    
        # we need to cut out a box from the image.
        # (x, y, ex, ey) are corrected coordinates of the box
        # in the image.
        # (dx, dy, edx, edy) are coordinates of the box in the cutout
        # from the image.
        dx, dy = np.zeros((num_boxes,)), np.zeros((num_boxes,))
        edx, edy = w.copy() - 1.0, h.copy() - 1.0
    
        # if box's bottom right corner is too far right
        ind = np.where(ex > width - 1.0)[0]
        edx[ind] = w[ind] + width - 2.0 - ex[ind]
        ex[ind] = width - 1.0
    
        # if box's bottom right corner is too low
        ind = np.where(ey > height - 1.0)[0]
        edy[ind] = h[ind] + height - 2.0 - ey[ind]
        ey[ind] = height - 1.0
    
        # if box's top left corner is too far left
        ind = np.where(x < 0.0)[0]
        dx[ind] = 0.0 - x[ind]
        x[ind] = 0.0
    
        # if box's top left corner is too high
        ind = np.where(y < 0.0)[0]
        dy[ind] = 0.0 - y[ind]
        y[ind] = 0.0
    
        return_list = [dy, edy, dx, edx, y, ey, x, ex, w, h]
        return_list = [i.astype('int32') for i in return_list]
    
        return return_list
    def pad(self, bboxes, w, h):
        """
            pad the the boxes
        Parameters:
        ----------
            bboxes: numpy array, n x 5
                input bboxes
            w: float number
                width of the input image
            h: float number
                height of the input image
        Returns :
        ------
            dy, dx : numpy array, n x 1
                start point of the bbox in target image
            edy, edx : numpy array, n x 1
                end point of the bbox in target image
            y, x : numpy array, n x 1
                start point of the bbox in original image
            ex, ex : numpy array, n x 1
                end point of the bbox in original image
            tmph, tmpw: numpy array, n x 1
                height and width of the bbox
        """
        # width and height
        tmpw = (bboxes[:, 2] - bboxes[:, 0] + 1).astype(np.int32)
        tmph = (bboxes[:, 3] - bboxes[:, 1] + 1).astype(np.int32)
        numbox = bboxes.shape[0]

        dx = np.zeros((numbox, ))
        dy = np.zeros((numbox, ))
        edx, edy  = tmpw.copy()-1, tmph.copy()-1
        # x, y: start point of the bbox in original image
        # ex, ey: end point of the bbox in original image
        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        tmp_index = np.where(ex > w-1)
        edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
        ex[tmp_index] = w - 1

        tmp_index = np.where(ey > h-1)
        edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
        ey[tmp_index] = h - 1

        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list


    def detect_pnet(self, item):
        """Get face candidates through pnet

        Parameters:
        ----------
        item: dict
            [1, h, w, c]

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_align: numpy array
            boxes after calibration
        """

        # im = self.unique_image_format(im)

        # original wider face data
        im = item['image']
        image = im[0]
        h, w, c = image.shape
        #image = mmcv.impad(image, shape=(1024, 1024), pad_val=0)
        print("h: ", h, "w: ", w, "c: ", c)

        net_size = 12

        current_scale = float(net_size) / self.min_face_size    # find initial scale
        #print('imgshape:{0}, current_scale:{1}'.format(image.shape, current_scale))
        #im_resized = self.resize_image(image, current_scale) # scale = 1.0
        current_height, current_width = h, w
        #print("im_resized.shape: ", im_resized.shape)
        # fcn
        all_boxes = list()
        i = 0
        while min(current_height, current_width) > net_size:
            #print("-------i: ", i)
            #feed_imgs = []
            #feed_imgs.append(im_resized)
            #print("feed_imgs len�?", len(feed_imgs))
            #im_resized = Tensor(feed_imgs, ms.float32)
            im_resized = Tensor(image, ms.float32)
            print("im_resized.shape: ", im_resized.shape)
            #im_resized = ops.transpose(im_resized, (0, 1, 3, 2))
            #im_resized = ops.transpose(im_resized, (0, 2, 1, 3))
            
            cls_map, reg = self.pnet_detector(im_resized)
           
            #cls_map_np = ops.transpose(cls_map, (0,2,1,3))
            #cls_map_np = ops.transpose(cls_map, (0,3,2,1))
            #reg_np = ops.transpose(reg, (0,2,1,3))
            #reg_np = ops.transpose(reg, (0,3,2,1))
            #print("cls_map shape", cls_map.shape)
            #print("reg shape", reg.shape)
            cls_map_np = cls_map.asnumpy()
            reg_np = reg.asnumpy()
            
            #print("cls_map_np.shape", cls_map_np.shape)
            #print("cls_map_np.type", type(cls_map_np))
            #print("reg_np.shape", reg_np.shape)
            #print("reg_np.shape", type(reg_np))
            boxes = self.generate_bounding_box(cls_map_np[ 0, :, :], reg_np, current_scale, self.thresh[0])
            #print("boxes: ", boxes)
            # generate pyramid images
            current_scale *= self.scale_factor # self.scale_factor = 0.709
            #current_height = current_height * current_scale
            #current_width = current_width * current_scale
            #print("current_scale: ", current_scale)
            image = self.resize_image(image, current_scale)
            current_height, current_width, _ = image.shape
            #print("resieze image shape: ", image.shape)

            if boxes.size == 0:
                continue

            # non-maximum suppresion
            keep = nms(boxes[:, :5], 0.5, 'Union')
            #print("keep: ", keep)
            boxes = boxes[keep]
            #print("after nms boxes ", boxes)
            all_boxes.append(boxes)
            i+=1
            #print("i : ", i)

        if len(all_boxes) == 0:
            return None, None
        #print("i: ", i)
        all_boxes = np.vstack(all_boxes)
        #print("shape of all boxes {0}".format(all_boxes.shape))
        # time.sleep(5)

        # merge the detection from first stage
        keep = nms(all_boxes[:, 0:5], 0.7, 'Union')
        all_boxes = all_boxes[keep]
        # boxes = all_boxes[:, :5]

        # x2 - x1
        # y2 - y1
        bw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bh = all_boxes[:, 3] - all_boxes[:, 1] + 1

        # landmark_keep = all_boxes[:, 9:].reshape((5,2))


        boxes = np.vstack([all_boxes[:,0],
                   all_boxes[:,1],
                   all_boxes[:,2],
                   all_boxes[:,3],
                   all_boxes[:,4],
                   # all_boxes[:, 0] + all_boxes[:, 9] * bw,
                   # all_boxes[:, 1] + all_boxes[:,10] * bh,
                   # all_boxes[:, 0] + all_boxes[:, 11] * bw,
                   # all_boxes[:, 1] + all_boxes[:, 12] * bh,
                   # all_boxes[:, 0] + all_boxes[:, 13] * bw,
                   # all_boxes[:, 1] + all_boxes[:, 14] * bh,
                   # all_boxes[:, 0] + all_boxes[:, 15] * bw,
                   # all_boxes[:, 1] + all_boxes[:, 16] * bh,
                   # all_boxes[:, 0] + all_boxes[:, 17] * bw,
                   # all_boxes[:, 1] + all_boxes[:, 18] * bh
                  ])

        boxes = boxes.T

        # boxes = boxes = [x1, y1, x2, y2, score, reg] reg= [px1, py1, px2, py2] (in prediction)
        align_topx = all_boxes[:, 0] + all_boxes[:, 5] * bw
        align_topy = all_boxes[:, 1] + all_boxes[:, 6] * bh
        align_bottomx = all_boxes[:, 2] + all_boxes[:, 7] * bw
        align_bottomy = all_boxes[:, 3] + all_boxes[:, 8] * bh

        # refine the boxes
        boxes_align = np.vstack([ align_topx,
                              align_topy,
                              align_bottomx,
                              align_bottomy,
                              all_boxes[:, 4],
                              # align_topx + all_boxes[:,9] * bw,
                              # align_topy + all_boxes[:,10] * bh,
                              # align_topx + all_boxes[:,11] * bw,
                              # align_topy + all_boxes[:,12] * bh,
                              # align_topx + all_boxes[:,13] * bw,
                              # align_topy + all_boxes[:,14] * bh,
                              # align_topx + all_boxes[:,15] * bw,
                              # align_topy + all_boxes[:,16] * bh,
                              # align_topx + all_boxes[:,17] * bw,
                              # align_topy + all_boxes[:,18] * bh,
                              ])
        boxes_align = boxes_align.T

        return boxes, boxes_align

    def detect_rnet(self, item, dets):
        """Get face candidates using rnet

        Parameters:
        ----------
        item: dict
            [1, h, w, c]
            input image array
        dets: numpy array
            detection results of pnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_align: numpy array
            boxes after calibration
        """
        # im: an input image
        im = item['image']
        image = im[0]
        print("im[0].shape",im[0].shape)
        print("imshape",im.shape)
        h, w, c = image.shape
        im = np.squeeze(im,axis = 0)
        print("im",im.shape)
        if dets is None:
            return None,None

        # (705, 5) = [x1, y1, x2, y2, score, reg]
        # print("pnet detection {0}".format(dets.shape))
        # time.sleep(5)

        # return square boxes
        dets = self.square_bbox(dets)
        # rounds
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad1(dets, w, h)
        num_boxes = dets.shape[0]
        '''
        # helper for setting RNet batch size
        batch_size = self.rnet_detector.batch_size
        ratio = float(num_boxes) / batch_size
        if ratio > 3 or ratio < 0.3:
            print "You may need to reset RNet batch size if this info appears frequently, \
        face candidates:%d, current batch_size:%d"%(num_boxes, batch_size)
        '''

        # cropped_ims_tensors = np.zeros((num_boxes, 3, 24, 24), dtype=np.float32)
        cropped_ims_tensors = []
        for i in range(num_boxes):
            #print("count:",i)
            if tmph[i]<20 or tmpw[i]<20:
                continue
            #print(tmpw[i],tmph[i])
            tmp = np.zeros((tmpw[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i]+1, dx[i]:edx[i]+1, :] = im[y[i]:ey[i]+1, x[i]:ex[i]+1, :]
            crop_im = cv2.resize(tmp, (24, 24))
            crop_im_tensor = Tensor(crop_im, ms.float32)
            # cropped_ims_tensors[i, :, :, :] = crop_im_tensor
            cropped_ims_tensors.append(crop_im_tensor)
        feed_imgs = Tensor(cropped_ims_tensors, ms.float32)

        cls_map, reg = self.rnet_detector(feed_imgs)

        cls_map = cls_map.asnumpy()
        reg = reg.asnumpy()
        # landmark = landmark.cpu().data.numpy()


        keep_inds = np.where(cls_map > self.thresh[1])[0]

        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            cls = cls_map[keep_inds]
            reg = reg[keep_inds]
            # landmark = landmark[keep_inds]
        else:
            return None, None

        keep = utils.nms(boxes, 0.7)

        if len(keep) == 0:
            return None, None

        keep_cls = cls[keep]
        keep_boxes = boxes[keep]
        keep_reg = reg[keep]
        # keep_landmark = landmark[keep]


        bw = keep_boxes[:, 2] - keep_boxes[:, 0] + 1
        bh = keep_boxes[:, 3] - keep_boxes[:, 1] + 1


        boxes = np.vstack([ keep_boxes[:,0],
                              keep_boxes[:,1],
                              keep_boxes[:,2],
                              keep_boxes[:,3],
                              keep_cls[:,0],
                              # keep_boxes[:,0] + keep_landmark[:, 0] * bw,
                              # keep_boxes[:,1] + keep_landmark[:, 1] * bh,
                              # keep_boxes[:,0] + keep_landmark[:, 2] * bw,
                              # keep_boxes[:,1] + keep_landmark[:, 3] * bh,
                              # keep_boxes[:,0] + keep_landmark[:, 4] * bw,
                              # keep_boxes[:,1] + keep_landmark[:, 5] * bh,
                              # keep_boxes[:,0] + keep_landmark[:, 6] * bw,
                              # keep_boxes[:,1] + keep_landmark[:, 7] * bh,
                              # keep_boxes[:,0] + keep_landmark[:, 8] * bw,
                              # keep_boxes[:,1] + keep_landmark[:, 9] * bh,
                            ])

        align_topx = keep_boxes[:,0] + keep_reg[:,0] * bw
        align_topy = keep_boxes[:,1] + keep_reg[:,1] * bh
        align_bottomx = keep_boxes[:,2] + keep_reg[:,2] * bw
        align_bottomy = keep_boxes[:,3] + keep_reg[:,3] * bh

        boxes_align = np.vstack([align_topx,
                               align_topy,
                               align_bottomx,
                               align_bottomy,
                               keep_cls[:, 0],
                               # align_topx + keep_landmark[:, 0] * bw,
                               # align_topy + keep_landmark[:, 1] * bh,
                               # align_topx + keep_landmark[:, 2] * bw,
                               # align_topy + keep_landmark[:, 3] * bh,
                               # align_topx + keep_landmark[:, 4] * bw,
                               # align_topy + keep_landmark[:, 5] * bh,
                               # align_topx + keep_landmark[:, 6] * bw,
                               # align_topy + keep_landmark[:, 7] * bh,
                               # align_topx + keep_landmark[:, 8] * bw,
                               # align_topy + keep_landmark[:, 9] * bh,
                             ])

        boxes = boxes.T
        boxes_align = boxes_align.T

        return boxes, boxes_align

    def detect_onet(self, item, dets):
        """Get face candidates using onet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of rnet

        Returns:
        -------
        boxes_align: numpy array
            boxes after calibration
        landmarks_align: numpy array
            landmarks after calibration

        """
        im = item[0]
        h, w, c = im.shape

        if dets is None:
            return None, None

        dets = self.square_bbox(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]


        # cropped_ims_tensors = np.zeros((num_boxes, 3, 24, 24), dtype=np.float32)
        cropped_ims_tensors = []
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            # crop input image
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            crop_im = cv2.resize(tmp, (48, 48))
            crop_im_tensor = Tensor(crop_im, ms.float32)
            # cropped_ims_tensors[i, :, :, :] = crop_im_tensor
            cropped_ims_tensors.append(crop_im_tensor)
        feed_imgs = Tensor(cropped_ims_tensors, ms.float32)

        cls_map, reg, landmark = self.onet_detector(feed_imgs)

        cls_map = cls_map.asnumpy()
        reg = reg.asnumpy()
        landmark = landmark.asnumpy()

        keep_inds = np.where(cls_map > self.thresh[2])[0]

        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            cls = cls_map[keep_inds]
            reg = reg[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return None, None

        keep = utils.nms(boxes, 0.7, mode="Minimum")

        if len(keep) == 0:
            return None, None

        keep_cls = cls[keep]
        keep_boxes = boxes[keep]
        keep_reg = reg[keep]
        keep_landmark = landmark[keep]

        bw = keep_boxes[:, 2] - keep_boxes[:, 0] + 1
        bh = keep_boxes[:, 3] - keep_boxes[:, 1] + 1


        align_topx = keep_boxes[:, 0] + keep_reg[:, 0] * bw
        align_topy = keep_boxes[:, 1] + keep_reg[:, 1] * bh
        align_bottomx = keep_boxes[:, 2] + keep_reg[:, 2] * bw
        align_bottomy = keep_boxes[:, 3] + keep_reg[:, 3] * bh

        align_landmark_topx = keep_boxes[:, 0]
        align_landmark_topy = keep_boxes[:, 1]




        boxes_align = np.vstack([align_topx,
                                 align_topy,
                                 align_bottomx,
                                 align_bottomy,
                                 keep_cls[:, 0],
                                 # align_topx + keep_landmark[:, 0] * bw,
                                 # align_topy + keep_landmark[:, 1] * bh,
                                 # align_topx + keep_landmark[:, 2] * bw,
                                 # align_topy + keep_landmark[:, 3] * bh,
                                 # align_topx + keep_landmark[:, 4] * bw,
                                 # align_topy + keep_landmark[:, 5] * bh,
                                 # align_topx + keep_landmark[:, 6] * bw,
                                 # align_topy + keep_landmark[:, 7] * bh,
                                 # align_topx + keep_landmark[:, 8] * bw,
                                 # align_topy + keep_landmark[:, 9] * bh,
                                 ])

        boxes_align = boxes_align.T

        landmark =  np.vstack([
                                 align_landmark_topx + keep_landmark[:, 0] * bw,
                                 align_landmark_topy + keep_landmark[:, 1] * bh,
                                 align_landmark_topx + keep_landmark[:, 2] * bw,
                                 align_landmark_topy + keep_landmark[:, 3] * bh,
                                 align_landmark_topx + keep_landmark[:, 4] * bw,
                                 align_landmark_topy + keep_landmark[:, 5] * bh,
                                 align_landmark_topx + keep_landmark[:, 6] * bw,
                                 align_landmark_topy + keep_landmark[:, 7] * bh,
                                 align_landmark_topx + keep_landmark[:, 8] * bw,
                                 align_landmark_topy + keep_landmark[:, 9] * bh,
                                 ])

        landmark_align = landmark.T

        return boxes_align, landmark_align


    def detect_face(self,item):
        """Detect face over image
        """
        boxes_align = np.array([])
        landmark_align =np.array([])

        t = time.time()

        # pnet
        if self.pnet_detector:
            boxes, boxes_align = self.detect_pnet(item)
            if boxes_align is None:
                return np.array([]), np.array([])

            t1 = time.time() - t
            t = time.time()

        # rnet
        if self.rnet_detector:
            boxes, boxes_align = self.detect_rnet(item, boxes_align)
            if boxes_align is None:
                return np.array([]), np.array([])

            t2 = time.time() - t
            t = time.time()

        # onet
        if self.onet_detector:
            boxes_align, landmark_align = self.detect_onet(item, boxes_align)
            if boxes_align is None:
                return np.array([]), np.array([])

            t3 = time.time() - t
            t = time.time()
            print("time cost " + '{:.3f}'.format(t1+t2+t3) + '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1, t2, t3))

        return boxes_align, landmark_align

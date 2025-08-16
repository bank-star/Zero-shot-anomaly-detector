import copy
import fileinput
import glob
import os
import random
import time

import cv2
import numpy as np
import torch.utils.data

from network_ import *
from util_ import *

def val_datasets(batchsize=8, id=0):
    data = []
    labels = []
    for a in range(batchsize):

        num_diff = np.random.randint(low=12, high=20, size=None, dtype='l')  # 生成差异的数量
        mask = np.zeros((H, W))  # 生成固定大小的空白图片（全黑）
        # id = np.random.randint(low=0, high=L)
        src = cv2.imread(imgs[id])
        src = cv2.resize(src, dsize=(W, H))  # 随机选择背景图片resize到640x480
        ex = copy.deepcopy(src)  # 将背景图复制一份用来添加差异

        if rot:
            if np.random.randint(0, 3) == 2:
                angle = np.random.randint(low=-10, high=11) / 4
                mask = rotation(img=mask, angle=angle)
                ex = rotation(img=ex, angle=angle)
            elif np.random.randint(0, 3) == 1:
                angle = np.random.randint(low=-10, high=11) / 4
                src = rotation(img=src, angle=angle)
        if jit:
            if np.random.randint(0, 3) == 2:
                dx, dy = np.random.randint(low=-5, high=6, size=2)
                ex = random_jitter(ex, dx, dy)
                mask = random_jitter(mask, dx, dy)
            elif np.random.randint(0, 3) == 1:
                dx, dy = np.random.randint(low=-5, high=6, size=2)
                src = random_jitter(src, dx, dy)
        if crop:
            src = black_edge_crop(src, H, W)
            ex = black_edge_crop(ex, H, W)
        if hue:
            if np.random.randint(0, 5) == 1:
                ex = random_h(ex)
        if bright:
            if np.random.randint(0, 5) == 1:
                ex = brightness_adjustment(ex)
        if chan_change:
            if np.random.randint(0, 2) == 1:
                sd = np.random.randint(low=0, high=6)
                ex = change_channel(ex, sd=sd)
                src = change_channel(src, sd=sd)
        if flip:
            if np.random.randint(0, 2) == 1:
                sd = np.random.randint(low=0, high=6)
                ex = random_flip(ex, sd)
                src = random_flip(src, sd)

        diff_img = cv2.imread(imgs_diff[np.random.randint(0, Ld)])
        diff_src = cv2.resize(diff_img, dsize=(W, H))
        ex, mask = gen_diffs(ex, mask, diff_src, num_diff, H, W)

        if curve:
            ex, mask = one_curve(ex, mask, H, W)
        if label_smoothing:
            mask[mask == 1] = 0.95
            mask[mask == 0] = 0.05
        if noise:
            if np.random.randint(0, 10) == 1:
                ex = noisy(noise_typ='gauss', image=ex)
            elif np.random.randint(0, 10) == 2:
                ex = noisy(noise_typ='s&p', image=ex)
            elif np.random.randint(0, 10) >= 5:
                ex = add_noise(ex)
        if blur:
            k = np.random.randint(1, 3) * 2 + 1
            if np.random.randint(0, 4) == 1:
                ex = cv2.GaussianBlur(ex, (k, k), 0)
        if light:
            if np.random.randint(0, 3) == 1:
                ex = virtual_light(ex)

        fg = np.random.randint(0, 2)
        right = np.random.randint(10, 15)
        up = np.random.randint(10, 15)
        ex = random_resize(ex, right, up, fg)
        src = random_resize(src, right, up, fg)
        mask = random_resize(mask, right, up, fg)

        if time_noise:
            if np.random.randint(0, 3) == 1:
                ex = add_time_noise(ex)

        if train_vis:
            cv2.imshow('mask', mask)
            cv2.imshow('inp', np.hstack((ex, src)).astype(np.uint8))
            cv2.imshow('diff', (ex - src).astype(np.uint8))
            # cv2.imwrite('./save.png', np.hstack((ex, src, 255*cv2.merge((mask,mask,mask)))).astype(np.uint8))
            cv2.waitKey()

        if input_norm:
            src = rgb_norm(src)
            ex = rgb_norm(ex)

        if input_shuffle:
            if np.random.randint(0, 2) == 0:
                data1 = torch.cuda.FloatTensor(ex).permute(2, 0, 1).unsqueeze(0)
                data2 = torch.cuda.FloatTensor(src).permute(2, 0, 1).unsqueeze(0)
            else:
                data2 = torch.cuda.FloatTensor(ex).permute(2, 0, 1).unsqueeze(0)
                data1 = torch.cuda.FloatTensor(src).permute(2, 0, 1).unsqueeze(0)
        else:
            data1 = torch.cuda.FloatTensor(ex).permute(2, 0, 1).unsqueeze(0)
            data2 = torch.cuda.FloatTensor(src).permute(2, 0, 1).unsqueeze(0)

        mask_label = torch.cuda.FloatTensor(mask).unsqueeze(0).unsqueeze(0)
        data.append(torch.cat((data1, data2), dim=1))
        labels.append(mask_label)

    return torch.cat(data, dim=0), torch.cat(labels, dim=0)


class Deep_PCB(object):
    def __init__(self, path='/home/yxh/DeepPCB-master/PCBData', is_train=False):
        super(Deep_PCB, self).__init__()
        self.path = path
        if is_train:
            with open(path + '/trainval.txt') as test_ids:
                ids_content = test_ids.read()
                test_ids.close()
        else:
            with open(path + '/test.txt') as test_ids:
                ids_content = test_ids.read()
                test_ids.close()
        ids_content = ids_content.split()
        self.test_imgs = []
        self.template_images = []
        self.labels = []
        for each in ids_content:
            if 'txt' not in each:
                self.test_imgs.append(path + '/' + each[:-4] + '_test.jpg')
                self.template_images.append(path + '/' + each[:-4] + '_temp.jpg')
            else:
                self.labels.append(path + '/' + each)

        self.cnt = -1
        self.biaoding_box = [0, 0, 640, 640]
        self.W,self.H = 640, 640
        self.polygon = [0, 0, 640, 0, 640, 640, 0, 640]
        self.length = len(self.labels)
        self.ids = list(range(self.length))

    def test_next(self):
        self.cnt += 1
        data, hard_mask, img1, biaoding_box_zoom = self.get_input(self.test_imgs[self.cnt],
                                                                  self.template_images[self.cnt],
                                                                  self.biaoding_box,
                                                                  self.polygon)
        self.boxes_list = []
        self.img2 = copy.deepcopy(img1)
        with open(self.labels[self.cnt]) as labels:
            labels_content = labels.read()
            labels.close()
        labels_content = labels_content.split('\n')
        img2 = copy.deepcopy(img1)
        for i, each in enumerate(labels_content[:-1]):
            x0, y0, x1, y1 = int(each.split(' ')[0]), int(each.split(' ')[1]), int(each.split(' ')[2]), int(
                each.split(' ')[3])
            xc, yc, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)
            
            wh_ratio = w/h 
            if wh_ratio >= 2:
                x0, y0, x1, y1 = xc - 0.4 * w, yc - 0.25 * h, xc + 0.4 * w, yc + 0.25 * h
            elif wh_ratio <= 0.5:
                x0, y0, x1, y1 = xc - 0.25 * w, yc - 0.4 * h, xc + 0.25 * w, yc + 0.4 * h
            else:
                x0, y0, x1, y1 = xc - 0.25 * w, yc - 0.25 * h, xc + 0.25 * w, yc + 0.25 * h
            
            color = (0, 255, 0)  # 边框颜色红
            thickness = 3  # 边框厚度1
            img2 = np.ascontiguousarray(img2)
            img2 = cv2.rectangle(img2, (int(each.split(' ')[0]), int(each.split(' ')[1])),
                                 (int(each.split(' ')[2]), int(each.split(' ')[3])), color, thickness)

            self.boxes_list.append([int(x0), int(y0), int(x1), int(y1)])
        self.img1 = img2
        return data, hard_mask, img1, biaoding_box_zoom, img2

    def aug(self, src, ex, mask_out):
        # aug
        right = np.random.randint(10, 20)
        up = np.random.randint(10, 15)
        fg = np.random.randint(0, 3)
        ex = random_resize(ex, right, up, fg)
        src = random_resize(src, right, up, fg)
        mask_out = random_resize(mask_out, right, up, fg)

        if rot:
            if np.random.randint(0, 3) == 2:
                angle = np.random.randint(low=-10, high=11) / 2
                ex = rotation(img=ex, angle=angle)
                src = rotation(img=src, angle=angle)
                mask_out = rotation(img=mask_out, angle=angle)
        if jit:
            if np.random.randint(0, 3) == 2:
                dx = np.random.randint(low=-10, high=11)
                dy = np.random.randint(low=-10, high=11)
                ex = random_jitter(ex, dx, dy)
                src = random_jitter(src, dx, dy)
                mask_out = random_jitter(mask_out, dx, dy)
        if flip and (np.random.randint(0, 2) == 1):
            sd = np.random.randint(low=0, high=6)
            ex = random_flip(ex, sd).copy()
            src = random_flip(src, sd).copy()
            mask_out = random_flip(mask_out, sd).copy()
        if chan_change and (np.random.randint(0, 2) == 1):
            sd = np.random.randint(low=0, high=6)
            ex = change_channel(ex, sd=sd)
            src = change_channel(src, sd=sd)

        return src, ex, mask_out

    def train_next(self, batch_size):
        datas = []
        masks = []
        if len(self.ids) >= batch_size:
            sample_ids = random.sample(self.ids, batch_size)
        else:
            sample_ids = random.sample(self.ids, len(self.ids))

        for k in sample_ids:
            self.ids.remove(k)

        if len(self.ids) == 0:
            self.ids = list(range(self.length))
        # print(sample_ids)
        for i in sample_ids:
            img1 = cv2.imread(self.test_imgs[i])
            img2 = cv2.imread(self.template_images[i])
            mask = np.zeros((H, W))
            img1 = cv2.resize(img1, dsize=(W, H), interpolation=cv2.INTER_NEAREST)  # 输入大小resize到640x480
            img2 = cv2.resize(img2, dsize=(W, H), interpolation=cv2.INTER_NEAREST)

            with open(self.labels[i]) as labels:
                labels_content = labels.read()
                labels.close()
            labels_content = labels_content.split('\n')
            for i, each in enumerate(labels_content[:-1]):
                if W != H:
                    x0, y0, x1, y1 = int(each.split(' ')[0]), int(each.split(' ')[1]) * 0.75, int(
                        each.split(' ')[2]), int(each.split(' ')[3]) * 0.75
                else:
                    x0, y0, x1, y1 = int(each.split(' ')[0]), int(each.split(' ')[1]), int(
                        each.split(' ')[2]), int(each.split(' ')[3])
                xc, yc, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)
                x0, y0, x1, y1 = xc - 0.25 * w, yc - 0.25 * h, xc + 0.25 * w, yc + 0.25 * h
                mask[int(y0):int(y1), int(x0):int(x1)] = 1

            if aug:
                img1, img2, mask = self.aug(img1, img2, mask)
            if train_vis:
                img3 = np.zeros_like(img1)
                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        if mask[i, j] == 1:
                            img3[i, j, :] = img1[i, j, :]
                cv2.imshow('1', img1)
                cv2.imshow('2', img2)
                cv2.imshow('m', img3)
                cv2.waitKey()

            img1 = rgb_norm(img1)
            img2 = rgb_norm(img2)

            img1 = torch.cuda.FloatTensor(img1).permute(2, 0, 1).unsqueeze(0)
            img2 = torch.cuda.FloatTensor(img2).permute(2, 0, 1).unsqueeze(0)
            datas.append(torch.cat((img1, img2), dim=1))

            masks.append(torch.cuda.FloatTensor(mask).unsqueeze(0).unsqueeze(0))

        return torch.cat(datas, dim=0), torch.cat(masks, dim=0)

    def get_input(self, defection_img_path, normal_img_path, biaoding_box, polygon):
        img1 = cv2.imread(defection_img_path)
        img2 = cv2.imread(normal_img_path)

        assert img1.shape == img2.shape
        hard_mask = np.zeros_like(img1)
        biaoding_box_zoom = copy.deepcopy(biaoding_box)
        biaoding_box_center_y = (biaoding_box[3] + biaoding_box[1]) / 2.
        biaoding_box_center_x = (biaoding_box[2] + biaoding_box[0]) / 2.
        biaoding_box_width = biaoding_box[2] - biaoding_box[0]
        biaoding_box_height = biaoding_box[3] - biaoding_box[1]
        biaoding_box_scale_y = biaoding_box_height / img1.shape[0]
        biaoding_box_scale_x = biaoding_box_width / img1.shape[1]
        zoom_scale_x = (1.0 / biaoding_box_scale_x) ** 0.5
        zoom_scale_y = (1.0 / biaoding_box_scale_y) ** 0.5

        biaoding_box_zoom[0] = biaoding_box_center_x - biaoding_box_width * (zoom_scale_x - 0.5)
        biaoding_box_zoom[2] = biaoding_box_center_x + biaoding_box_width * (zoom_scale_x - 0.5)
        biaoding_box_zoom[1] = biaoding_box_center_y - biaoding_box_height * (zoom_scale_y - 0.5)
        biaoding_box_zoom[3] = biaoding_box_center_y + biaoding_box_height * (zoom_scale_y - 0.5)

        top_size, bottom_size, left_size, right_size = biaoding_box[1] - biaoding_box_zoom[1], \
                                                       biaoding_box_zoom[3] - biaoding_box[3], \
                                                       biaoding_box[0] - biaoding_box_zoom[0], \
                                                       biaoding_box_zoom[2] - biaoding_box[2],

        for each, i in zip(biaoding_box_zoom, range(len(biaoding_box_zoom))):
            biaoding_box_zoom[i] = int(each)

        roi1 = img1[biaoding_box[1]:biaoding_box[3], biaoding_box[0]:biaoding_box[2], :]
        roi2 = img2[biaoding_box[1]:biaoding_box[3], biaoding_box[0]:biaoding_box[2], :]
        roi1_pad = cv2.copyMakeBorder(roi1, int(top_size), int(bottom_size), int(left_size), int(right_size),
                                      cv2.BORDER_CONSTANT, value=0)
        roi2_pad = cv2.copyMakeBorder(roi2, int(top_size), int(bottom_size), int(left_size), int(right_size),
                                      cv2.BORDER_CONSTANT, value=0)

        roi1 = cv2.resize(roi1_pad, dsize=(self.W, self.H), interpolation=cv2.INTER_NEAREST)  # 输入大小resize到640x480
        roi2 = cv2.resize(roi2_pad, dsize=(self.W, self.H), interpolation=cv2.INTER_NEAREST)

        # roi1 = rgb_norm(roi1)
        # roi2 = rgb_norm(roi2)

        roi1 = torch.cuda.FloatTensor(roi1).permute(2, 0, 1).unsqueeze(0)
        roi2 = torch.cuda.FloatTensor(roi2).permute(2, 0, 1).unsqueeze(0)
        data = torch.cat((roi1, roi2), dim=1)

        data = rgb_norm_torch(data)

        hard_mask = polygon2mask(polygon=polygon, mask=hard_mask)

        return data, hard_mask, img1, biaoding_box_zoom

    def save_pred_boxes(self, result_box):
        res_file = open(
           '/home/yxh/DeepPCB-master/evaluation/res' + self.test_imgs[self.cnt][-18:-9] + '.txt',
            mode='w')
        # print(self.test_imgs[self.cnt][-18:-9])
        for each in result_box:
            iou = [get_iou(each, self.boxes_list[i]) for i in range(len(self.boxes_list))]
            confidence = max(iou)
            # print('result', each)
            # print(iou)
            res_file.write(str(each[0]) + ',')
            res_file.write(str(each[1]) + ',')
            res_file.write(str(each[2]) + ',')
            res_file.write(str(each[3]) + ',')
            res_file.write(str(round(confidence, 2)) + ',')
            # res_file.write(str(confidence) + ',')
            res_file.write('1' + '\n')
            # cv2.rectangle(self.img2, (each[0], each[1]), (each[2], each[3]), (0, 255, 0), 1)
        res_file.close()
        # print(self.labels[self.cnt])
        # cv2.imshow('gt', self.img1)
        # cv2.imshow('pre', self.img2)
        # cv2.waitKey()

    def add_gt_boxes(self, img):
        for box in self.boxes_list:
            color = (0, 255, 0)  # 边框颜色绿
            thickness = 3  # 边框厚度1
            img = np.ascontiguousarray(img)
            img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, thickness)
        return img


    def test_rot_jit_next(self):
        self.cnt += 1
        data, hard_mask, img1, biaoding_box_zoom = self.get_rot_jit_input(self.test_imgs[self.cnt],
                                                                  self.template_images[self.cnt],
                                                                  self.biaoding_box,
                                                                  self.polygon)
        self.boxes_list = []
        self.img2 = copy.deepcopy(img1)
        with open(self.labels[self.cnt]) as labels:
            labels_content = labels.read()
            labels.close()
        labels_content = labels_content.split('\n')
        img2 = copy.deepcopy(img1)
        for i, each in enumerate(labels_content[:-1]):
            x0, y0, x1, y1 = int(each.split(' ')[0]), int(each.split(' ')[1]), int(each.split(' ')[2]), int(
                each.split(' ')[3])
            xc, yc, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)

            wh_ratio = w/h
            if wh_ratio >= 2:
                x0, y0, x1, y1 = xc - 0.4 * w, yc - 0.25 * h, xc + 0.4 * w, yc + 0.25 * h
            elif wh_ratio <= 0.5:
                x0, y0, x1, y1 = xc - 0.25 * w, yc - 0.4 * h, xc + 0.25 * w, yc + 0.4 * h
            else:
                x0, y0, x1, y1 = xc - 0.25 * w, yc - 0.25 * h, xc + 0.25 * w, yc + 0.25 * h
            color = (0, 255, 0)  # 边框颜色红
            thickness = 1  # 边框厚度1
            img2 = np.ascontiguousarray(img2)
            img2 = cv2.rectangle(img2, (int(each.split(' ')[0]), int(each.split(' ')[1])),
                                 (int(each.split(' ')[2]), int(each.split(' ')[3])), color, thickness)

            ##########
            if self.ex_rot==99:
                x0, y0, x1, y1 = get_rotated_rectangle_coords(x0, y0, x1, y1, self.angle)
            if self.ex_jit==99:
                x0, y0, x1, y1 = get_translated_rectangle_coords(x0, y0, x1, y1, self.dx, self.dy)
            ##########

            cv2.rectangle(img2, (int(x0), int(y0)), (int(x1), int(y1)), (0,0,255),1)

            self.boxes_list.append([int(x0), int(y0), int(x1), int(y1)])
        self.img1 = img2
        self.save_gt_boxes(self.boxes_list)
        return data, hard_mask, img1, biaoding_box_zoom, img2

    def get_rot_jit_input(self, defection_img_path, normal_img_path, biaoding_box, polygon):
        img1 = cv2.imread(defection_img_path)
        img2 = cv2.imread(normal_img_path)

        self.angle = np.random.randint(low=101, high=126) / 10. 
        self.ex_rot = np.random.randint(0, 2)
        if self.ex_rot == 99:
            img1 = rotation(img=img1, angle=self.angle)
        else:
            img2 = rotation(img=img2, angle=self.angle)

        self.dx, self.dy = np.random.randint(low=-300, high=301, size=2)/10.
        self.ex_jit = np.random.randint(0, 2)
        #if self.ex_jit == 99:
        #    img1 = random_jitter(img1, self.dx, self.dy)
        #else:
        #    img2 = random_jitter(img2, self.dx, self.dy)


        assert img1.shape == img2.shape
        hard_mask = np.zeros_like(img1)
        biaoding_box_zoom = copy.deepcopy(biaoding_box)
        biaoding_box_center_y = (biaoding_box[3] + biaoding_box[1]) / 2.
        biaoding_box_center_x = (biaoding_box[2] + biaoding_box[0]) / 2.
        biaoding_box_width = biaoding_box[2] - biaoding_box[0]
        biaoding_box_height = biaoding_box[3] - biaoding_box[1]
        biaoding_box_scale_y = biaoding_box_height / img1.shape[0]
        biaoding_box_scale_x = biaoding_box_width / img1.shape[1]
        zoom_scale_x = (1.0 / biaoding_box_scale_x) ** 0.5
        zoom_scale_y = (1.0 / biaoding_box_scale_y) ** 0.5

        biaoding_box_zoom[0] = biaoding_box_center_x - biaoding_box_width * (zoom_scale_x - 0.5)
        biaoding_box_zoom[2] = biaoding_box_center_x + biaoding_box_width * (zoom_scale_x - 0.5)
        biaoding_box_zoom[1] = biaoding_box_center_y - biaoding_box_height * (zoom_scale_y - 0.5)
        biaoding_box_zoom[3] = biaoding_box_center_y + biaoding_box_height * (zoom_scale_y - 0.5)

        top_size, bottom_size, left_size, right_size = biaoding_box[1] - biaoding_box_zoom[1], \
                                                       biaoding_box_zoom[3] - biaoding_box[3], \
                                                       biaoding_box[0] - biaoding_box_zoom[0], \
                                                       biaoding_box_zoom[2] - biaoding_box[2],

        for each, i in zip(biaoding_box_zoom, range(len(biaoding_box_zoom))):
            biaoding_box_zoom[i] = int(each)

        roi1 = img1[biaoding_box[1]:biaoding_box[3], biaoding_box[0]:biaoding_box[2], :]
        roi2 = img2[biaoding_box[1]:biaoding_box[3], biaoding_box[0]:biaoding_box[2], :]
        roi1_pad = cv2.copyMakeBorder(roi1, int(top_size), int(bottom_size), int(left_size), int(right_size),
                                      cv2.BORDER_CONSTANT, value=0)
        roi2_pad = cv2.copyMakeBorder(roi2, int(top_size), int(bottom_size), int(left_size), int(right_size),
                                      cv2.BORDER_CONSTANT, value=0)

        roi1 = cv2.resize(roi1_pad, dsize=(self.W, self.H), interpolation=cv2.INTER_NEAREST)  # 输入大小resize到640x480
        roi2 = cv2.resize(roi2_pad, dsize=(self.W, self.H), interpolation=cv2.INTER_NEAREST)

        try:
            roi1 = torch.FloatTensor(roi1).permute(2, 0, 1).unsqueeze(0).cuda()
            roi2 = torch.FloatTensor(roi2).permute(2, 0, 1).unsqueeze(0).cuda()
        except:
            roi1 = torch.FloatTensor(roi1).permute(2, 0, 1).unsqueeze(0)
            roi2 = torch.FloatTensor(roi2).permute(2, 0, 1).unsqueeze(0)
        data = torch.cat((roi1, roi2), dim=1)

        data = rgb_norm_torch(data)

        hard_mask = polygon2mask(polygon=polygon, mask=hard_mask)

        return data, hard_mask, img1, biaoding_box_zoom

    def save_gt_boxes(self, rot_jit_box):
        gt_file = open(
           '/home/yxh/DeepPCB-master/evaluation/gt_rot_jit' + self.test_imgs[self.cnt][-18:-9] + '.txt',
            mode='w')
        for each in rot_jit_box:
            gt_file.write(str(each[0]) + ',')
            gt_file.write(str(each[1]) + ',')
            gt_file.write(str(each[2]) + ',')
            gt_file.write(str(each[3]) + ',')
            gt_file.write('1' + '\n')
        gt_file.close()


from tqdm import tqdm
# from thop import profile


def demo(ckpt=None):
    mode_type = 'rst_flow'
    rot_jit_flag = False
    show_result = False
    net = Net5(Norm=False, mode_type=mode_type).cuda()
    #ckpt = './ckpt/checkpoint_49000.pth'
    if ckpt is not None:
        print(ckpt)
        checkpoint = torch.load(ckpt)
        net.load_state_dict(checkpoint)
    else:
        ckpt_matched = glob.glob('./ckpt_' + mode_type + '/checkpoint_49000_' + mode_type + '_lk_lr_*.pth')
        checkpoint = torch.load(ckpt_matched[0])
        net = get_state_static(net, checkpoint)
    net.eval()

    pcb = Deep_PCB()
    pre_cost_time = AverageMeter()
    infer_cost_time = AverageMeter()
    post_cost_time = AverageMeter()

    for i in tqdm(range(pcb.length)):

        # 加载和预处理时间
        ###############################################################
        start = time.time()
        if rot_jit_flag:
            data, hard_mask, img1, biaoding_box_zoom, gt = pcb.test_rot_jit_next()
        else:
            data, hard_mask, img1, biaoding_box_zoom, gt = pcb.test_next()

        biaoding_box, polygon = pcb.biaoding_box, pcb.polygon
        end1 = time.time()
        pre_cost_time.update(end1 - start)
        ###############################################################

        # 网络前向传播时间
        ###############################################################
        out = net(data)
        # flops, params = profile(net, (data,))
        # print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))
        end2 = time.time()
        infer_cost_time.update(end2 - end1)
        ###############################################################

        # 结果后处理时间
        ###############################################################
        result = out["mask"].squeeze(0).squeeze(0).cpu().detach().numpy()  # sigmoid
        # src = data[:, 3:6, :, :].squeeze(0).permute(1, 2, 0).cpu().numpy()
        # ex = data[:, 0:3, :, :].squeeze(0).permute(1, 2, 0).cpu().numpy()
        bbox_on_ex, pre_boxes = get_bbox_from_mask_zoom_pcb(result, img1, biaoding_box_zoom, biaoding_box,
                                                            polygon, p=0.3)  # 在框出来的图片中将缺陷标注出来
        end3 = time.time()
        post_cost_time.update(end3 - end2)
        ###############################################################

        pcb.save_pred_boxes(pre_boxes)
        # bbox_on_ex = pcb.add_gt_boxes(bbox_on_ex)
        bbox_on_ex = bbox_on_ex * hard_mask

        # if save_result:
            # cv2.imwrite(result_save_path + 'result.png', result * 255)
            # cv2.imwrite(result_save_path + 'ex.png', ex)
            # cv2.imwrite(result_save_path + 'src.png', src)
            # cv2.imwrite(result_save_path + 'mask_on_ex.png', 0.4 * result * 255 + 0.6 * ex)
            # cv2.imwrite(result_save_path + '/bbox_on_ex_' + input_name + '.png', bbox_on_ex)
            # print('result have been saved to:' + result_save_path + '/bbox_on_ex_' + input_name + '.png')
        # if show_result:
            # cv2.imshow('result', result)
            # cv2.imshow('ex', ex / 255)
            # cv2.imshow('src', src / 255)
            # cv2.imshow('gt', gt / 255)
            # cv2.imshow('mask_on_ex', 0.4 * result + 0.6 * (ex / 255))
            # cv2.namedWindow('bbox_on_ex', cv2.WINDOW_NORMAL)
            # cv2.imshow('bbox_on_ex', bbox_on_ex / 255)
            # cv2.imwrite('./pcb_result/pcb_src' + str(i) + '_.png', src)
            # cv2.imwrite('./pcb_result/pcb_ex' + str(i) + '_.png', ex)
            # cv2.imwrite('./pcb_result_v2/pcb_gt' + str(i) + '_.png', gt)
            # cv2.imwrite('./pcb_result_v2/pcb_pre' + str(i) + '_.png', bbox_on_ex)
            # cv2.imwrite('./pcb_result_v2/pcb_pre_non_finetune' + str(i) + '_.png', bbox_on_ex)
            # cv2.waitKey()
    # print(pre_cost_time.avg * 1000)
    # print(infer_cost_time.avg * 1000)
    # print(post_cost_time.avg * 1000)
    if ckpt is not None:
        return run_commands(rot_jit_flag=rot_jit_flag)
    else:
        print(run_commands(rot_jit_flag=rot_jit_flag))


def demo_connector():
    # input_name = '203'  # 输入图片名,使用之前,带缺陷的图请重命名为名‘原名+_d’,无缺陷的图请重命名为名‘原名+_n’
    polygon, biaoding_box = get_list0(input_name + '_n')

    if len(polygon) == 4:
        x0, y0, x1, y1 = polygon[0], polygon[1], polygon[2], polygon[3]
        polygon = [x0, y0, x1, y0, x1, y1, x0, y1]

    data, hard_mask, imgs, biaoding_box_zoom, polygon, cls, input_mask = get_demo_input_zoom_connector(
        defection_img_path=input_path + '/' + input_name + '_d.jpg',
        normal_img_path=input_path + '/' + input_name + '_n.jpg',
        biaoding_box=biaoding_box, polygon=polygon, H=H, W=W)  # data:框出来的部分从原始比例缩放到(640,480), img_d:带缺陷的原图， 框出来部分的坐标

    # if cls == 'small_circle':
    #     net = Net2(Norm=True)
    #     checkpoint = torch.load('./checkpoint_for_small_circle.pth', map_location=torch.device('cpu'))
    # elif cls == 'normal_circle':
    #     net = Net(Norm=True)
    #     checkpoint = torch.load('./checkpoint_for_circle.pth', map_location=torch.device('cpu'))
    # else:
    #     net = Net(Norm=True)
    #     checkpoint = torch.load('./checkpoint_for_circle.pth', map_location=torch.device('cpu'))

    net = Net3(Norm=False)
    # checkpoint = torch.load('./rstdet_pretrained/pretrained_9538.pth', map_location=torch.device('cpu'))
    checkpoint = torch.load('./pretrained_ScaleLong_SiLU_9700.pth', map_location=torch.device('cpu'))


    net = get_state_static(net, checkpoint)
    # net.load_state_dict(checkpoint)
    net.eval()

    res, out = net(data)

    result = out.squeeze(0).permute(1, 2, 0).detach().numpy()  # sigmoid
    # result = out[1].permute(1, 2, 0).cpu().detach().numpy()  # softmax
    src = data[:, 3:6, :, :].squeeze(0).permute(1, 2, 0).numpy()
    ex = data[:, 0:3, :, :].squeeze(0).permute(1, 2, 0).numpy()
    bbox_on_ex, pre_boxes, show_boxes = get_bbox_from_mask_zoom_connector(result, imgs, biaoding_box_zoom, hard_mask,
                                                                          polygon, cls, input_mask)  # 在框出来的图片中将缺陷标注出来

    bbox_on_ex = bbox_on_ex * hard_mask[0]

    # res = res.squeeze(0).permute(1, 2, 0).detach().numpy()

    if save_result:
        # cv2.imwrite(result_save_path + 'result.png', result * 255)
        # cv2.imwrite(result_save_path + 'ex.png', ex)
        # cv2.imwrite(result_save_path + 'src.png', src)
        # cv2.imwrite(result_save_path + 'mask_on_ex.png', 0.4 * result * 255 + 0.6 * ex)
        # cv2.imwrite(result_save_path + '/bbox_on_ex_' + input_name + '.png', bbox_on_ex)
        print('result have been saved to:' + result_save_path + '/bbox_on_ex_' + input_name + '.png')
    if show_result:
        ex = cv2.normalize(ex, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        src = cv2.normalize(src, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        heatmapshow = gen_heatmap(result)

        for box in show_boxes:
            ex = cv2.rectangle(ex, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
        results_shared = ((np.hstack((src, ex, heatmapshow)))).astype(np.uint8)
        cv2.imshow('ex', results_shared)

        # res = cv2.normalize(res, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        # cv2.imshow('res', (res * 255).astype(np.uint8))

        cv2.imwrite('./results_shared/' + str(1) + '//' + input_name + '.png', results_shared)
        # cv2.imwrite(result_save_path + '/test_on_' + input_name + '.png', np.hstack((src, ex, mask_show)))
        # cv2.namedWindow('bbox_on_ex', cv2.WINDOW_NORMAL)
        # cv2.imshow('bbox_on_ex', bbox_on_ex / 255)

        cv2.waitKey()


def demo_diff():
    import lpips

    # pretrained_lpips = lpips.LPIPS(spatial=True, lpips=False).cuda()

    net = MaskDecoder(Norm=False).cuda()
    checkpoint = torch.load('checkpoint_2000.pth')

    net = get_state_static(net, checkpoint)
    net.eval()

    input_path = r'D:\DL_extend\Processing_anomaly\test_examples\1'
    im_type = 'jpg'

    for input_name in range(1, 47):
        input_name = str(input_name)
        if os.path.exists(input_path + '/' + input_name + '_d.' + im_type):

            if im_type != 'png':
                data, ex, src = get_demo_input_diff(
                    defection_img_path=input_path + '/' + input_name + '_d.' + im_type,
                    normal_img_path=input_path + '/' + input_name + '_n.' + im_type,
                    H=H, W=W)  # data:框出来的部分从原始比例缩放到(640,480), img_d:带缺陷的原图， 框出来部分的坐标
            else:
                data, hard_mask, imgs, biaoding_box_zoom, polygon, cls, input_mask = get_demo_input_zoom_connector(
                    defection_img_path=input_path + '/' + input_name + 'd.png',
                    normal_img_path=input_path + '/' + input_name + 'n.png',
                    biaoding_box=[0, 0, 640, 640], polygon=None, H=H, W=W)

            with torch.no_grad():
                out = net(data)

            result = out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            # rst = rst.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            ex = data[:, :3, :, :].squeeze(0).permute(1, 2, 0).cpu().numpy()
            src = data[:, 3:, :, :].squeeze(0).permute(1, 2, 0).cpu().numpy()
            ex = cv2.normalize(ex, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            src = cv2.normalize(src, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            # rst = cv2.normalize(rst, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

            if im_type != 'png':
                bbox_on_ex = get_bbox_from_mask_diff(result, copy.deepcopy(ex))  # 在框出来的图片中将缺陷标注出来
            else:
                bbox_on_ex, pre_boxes, show_boxes = get_bbox_from_mask_connector(result, imgs, biaoding_box_zoom,
                                                                                 hard_mask,
                                                                                 polygon, cls, input_mask)

            if show_result:
                heatmapshow = None
                heatmapshow = cv2.normalize(result, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                            dtype=cv2.CV_8U)
                heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)

                cv2.namedWindow('input', cv2.WINDOW_NORMAL)
                cv2.imshow('input', np.hstack((ex, src, heatmapshow)).astype(np.uint8))
                # cv2.imshow('input', heatmapshow.astype(np.uint8))
                cv2.waitKey()

                # cv2.imwrite(r'D:\DL_extend\Difference_detecter\Data\test\test_pics\0416\results\\'+input_name+'_r.jpg', np.hstack((ex, src, cv2.addWeighted(heatmapshow, 0.7, ex.astype(np.uint8), 0.3, 0))).astype(np.uint8))


def demo_dataset():
    net = Net3(Norm=False).cuda()
    checkpoint = torch.load('./pretrained_9601.pth')
    net = get_state_static(net, checkpoint)
    net.eval()

    mPA = AverageMeter()
    mIOU = AverageMeter()
    Percision = AverageMeter()
    Recall = AverageMeter()
    Acc = AverageMeter()
    F1 = AverageMeter()

    test_len = L
    with tqdm(total=test_len) as pbar:
        for i in range(test_len):
            data, label = val_datasets(batchsize=1, id=i)
            out = net(data)
            gtboxes = []
            gt_mask = label.detach().cpu().squeeze(0).squeeze(0).numpy()
            # cv2.imshow('a', gt_mask)

            contours, hierarchy = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            gt_file = open('D:\yxh\DeepPCB-master\\evaluation_syn\gt\\' + str(i) + '.txt', mode='w')
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                x0, y0, x1, y1 = x, y, x + w, y + h
                gtboxes.append([x0, y0, x1, y1])
                gt_file.write(str(x0) + ',')
                gt_file.write(str(y0) + ',')
                gt_file.write(str(x1) + ',')
                gt_file.write(str(y1) + ',')
                gt_file.write('1' + '\n')
            gt_file.close()

            result = out[1].squeeze(0).cpu().permute(1, 2, 0).detach().numpy()
            mask = cv2.normalize(result, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            mask = np.uint8(mask * 255)
            # 转换成二值图
            ret, mask = cv2.threshold(mask, 255 * 0.1, 255, cv2.THRESH_BINARY)  # diff
            mPA.update(binary_pa(mask // 255, gt_mask))
            mIOU.update(binary_iou(mask // 255, gt_mask))
            recall, precision, F1_score, acc = binary_evaluation(mask // 255, gt_mask)
            Recall.update(recall)
            Percision.update(precision)
            F1.update(F1_score)
            Acc.update(acc)

            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            res_file = open('D:\yxh\DeepPCB-master\\evaluation_syn\\res\\' + str(i) + '.txt',
                            mode='w')
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w * h > 100:
                    x0, y0, x1, y1 = x, y, x + w, y + h
                    iou = [get_iou([x0, y0, x1, y1], gtboxes[i]) for i in range(len(gtboxes))]
                    confidence = max(iou)
                    res_file.write(str(x0) + ',')
                    res_file.write(str(y0) + ',')
                    res_file.write(str(x1) + ',')
                    res_file.write(str(y1) + ',')
                    res_file.write(str(round(float(confidence), 2)) + ',')
                    res_file.write('1' + '\n')

            gt_file.close()
            # cv2.imshow('b', mask)
            # cv2.waitKey()
            pbar.set_postfix(cur_pa='{:.3}'.format(mPA.val), cur_iou='{:.3}'.format(mIOU.val),
                             mpa='{:.3}'.format(mPA.avg), miou='{:.3}'.format(mIOU.avg),
                             Recall='{:.3}'.format(Recall.avg), Percision='{:.3}'.format(Percision.avg),
                             F1='{:.3}'.format(F1.avg), Acc='{:.3}'.format(Acc.avg))
            pbar.update()
    pbar.close()
    print('mIOU, mPA:', mIOU.avg, mPA.avg)


def demo_freestyle():
    net = Net5(Norm=False).cuda()
    checkpoint = torch.load('./ckpt_rst_flow/checkpoint_49000_rst_flow_lk_lr_972.pth')
    net = get_state_static(net, checkpoint)
    net.eval()
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    template_path = filedialog.askopenfilename(title="Select template path")
    assert template_path != " "
    print("Select template path:", template_path)

    test_path = filedialog.askopenfilename(title="Select test path")
    assert test_path != " "
    print("Select test path:", test_path)

    data, ex, src = get_demo_input_diff(test_path, template_path, H, W)

    out = net(data.cuda())
    result = out["mask"].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    flow = out["flow"][-1].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    src_warp = out["image"].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    ex = data[:, 0:3, :, :].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    src = data[:, 3:6, :, :].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    ex = cv2.normalize(ex, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    src = cv2.normalize(src, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    src_warp = cv2.normalize(src_warp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    bbox_on_ex = get_bbox_from_mask_diff(result, copy.deepcopy(ex))  # 在框出来的图片中将缺陷标注出来
    flow = flow2rgb(flow)
    if show_result:
        cv2.namedWindow('res',0)
        cv2.imshow('res', np.vstack((
             np.hstack((ex, src)),
             np.hstack((flow, 0.7 * gen_heatmap(result) + 0.3 * ex)
                                               ))).astype(np.uint8))

        # cv2.imwrite(r'C:\Users\root\Desktop\temp\\' + str(i) + '_template.jpg', src)
        # cv2.imwrite(r'C:\Users\root\Desktop\temp\\' + str(i) + '_test.jpg', ex)
        # cv2.imwrite(r'C:\Users\root\Desktop\temp\\' + str(i) + '_result.jpg', 0.7 * gen_heatmap(result) + 0.3 * ex)
        # cv2.imwrite(test_path.replace('.jpg','_result.jpg'), np.vstack((src, ex, 0.7 * gen_heatmap(result) + 0.3 * ex)))
        cv2.waitKey()


if __name__ == '__main__':

    # np.random.seed(3047)
    # torch.manual_seed(3407)

    seed_torch(3407)

    H, W = 640, 640
    is_train = False  # 默认为测试

    aug = True

    # 训练参数
    jit = aug  # 是否进行抖动增强
    rot = aug  # 是否进行旋转增强
    chan_change = aug  # 是否进行通道变化增强
    crop = aug  # 是否进行随机裁剪增强
    hue = False  # 是否进行色调随机增强
    bright = False  # 是否进行亮度随机增强
    flip = aug  # 是否进行随机翻转
    light = aug  # 是否模拟光照增强
    noise = aug  # 是否添加随机噪声
    blur = aug  # 是否添加随机模糊
    curve = aug  # 是否训练丝状异物
    time_noise = False  # 是否模拟时间噪声
    label_smoothing = False  # 是否平滑标签
    input_shuffle = aug  # 是否随机交换图片位置
    input_norm = aug  # 图像是否标准化
    train_vis = False  # 训练数据可视化
    finetune = False  # 在之前权重基础上微调

    LR = 0.0001
    iters = 31500  # 训练迭代次数
    Batchsize = 16
    check_num = 63  # 每迭代check_num次保存一次权重
    accumulation_steps = 8

    # 测试参数
    save_result = False  # 是否保存预测结果图
    show_result = True  # 预测结果可视化
    bg_data = 'pacs'  # 背景数据集名称

    input_path = './Data/test/test_pics'  # 输入图片所在路径
    input_name = '35'  # 输入图片名,使用之前,带缺陷的图请重命名为名‘原名+_d’,无缺陷的图请重命名为名‘原名+_n’

    result_save_path = './result/'

    # 进行演示
    if bg_data == 'sintel':
        path = 'E:/Datasets/MPI-Sintel/'
        path1 = path + 'training/albedo/'
        path2 = path + 'training/clean/'
        path3 = path + 'training/final/'
        path4 = path + 'test/clean/'
        path5 = path + 'test/final/'

        path6 = path + 'training/flow_viz/'

        # imgs = glob.glob(path1 + '*/' + '*.png')
        # imgs.extend(glob.glob(path2 + '*/' + '*.png'))
        # imgs.extend(glob.glob(path3 + '*/' + '*.png'))
        # imgs.extend(glob.glob(path4 + '*/' + '*.png'))
        # imgs.extend(glob.glob(path5 + '*/' + '*.png'))

        imgs = glob.glob(path6 + '*/' + '*.png')
    elif bg_data == 'voc':
        path = '..\\EPro_PnP_main\\EPro_PnP_6DoF\\dataset\\bg_images\\VOC2012\\JPEGImages\\'
        imgs = glob.glob(path + '*.jpg')
    elif bg_data == 'pacs':
        path = 'E:/Datasets/PACS-master/PACS/'
        path1 = path + 'art_painting/'
        path2 = path + 'cartoon/'
        path3 = path + 'photo/'
        path4 = path + 'sketch/'
        imgs = glob.glob(path1 + '*/*.jpg')
        imgs.extend(glob.glob(path1 + '*/*.png'))
    elif bg_data == 'dtd':
        path = 'E:/Datasets/dtd/images/'
        imgs = glob.glob(path + '*/*.jpg')

    L = len(imgs)
    print('train on {} background pictures'.format(L))
    path = 'E:\\DL_Projects\\EPro_PnP_main\\EPro_PnP_6DoF\\dataset\\bg_images\\VOC2012\\JPEGImages\\'
    imgs_diff = glob.glob(path + '*.jpg')
    Ld = len(imgs_diff)
    imgs = glob.glob(r'D:\DL_extend\Difference_detecter\Data\test\test_pics\*_n.jpg')
    # for each in tqdm(imgs):
    #     input_name = each[52:-6]
    #     print(each.replace('_n', '_d'))
    #     print(each.replace('\\Data\\test\\test_pics', '\\DrawRect\\biaozhun\\labels').replace('jpg', 'txt'))
    #     if os.path.exists(each.replace('_n', '_d')) and os.path.exists(
    #             each.replace('Data\\test\\test_pics', 'DrawRect\\biaozhun\\labels').replace('jpg', 'txt')):
    #         print(input_name)
    #         demo_connector()
    # demo_connector()
    # demo_diff()
    # demo_dataset()
    # demo()
    # print(run_commands())
    demo_freestyle()
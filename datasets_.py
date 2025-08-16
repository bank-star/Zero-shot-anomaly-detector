import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import copy
import json
from util_ import (rotation, random_jitter, change_channel, add_noise, noisy, random_resize, rgb_norm_torch,
                  black_edge_crop, random_flip, gen_diffs, one_curve, virtual_light, flow_warp)

# from util_ import (rotation, random_jitter, change_channel, add_noise, noisy, random_resize, rgb_norm_torch, flow2rgb,
#                    black_edge_crop, random_flip, gen_diffs, one_curve, virtual_light, flow_warp, load_flow_to_numpy,
#                    flow_warp_v2, flow_warp_torch_v2, flow_normalize, random_perspective)


class DiffDataSet(Dataset):
    def __init__(self, cfg):

        self.background_path = cfg["background_path"]
        self.background_imgs = glob.glob(self.background_path + "*.jpg")
        self.background_imgs.extend(glob.glob(self.background_path + "*.png"))
        self.aug_items = cfg["aug_items"]
        self.H, self.W = cfg["H"], cfg["W"]
        self.moves = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]

    def __len__(self):
        return len(self.background_imgs)

    def __getitem__(self, index):
        data = {"inputb": None, "maskf": None, "maskb": None, "flowf": None, "flowb": None, "warped_src": None, "warped_ex": None}
        num_diff = np.random.randint(low=12, high=20, size=None, dtype='l')  # 生成差异的数量
        mask = np.zeros((self.H, self.W))  # 生成固定大小的空白图片（全黑）
        src = cv2.imread(self.background_imgs[index])
        src = cv2.resize(src, dsize=(self.W, self.H))  # 随机选择背景图片resize到640x480
        ex = copy.deepcopy(src)  # 将背景图复制一份用来添加差异
        xv, yv = np.meshgrid(np.arange(self.H) / 639., np.arange(self.W) / 639.)
        grids = np.stack((xv, yv, np.ones_like(yv)), 2)
        grids_processed = np.stack((xv, yv, np.ones_like(yv)), 2)
        if "rot" in self.aug_items:
            rot_flag = np.random.randint(0, 2)
            if rot_flag == 1:
                angle = np.random.randint(-200, 201) / 10
                ex = rotation(img=ex, angle=angle)
                grids_processed = rotation(img=grids_processed, angle=angle)
            elif rot_flag == 0:
                angle = np.random.randint(-200, 201) / 10
                src = rotation(img=src, angle=angle)
                grids = rotation(img=grids, angle=angle)
        if "jit" in self.aug_items:
            jit_flag = np.random.randint(0, 2)
            if jit_flag == 1:
                tx, ty = np.random.choice(self.moves, size=2)
                ex = random_jitter(ex, tx, ty)
                grids_processed = random_jitter(grids_processed, tx, ty)
            elif jit_flag == 0:
                tx, ty = np.random.choice(self.moves, size=2)
                src = random_jitter(src, tx, ty)
                grids = random_jitter(grids, tx, ty)
        flow = grids_processed - grids
        if "crop" in self.aug_items:
            src = black_edge_crop(src, self.H, self.W)
            ex = black_edge_crop(ex, self.H, self.W)
            flow = black_edge_crop(flow, self.H, self.W)
        if "chan_change" in self.aug_items:
            if np.random.randint(0, 2) == 1:
                sd = np.random.randint(low=0, high=6)
                ex = change_channel(ex, sd=sd)
                src = change_channel(src, sd=sd)
        if "flip" in self.aug_items:
            if np.random.randint(0, 2) == 1:
                sd = np.random.randint(low=0, high=6)
                ex = random_flip(ex, sd)
                src = random_flip(src, sd)
        if "chan_change" in self.aug_items:
            if np.random.randint(0, 2) == 1:
                sd = np.random.randint(low=0, high=6)
                ex = change_channel(ex, sd=sd)
                src = change_channel(src, sd=sd)
        if "flip" in self.aug_items:
            if np.random.randint(0, 2) == 1:
                sd = np.random.randint(low=0, high=6)
                ex = random_flip(ex, sd)
                src = random_flip(src, sd)
                flow = random_flip(flow, sd)
                if sd == 1:
                    flow[:, :, 0:1] = -flow[:, :, 0:1]
                elif sd == 2:
                    flow[:, :, 1:2] = -flow[:, :, 1:2]
                elif sd == 3:
                    flow = -flow
        index2 = np.random.randint(1, len(self.background_imgs))
        index2 = index2 - 1 if index==index2 else index2
        diff_img = cv2.imread(self.background_imgs[index2])
        diff_src = cv2.resize(diff_img, dsize=(self.W, self.H))
        ex, mask = gen_diffs(ex, mask, diff_src, num_diff, self.H, self.W)
        if "curve" in self.aug_items:
            ex, mask = one_curve(ex, mask, self.H, self.W)
        if "noise" in self.aug_items:
            if np.random.randint(0, 10) == 1:
                ex = noisy(noise_typ='gauss', image=ex)
            elif np.random.randint(0, 10) == 2:
                ex = noisy(noise_typ='s&p', image=ex)
            elif np.random.randint(0, 10) >= 5:
                ex = add_noise(ex)
        if "blur" in self.aug_items:
            k = np.random.randint(1, 3) * 2 + 1
            if np.random.randint(0, 4) == 1:
                ex = cv2.GaussianBlur(ex, (k, k), 0)
        if "light" in self.aug_items:
            if np.random.randint(0, 3) == 1:
                ex = virtual_light(ex)
        fg = np.random.randint(0, 2)
        right = np.random.randint(10, 15)
        up = np.random.randint(10, 15)
        flow = random_resize(flow, right, up, fg, inter_type=cv2.INTER_LINEAR)
        ex = random_resize(ex, right, up, fg)
        src = random_resize(src, right, up, fg)
        mask = random_resize(mask, right, up, fg)
        flowf = -flow[:, :, 0:2]
        flowb = flow[:, :, 0:2]
        maskb = mask
        maskf = flow_warp(maskb, flowf)
        warped_ex = flow_warp(ex, flowf, inter_type=cv2.INTER_LINEAR)
        warped_src = flow_warp(src, flowb, inter_type=cv2.INTER_LINEAR)
        data1 = torch.cuda.FloatTensor(ex).permute(2, 0, 1)
        data2 = torch.cuda.FloatTensor(src).permute(2, 0, 1)
        inputb = torch.cat((data1, data2), dim=0)
        flowf = torch.cuda.FloatTensor(flowf).permute(2,0,1)
        flowb = torch.cuda.FloatTensor(flowb).permute(2,0,1)
        maskf = torch.cuda.FloatTensor(maskf).unsqueeze(0)
        maskb = torch.cuda.FloatTensor(maskb).unsqueeze(0)
        warped_src = torch.cuda.FloatTensor(warped_src).permute(2,0,1)
        warped_ex = torch.cuda.FloatTensor(warped_ex).permute(2,0,1)
        data["inputb"]=inputb
        data["flowf"]=flowf
        data["flowb"]=flowb
        data["maskf"]=maskf
        data["maskb"]=maskb
        data["warped_src"]=warped_src
        data["warped_ex"]=warped_ex

        return data


if __name__ == "__main__":
    with open('./diff_configs.json', 'r') as f:
        cfg=json.load(f)
    dataset = DiffDataSet(cfg=cfg)
    loader = DataLoader(dataset=dataset, batch_size=cfg["Batchsize"])
    for i, (_) in enumerate(loader):
        pass

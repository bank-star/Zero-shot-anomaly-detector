import copy
import os

import cv2
import numpy as np
import torch
import torchvision.models
import multiprocessing
from network_ import *
from main_test_ import demo
from datasets_ import DiffDataSet
from torch.utils.data import DataLoader
import json
from util_ import buildlogger, seed_torch, rgb_norm_torch

def train_cls():

    with open('./diff_configs.json', 'r') as f:
        cfg=json.load(f)

    dd = DiffDataSet(cfg=cfg)

    dl = DataLoader(dataset=dd, batch_size=cfg["Batchsize"], shuffle=True, num_workers=8)

    logger = buildlogger()

    net = Net5(Norm=False, mode_type=cfg["mode_type"]).cuda()

    net.train()
    optimizer = torch.optim.AdamW([
        {'params': net.parameters(), 'weight_decay': cfg["wd"], 'lr': cfg["LR"]},
    ])
    id = 0
    for epoch in range(cfg["epoches"]):
        for i, data in enumerate(dl):
            if "input_norm" in cfg["aug_items"]:
                data["inputb"] = rgb_norm_torch(data["inputb"]) 
                data["warped_src"] = rgb_norm_torch(data["warped_src"])
                data["warped_ex"] = rgb_norm_torch(data["warped_ex"])

            # backward
            outb = net(data["inputb"])
            loss_rst1 = nn.SmoothL1Loss()(outb["image"], data["warped_src"])
            loss_bce1 = nn.BCEWithLogitsLoss()(outb["mask"], data["maskb"])
            loss_flow1 = (nn.SmoothL1Loss(reduction="none")(outb["flow"], data["flowb"]) * (1 - data["maskb"])).mean()
            loss1 = loss_bce1 + loss_flow1 + loss_rst1

            # forward
            outf = net(torch.cat((data["inputb"][:, 3:6, :, :], data["inputb"][:, 0:3, :, :]), dim=1))
            loss_rst2 = nn.SmoothL1Loss()(outf["image"], data["warped_ex"])
            loss_bce2 = nn.BCEWithLogitsLoss()(outf["mask"], data["maskf"])
            loss_flow2 = (nn.SmoothL1Loss(reduction="none")(outf["flow"], data["flowf"]) * (1 - data["maskf"])).mean()
            loss2 = loss_bce2 + loss_flow2 + loss_rst2

            loss = loss1 + loss2

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            logger.info(
                'iter:[%d/%d], lr: %.8f, weight_decay: %.6f,loss: %.6f loss_bce: %.6f loss_rst: %.6f loss_flow: %.6f'
                % (id, len(dd) * cfg["epoches"], optimizer.param_groups[0]['lr'],
                   optimizer.param_groups[0]['weight_decay'],
                   loss, loss_bce1 + loss_bce2, loss_rst1 + loss_rst2, loss_flow1 + loss_flow2))
            if (id + 1) % cfg["check_num"] == 0:
                torch.save(net.state_dict(), './ckpt/checkpoint_' + str(id + 1) + '.pth')
                tmp_result = demo(ckpt='./ckpt/checkpoint_' + str(id + 1) + '.pth')
                logger.info("metrics: %s" % (tmp_result))
                net.train()
            id += 1
    torch.save(net.state_dict(), './ckpt/model.pth')
    print('done')


if __name__ == '__main__':

    seed_torch(3407)

    is_train = True  # 默认为测试
    aug = True
    multiprocessing.set_start_method('spawn')
    train_cls()

import copy
import os

import cv2
import numpy as np
import torch
import torchvision.models
from torch.utils.data import DataLoader

from network_ import *
from util_ import *
from main_test_ import demo
angles = [-10, -9, -8, -7, -6, -5, 5, 6, 7, 8, 9, 10]
moves = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]

def make_datasets(batchsize ,kkk):
    data = {"inputb": [], "maskf": [], "maskb": [], "flowf": [], "flowb": [], "warped_src": [], "warped_ex": []}

    for a in range(batchsize):
        num_diff = np.random.randint(low=12, high=20, size=None, dtype='l')  # 生成差异的数量
        mask = np.zeros((H, W))  # 生成固定大小的空白图片（全黑）
        id = np.random.randint(low=0, high=L)
        src = cv2.imread(imgs[id])
        src = cv2.resize(src, dsize=(W, H))  # 随机选择背景图片resize到640x480
        ex = copy.deepcopy(src)  # 将背景图复制一份用来添加差异
        if train_vis:
            cv2.line(src, (319, 0), (319, 639), (0, 255, 0), 1)
            cv2.line(src, (0, 319), (639, 319), (0, 255, 0), 1)
            cv2.line(ex, (319, 0), (319, 639), (0, 255, 0), 1)
            cv2.line(ex, (0, 319), (639, 319), (0, 255, 0), 1)

        xv, yv = np.meshgrid(np.arange(H) / 639., np.arange(W) / 639.)
        grids = np.stack((xv, yv, np.zeros_like(yv)), 2)
        grids_processed = np.stack((xv, yv, np.zeros_like(yv)), 2)

        if rot:
            rot_flag = np.random.randint(0, 2)
            if rot_flag == 1:
                #angle = np.random.choice(angles) / 4
                angle = np.random.randint(-100, 101)/10
                mask = rotation(img=mask, angle=angle)
                ex = rotation(img=ex, angle=angle)
                grids_processed = rotation(img=grids_processed, angle=angle)
            elif rot_flag == 0:
                #angle = np.random.choice(angles) / 4
                angle = np.random.randint(-100, 101)/10
                src = rotation(img=src, angle=angle)
                grids = rotation(img=grids, angle=angle)
        if jit:
            jit_flag = np.random.randint(0, 2)
            if jit_flag == 1:
                tx, ty = np.random.choice(moves, size=2)
                ex = random_jitter(ex, tx, ty)
                mask = random_jitter(mask, tx, ty)
                grids_processed = random_jitter(grids_processed, tx, ty)
            elif jit_flag == 0:
                tx, ty = np.random.choice(moves, size=2)
                src = random_jitter(src, tx, ty)
                grids = random_jitter(grids, tx, ty)

        flow = grids_processed - grids
        if crop:
            src = black_edge_crop(src, H, W)
            ex = black_edge_crop(ex, H, W)
            flow = black_edge_crop(flow, H, W)
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
                flow = random_flip(flow, sd)
                if sd==1:
                    flow[:,:,0:1] = -flow[:,:,0:1]
                elif sd==2:
                    flow[:,:,1:2] = -flow[:,:,1:2]
                elif sd==3:
                    flow = -flow

        diff_img = cv2.imread(imgs[np.random.randint(0, len(imgs))])
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
        flow = random_resize(flow, right, up, fg, inter_type=cv2.INTER_LINEAR)
        ex = random_resize(ex, right, up, fg)
        src = random_resize(src, right, up, fg)
        mask = random_resize(mask, right, up, fg)


        if time_noise:
            if np.random.randint(0, 3) == 1:
                ex = add_time_noise(ex)

        # forward: src--->ex
        # backward: ex--->src
        flowf = -flow[:,:,0:2]
        flowb = flow[:,:,0:2]

        maskb = mask
        maskf = flow_warp(maskb, flowf)

        warped_ex = flow_warp(ex, flowf, inter_type=cv2.INTER_LINEAR)
        warped_src = flow_warp(src, flowb, inter_type=cv2.INTER_LINEAR)

        # invalid_region = flow_warp(np.zeros_like(maskf), flowf)
        # maskb[invalid_region == 1] = 1

        if train_vis:
            # occlude_mask = (1 - mask)[:, :, np.newaxis].repeat(repeats=3, axis=2)
            # show_warp(flowb, src, ex, occlude_mask)
            # cv2.imshow("flowb-flowf", np.hstack((flow2rgb(flowb), flow2rgb(flowf))).astype(np.uint8))
            # cv2.imshow("ex-src", np.hstack((ex, src)).astype(np.uint8))
            # cv2.imshow("maskb-maskf", np.hstack((maskb, maskf)))
            cv2.imshow("warpex", np.hstack((warped_ex, ex)).astype(np.uint8))
            cv2.imshow("warpsrc", np.hstack((warped_src, src)).astype(np.uint8))
            cv2.waitKey()
        # mask_temp=np.repeat(mask[:,:,np.newaxis], 3, 2) * 255
        # cv2.imwrite('./temp/'+str(batchsize*kkk+a)+'.jpg', np.hstack((src,ex,mask_temp, flow2rgb(flowb))))

        # if batchsize * kkk + a <= 99:
        #     cv2.imwrite('E:/Papers/word/revise2/samples_100/train/' + str(batchsize * kkk + a) + '_reference.jpg', src)
        #     cv2.imwrite('E:/Papers/word/revise2/samples_100/train/' + str(batchsize * kkk + a) + '_test.jpg', ex)
        #     cv2.imwrite('E:/Papers/word/revise2/samples_100/train/' + str(batchsize * kkk + a) + '_mask.jpg', mask_temp)
        #     cv2.imwrite('E:/Papers/word/revise2/samples_100/train/' + str(batchsize * kkk + a) + '_flow.jpg', flow2rgb(flowb))

        if input_shuffle:
            if np.random.randint(0, 2) == 0:
                data1 = torch.FloatTensor(ex).permute(2, 0, 1).unsqueeze(0)
                data2 = torch.FloatTensor(src).permute(2, 0, 1).unsqueeze(0)
            else:
                data2 = torch.FloatTensor(ex).permute(2, 0, 1).unsqueeze(0)
                data1 = torch.FloatTensor(src).permute(2, 0, 1).unsqueeze(0)
        else:
            data1 = torch.cuda.FloatTensor(ex).permute(2, 0, 1).unsqueeze(0)
            data2 = torch.cuda.FloatTensor(src).permute(2, 0, 1).unsqueeze(0)

        inputb = torch.cat((data1, data2), dim=1)
        flowf = torch.cuda.FloatTensor(flowf).unsqueeze(0).permute(0, 3, 1, 2)
        flowb = torch.cuda.FloatTensor(flowb).unsqueeze(0).permute(0, 3, 1, 2)
        maskf = torch.cuda.FloatTensor(maskf).unsqueeze(0).unsqueeze(0)
        maskb = torch.cuda.FloatTensor(maskb).unsqueeze(0).unsqueeze(0)
        warped_src = torch.cuda.FloatTensor(warped_src).unsqueeze(0).permute(0, 3, 1, 2)
        warped_ex = torch.cuda.FloatTensor(warped_ex).unsqueeze(0).permute(0, 3, 1, 2)

        data["inputb"].append(inputb)
        data["flowf"].append(flowf)
        data["flowb"].append(flowb)
        data["maskf"].append(maskf)
        data["maskb"].append(maskb)
        data["warped_src"].append(warped_src)
        data["warped_ex"].append(warped_ex)

    if input_norm:
        data["inputb"] = rgb_norm_torch(torch.cat(data["inputb"], dim=0))

    else:
        data["inputb"] = torch.cat(data["inputb"], dim=0)

    data["flowf"] = torch.cat(data["flowf"], dim=0)
    data["flowb"] = torch.cat(data["flowb"], dim=0)
    data["maskf"] = torch.cat(data["maskf"], dim=0)
    data["maskb"] = torch.cat(data["maskb"], dim=0)
    data["warped_src"] = rgb_norm_torch(torch.cat(data["warped_src"], dim=0))
    data["warped_ex"] = rgb_norm_torch(torch.cat(data["warped_ex"], dim=0))

    return data


def train_func():
    logger = buildlogger()

    net = Net5(Norm=False, mode_type="rst_flow").cuda()

    if finetune:
        checkpoint = torch.load('flownets_EPE1.951.pth.tar')
        net = get_state_static(net, checkpoint["state_dict"])

    net.train()
    optimizer = torch.optim.AdamW([
        {'params': net.parameters(), 'weight_decay': wd, 'lr': LR},
    ])

    for i in range(iters):
        data = make_datasets(batchsize=Batchsize, kkk=i)
    
        # backward
        outb = net(data["inputb"])
        loss_rst1 = nn.SmoothL1Loss()(outb["image"], data["warped_src"])
        loss_bce1 = nn.BCEWithLogitsLoss()(outb["mask"], data["maskb"])
        loss_flow1 = (nn.SmoothL1Loss(reduction="none")(outb["flow"], data["flowb"])*(1-data["maskb"])).mean() 
        loss1 = loss_bce1 + loss_flow1 + loss_rst1

        # forward
        outf = net(torch.cat((data["inputb"][:, 3:6, :, :], data["inputb"][:, 0:3, :, :]), dim=1))
        loss_rst2 = nn.SmoothL1Loss()(outf["image"], data["warped_ex"])
        loss_bce2 = nn.BCEWithLogitsLoss()(outf["mask"], data["maskf"])
        loss_flow2 = (nn.SmoothL1Loss(reduction="none")(outf["flow"], data["flowf"])*(1-data["maskf"])).mean() 
        loss2 = loss_bce2 + loss_flow2 + loss_rst2

        loss = loss1 + loss2
    
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
        logger.info(
                'iter:[%d/%d], lr: %.8f, weight_decay: %.6f,loss: %.6f loss_bce: %.6f loss_rst: %.6f loss_flow: %.6f'
            % (i, iters, optimizer.param_groups[0]['lr'],
               optimizer.param_groups[0]['weight_decay'],
               loss, loss_bce1 + loss_bce2, loss_rst1 + loss_rst2, loss_flow1 + loss_flow2))
        if (i + 1) % check_num == 0:
            torch.save(net.state_dict(), './ckpt/checkpoint_' + str(i + 1) + '.pth')
            tmp_result = demo(ckpt='./ckpt/checkpoint_' + str(i + 1) + '.pth')
            logger.info("metrics: %s" % (tmp_result))
            net.train()
    torch.save(net.state_dict(), './ckpt/model.pth')
    print('done')


if __name__ == '__main__':

    seed_torch(3407)

    H, W = 640, 640
    is_train = True  # 默认为测试
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
    input_shuffle = False  # 是否随机交换图片位置
    input_norm = aug  # 图像是否标准化
    train_vis = False  # 训练数据可视化
    finetune = False  # 在之前权重基础上微调

    # LR = 0.01
    LR = 0.0001
    wd = 0.5
    iters = 100000  # 训练迭代次数
    Batchsize = 8
    check_num = 1000  # 每迭代check_num次保存一次权重
    accumulation_steps = 8

    # 测试参数
    save_result = False  # 是否保存预测结果图
    show_result = True  # 预测结果可视化
    data = 'voc'  # 背景数据集名称

    input_path = './Data/test/test_pics'  # 输入图片所在路径
    input_name = '100'  # 输入图片名,使用之前,带缺陷的图请重命名为‘原名+_d’,无缺陷的图请重命名为‘原名+_n’

    result_save_path = './result/'

    if is_train:
        if data == 'sintel':
            path1 = 'E:/Datasets/MPI-Sintel-complete/training/albedo/'
            path2 = 'E:/Datasets/MPI-Sintel-complete/training/clean/'
            path3 = 'E:/Datasets/MPI-Sintel-complete/training/final/'
            imgs = glob.glob(path1 + '*/' + '*.png')
            imgs.extend(glob.glob(path2 + '*/' + '*.png'))
            imgs.extend(glob.glob(path3 + '*/' + '*.png'))
        elif data == 'coco_test':
            path = 'E:\\datasets\\test2017\\'
            imgs = glob.glob(path + '*.jpg')
        elif data == 'voc':
            path = r'E:\CV_Datasets\VOCdevkit\VOC2012\JPEGImages\\'
            imgs = glob.glob(path + '*.jpg')
        else:
            path = './data_for_train/'
            imgs = glob.glob(path + '*.jpg')

        diff_imgs = glob.glob(r'.\data_for_train\finetune\train\diffs\*.png')
        L = len(imgs)
        print('train on {} background pictures'.format(L))
        # 开始训练
        train_func()

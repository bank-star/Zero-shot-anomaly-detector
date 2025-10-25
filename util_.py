import copy
import glob
import math
import shutil
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import random
from scipy.stats import truncnorm
import datetime
import subprocess


def random_perspective(image, pts1, pts2):
    M = cv2.getPerspectiveTransform(pts1, pts2)
    height, width = image.shape[:2]
    warped_image = cv2.warpPerspective(image, M, (width, height), cv2.INTER_NEAREST)
    return warped_image


def flow_warp_v2(img, flow, inter_type=cv2.INTER_NEAREST, is_normed=True):
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    h, w, c = img.shape
    y, x = np.meshgrid(np.arange(w), np.arange(h))
    grid = [y, x]
    grid = np.stack(grid, axis=-1)
    flow_ = flow.copy()

    if is_normed:
        # 输入光流已经归一化
        flow_ = flow_anti_normalize(flow_)

    new_flow = grid + flow_
    new_flow = new_flow.astype(np.float32)
    new_img = cv2.remap(img, new_flow, None, inter_type, borderValue=0)
    return new_img


def flow_warp_torch_v2(img, flow):
    n = img.ndim
    if n == 3:
        img = torch.FloatTensor(img).unsqueeze(0).permute(0, 3, 1, 2)
        flow = torch.FloatTensor(flow).unsqueeze(0)
    b, c, h, w = img.size()
    y, x = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')
    grid = [y, x]
    grid = torch.stack(grid, dim=-1).unsqueeze(0).cuda()
    flow = flow.permute(0, 2, 3, 1)
    new_flow = grid + flow
    wh = torch.cuda.FloatTensor([w - 1, h - 1]).reshape(1, 1, 1, 2)
    new_flow = torch.div(new_flow, wh)
    # flow[:, :, 0:1] = flow[:, :, 0:1] / float(w-1)
    # flow[:, :, 1:2] = flow[:, :, 1:2] / float(h-1)
    new_flow = (2. * new_flow - 1.).float()
    new_img = F.grid_sample(img, new_flow, mode='bilinear', align_corners=True)
    if n == 3:
        new_img = new_img.squeeze(0).permute(1, 2, 0).numpy()
    return new_img

def flow_anti_normalize(flow):
    h, w, c = flow.shape
    if c == 2:
        wh = np.asarray([w - 1, h - 1], dtype=np.float32).reshape(1, 1, 2)
    if c == 3:
        wh = np.asarray([w - 1, h - 1, 0], dtype=np.float32).reshape(1, 1, 3)
    return flow * wh


def flow_normalize(flow):
    h, w, c = flow.shape
    if c == 2:
        wh = np.asarray([w - 1, h - 1], dtype=np.float32).reshape(1, 1, 2)
    if c == 3:
        wh = np.asarray([w - 1, h - 1, 0], dtype=np.float32).reshape(1, 1, 3)
    return flow / wh


def load_flow_to_numpy(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert (202021.25 == magic), 'Magic number incorrect. Invalid .flo file'
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
    data2D = np.resize(data, (w, h, 2))
    return data2D


def load_flow_to_png(path):
    flow = load_flow_to_numpy(path)
    image = flow_to_image(flow)
    return image


def flow_to_image(flow, max_flow=256):
    if max_flow is not None:
        max_flow = max(max_flow, 1.)
    else:
        max_flow = np.max(flow)

    n = 8
    u, v = flow[:, :, 0], flow[:, :, 1]
    mag = np.sqrt(np.square(u) + np.square(v))
    angle = np.arctan2(v, u)
    im_h = np.mod(angle / (2 * np.pi) + 1, 1)
    im_s = np.clip(mag * n / max_flow, a_min=0, a_max=1)
    im_v = np.clip(n - im_s, a_min=0, a_max=1)
    im = cv2.cvtColor(np.stack([im_h, im_s, im_v], 2), cv2.COLOR_HSV2BGR)
    return (im * 255).astype(np.uint8)

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(NonLocalBlock, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

def run_commands(rot_jit_flag):
    if not rot_jit_flag:
        print("src_test")
        result = subprocess.run("/home/yxh/DeepPCB-master/evaluation/get_metrics.sh",shell=True,stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    else:
        print("rot_jit_test")
        result = subprocess.run("/home/yxh/DeepPCB-master/evaluation/get_rot_jit_metrics.sh",shell=True,stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        #print(result.stdout) 
    # 返回第4条执行的结果并打印
    return result.stdout


def show_warp(flow, src, ex, occlude_mask):
    warped_ex = flow_warp_torch(ex, flow)
    warped_occlude_mask = flow_warp_torch(occlude_mask, flow)

    warped_common_mask = flow_warp_torch(np.ones_like(ex), flow)
    occlude_mask[warped_common_mask == 0] = 0

    warped_ex[warped_occlude_mask == 0] = 0
    showing = np.vstack((np.hstack((ex, src)), np.hstack((warped_ex, flow2rgb(flow))))).astype(np.uint8)
    cv2.namedWindow('warped', 0)
    cv2.namedWindow('forward/backward_occlusion', 0)
    cv2.imshow('forward/backward_occlusion', 1 - np.hstack((occlude_mask, warped_occlude_mask)))
    cv2.imshow('warped', showing)
    cv2.waitKey()


def flow_warp(img, flow, inter_type=cv2.INTER_NEAREST):
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    h, w, c = img.shape
    y, x = np.meshgrid(np.arange(w), np.arange(h))
    grid = [y, x]
    grid = np.stack(grid, axis=-1)
    # flow = flow * 639.
    flow = flow * (h-1)
    new_flow = grid + flow
    new_flow = new_flow.astype(np.float32)
    new_img = cv2.remap(img, new_flow, None, inter_type, borderValue=0)
    return new_img


def flow_warp_torch(img, flow):
    n = img.ndim
    if n == 3:
        img = torch.FloatTensor(img).unsqueeze(0).permute(0, 3, 1, 2)
        flow = torch.FloatTensor(flow).unsqueeze(0)
    b, c, h, w = img.size()
    y, x = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')
    grid = [y, x]
    grid = torch.stack(grid, dim=-1).unsqueeze(0).cuda()
    # flow = flow.permute(0, 2, 3, 1) * 639.
    flow = flow.permute(0, 2, 3, 1) * (h-1)
    new_flow = grid + flow
    # new_flow = (2. * new_flow / 639. - 1.).float()
    new_flow = (2. * new_flow / (h-1) - 1.).float()
    new_img = F.grid_sample(img, new_flow, mode='bilinear', align_corners=True)
    if n == 3:
        new_img = new_img.squeeze(0).permute(1, 2, 0).numpy()
    return new_img


def EPE(input_flow, target_flow, sparse=False, mean=True):
    EPE_map = torch.norm((target_flow - input_flow), 2, 1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum() / batch_size


def multiscaleEPE(network_output, target_flow, weights=None, sparse=False):
    def one_scale(output, target, sparse):
        b, _, h, w = output.size()
        target_scaled = F.interpolate(target, (h, w), mode="area")
        return EPE(output, target_scaled, sparse, mean=False)

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
    assert len(weights) == len(network_output)

    loss = 0
    for output, weight in zip(network_output, weights):
        loss += weight * one_scale(output, target_flow, sparse)
    return loss


def flow2rgb(flow_map, max_value=None):
    flow_map_np = flow_map.transpose(2, 0, 1)
    _, h, w = flow_map_np.shape
    flow_map_np[:, (flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float("nan")
    rgb_map = np.ones((3, h, w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5 * (normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return 255 * rgb_map.clip(0, 1).transpose(1, 2, 0)


def buildlogger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # 使用FileHandler输出到文件
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    fh = logging.FileHandler('./logs/log_' + str(timestamp) + '.txt')
    fh.setFormatter(formatter)

    # 使用StreamHandler输出到屏幕
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    # 添加两个Handler
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


def generate_curve(img, control_points, color, thickness):
    # 使用贝塞尔曲线拟合头发弯曲线条
    curve_points = []
    num_points = 100  # 调整生成的曲线上的点的数量
    for t in np.linspace(0, 1, num_points):
        curve_point = np.power(1 - t, 3) * control_points[0] + 3 * np.power(1 - t, 2) * t * control_points[1] + 3 * (
                1 - t) * np.power(t, 2) * control_points[2] + np.power(t, 3) * control_points[3]
        curve_points.append(curve_point)
    curve_points = np.array(curve_points, dtype=np.int32)
    # 绘制头发弯曲线条
    cv2.polylines(img, [curve_points], isClosed=False, color=color, thickness=thickness)
    return img


def seed_torch(seed=1029):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def get_state_static(model, data):
    if data is not None:
        model_dict = model.state_dict()
        print('model keys')
        print('=================================================')
        for k, v in model_dict.items():
            print(k)
        print('=================================================')

        print('data keys')
        print('=================================================')
        for k, v in data.items():
            print(k)
        print('=================================================')

        pretrained_dict = {k: v for k, v in data.items() if
                           k in model_dict and v.size() == model_dict[k].size()}
        print('load the following keys from the pretrained model')
        print('=================================================')
        for k, v in pretrained_dict.items():
            print(k)
        print('=================================================')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        print('No pretrained')
    return model


class CSA(nn.Module):
    def __init__(self, in_channel, scale_factor):
        super(CSA, self).__init__()
        self.scale_factor = scale_factor
        self.qkv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel // scale_factor * 3,
                             kernel_size=1, stride=1, padding=0, bias=False)
        self.upc = nn.Conv2d(in_channels=in_channel // scale_factor, out_channels=in_channel,
                             kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        q, k, v = torch.split(self.qkv(x), split_size_or_sections=c // self.scale_factor, dim=1)
        q = q.view(b, c // self.scale_factor, -1).transpose(1, 2).unsqueeze(-1)
        k = k.view(b, c // self.scale_factor, -1).transpose(1, 2).unsqueeze(-1)
        v = v.view(b, c // self.scale_factor, -1).transpose(1, 2).unsqueeze(-1)

        at = nn.Softmax(dim=-1)(torch.matmul(q, k.transpose(2, 3)) / ((c // self.scale_factor) ** 0.5))

        return self.upc(torch.matmul(at, v).squeeze(-1).transpose(1, 2).view(b, c // self.scale_factor, h, w)) + x


class CSA2(nn.Module):

    def __init__(self, in_channel, scale, input_size, kernel_size, stride):
        super(CSA2, self).__init__()

        assert (input_size[0] - kernel_size[0]) % stride[0] == 0 and (input_size[1] - kernel_size[1]) % stride[1] == 0
        assert in_channel % scale == 0
        self.patched_size = (int((input_size[0] - kernel_size[0]) / stride[0] + 1),
                             int((input_size[1] - kernel_size[1]) / stride[1] + 1))
        self.num_patch = self.patched_size[0] * self.patched_size[1]
        self.scale = scale
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.dk = kernel_size[0] * kernel_size[1]
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride)
        self.fold = nn.Fold(output_size=input_size, kernel_size=kernel_size, stride=stride)
        self.softmax = nn.Softmax(dim=-1)
        self.qkv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel * 3 // scale, kernel_size=(1, 1),
                             stride=(1, 1), padding=(0, 0), bias=True)
        self.upc = nn.Conv2d(in_channels=in_channel // scale, out_channels=in_channel, kernel_size=(1, 1),
                             stride=(1, 1), padding=(0, 0), bias=True)

    def forward(self, x):
        short_cut = x
        b, c, h, w = x.size()
        c = c // self.scale
        qkv = self.unfold(self.qkv(x)).transpose(1, 2).reshape(b, self.num_patch, c, -1)
        q, k, v = torch.split(qkv, split_size_or_sections=self.dk, dim=3)
        energy = torch.matmul(q, k.transpose(2, 3)) / (c ** 0.5)
        x = torch.matmul(self.softmax(energy), v)
        x = self.fold(x.reshape(b, self.num_patch, -1).transpose(1, 2))

        return self.upc(x) + short_cut


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, act=True):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out) if act else nn.Identity(),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

def upconv(in_planes, out_planes, scale_factor=2, act=False):
    if act:
        return nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
    else:
        return nn.Sequential(
        nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True),
    )

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=False),
            LayerNorm(out_planes, data_format='channels_first'),
            # nn.InstanceNorm2d(out_planes),
            # nn.ReLU(inplace=True),
            nn.GELU(),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=True),
            # nn.GELU()
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(0.1, True)
            # nn.SiLU(inplace=True)
        )


def predict_mask(in_planes):
    return nn.Conv2d(in_planes, 1, kernel_size=3, stride=1, padding=1, bias=False)


def predict_image(in_planes):
    return nn.Conv2d(in_planes, 3, kernel_size=3, stride=1, padding=1, bias=False)


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=False)


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        # nn.GELU(),
        nn.ReLU(inplace=True),
        # nn.LeakyReLU(0.1,True)
        # nn.SiLU(inplace=True),
    )


def crop_like(input, target=None):
    if target is None:
        return input[:, :, :640, :640]
    else:
        if input.size()[2:] == target.size()[2:]:
            return input
        else:
            return input[:, :, :target.size(2), :target.size(3)]


def rgb_norm(image):
    r, g, b = cv2.split(image)
    eps = 1e-7
    r = (r - np.mean(r)) / (np.std(r) + eps)
    g = (g - np.mean(g)) / (np.std(g) + eps)
    b = (b - np.mean(b)) / (np.std(b) + eps)
    return cv2.merge((r, g, b))
    # return image / 127.5 - 1


def gen_heatmap(result):
    heatmapshow = None
    heatmapshow = cv2.normalize(result, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                dtype=cv2.CV_8U)
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
    return heatmapshow


def rgb_norm_torch(image):
    mean = torch.mean(image, dim=(2, 3), keepdim=True)
    std = torch.std(image, dim=(2, 3), keepdim=True)
    normalized_tensor = (image - mean) / (std + 1e-7)
    return normalized_tensor

def gaussian_weight(distance, sigma=1):
    return np.exp(-0.5 * (distance / sigma) ** 2)


def apply_gaussian_blur(mask):
    blurred_image = np.zeros_like(mask, dtype=np.float32)

    # 找到掩码中每个封闭区域的中心点
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])

            # 生成区域内的坐标
            region_mask = np.zeros_like(mask)
            cv2.drawContours(region_mask, [contour], -1, 1, thickness=cv2.FILLED)
            region_y, region_x = np.nonzero(region_mask)

            # 计算每个像素到中心点的距离
            distance = np.sqrt((region_y - center_y) ** 2 + (region_x - center_x) ** 2)

            x, y, w, h = cv2.boundingRect(contour)  # 获取轮廓顶点及边长
            # 计算高斯权重
            weight = gaussian_weight(distance, (w + h) / 4)

            # 加权平均得到模糊后的像素值
            blurred_image[region_y, region_x] += mask[region_y, region_x] * weight

    return blurred_image


def random_flip(input, flag):
    if flag == 1:
        return np.fliplr(input)
    elif flag == 2:
        return np.flipud(input)
    elif flag == 3:
        return np.flipud(np.fliplr(input))
    else:
        return input


def chromatic_transform(im, label=None, d_h=None, d_s=None, d_l=None):
    """
    Given an image array, add the hue, saturation and luminosity to the image
    """
    # Set random hue, luminosity and saturation which ranges from -0.1 to 0.1
    if d_h is None:
        d_h = (np.random.rand(1) - 0.5) * 0.02 * 180
    if d_l is None:
        d_l = (np.random.rand(1) - 0.5) * 0.2 * 256
    if d_s is None:
        d_s = (np.random.rand(1) - 0.5) * 0.2 * 256
    # Convert the BGR to HLS
    hls = cv2.cvtColor(im, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)
    # Add the values to the image H, L, S
    new_h = (h + d_h) % 180
    new_l = np.clip(l + d_l, 0, 255)
    new_s = np.clip(s + d_s, 0, 255)
    # Convert the HLS to BGR
    new_hls = cv2.merge((new_h, new_l, new_s)).astype('uint8')
    new_im = cv2.cvtColor(new_hls, cv2.COLOR_HLS2BGR)

    if label is not None:
        I = np.where(label > 0)
        new_im[I[0], I[1], :] = im[I[0], I[1], :]
    return new_im


def gen_heatmap(result):
    heatmapshow = None
    heatmapshow = cv2.normalize(result, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                dtype=cv2.CV_8U)
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
    return heatmapshow


def add_noise(image):
    # random number
    r = np.random.rand(1)
    # gaussian noise
    if r < 0.9:
        row, col, ch = image.shape
        mean = 0
        var = np.random.rand(1) * 0.3 * 256
        sigma = var ** 0.5
        gauss = sigma * np.random.randn(row, col) + mean
        gauss = np.repeat(gauss[:, :, np.newaxis], ch, axis=2)
        noisy = image + gauss
        noisy = np.clip(noisy, 0, 255)
    else:
        # motion blur
        sizes = [3, 5, 7, 9, 11, 15]
        size = sizes[int(np.random.randint(len(sizes), size=1))]
        kernel_motion_blur = np.zeros((size, size))
        if np.random.rand(1) < 0.5:
            kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
        else:
            kernel_motion_blur[:, int((size - 1) / 2)] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        noisy = cv2.filter2D(image, -1, kernel_motion_blur)

    return noisy


def one_curve(ex, mask, H, W):
    x1 = np.random.randint(20, W - 20)
    y1 = np.random.randint(20, H - 20)
    x2 = max(min(x1 + np.random.randint(-20, 20), W - 1), 0)
    y2 = max(min(y1 + np.random.randint(-20, 20), H - 1), 0)
    x3 = max(min(x2 + np.random.randint(-40, 40), W - 1), 0)
    y3 = max(min(y2 + np.random.randint(-40, 40), H - 1), 0)
    x4 = max(min(x3 + np.random.randint(-80, 80), W - 1), 0)
    y4 = max(min(y3 + np.random.randint(-80, 80), H - 1), 0)
    control_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)
    thickness = np.random.randint(1, 3)
    mask_for_curve = np.zeros_like(mask)
    mask_for_curve = generate_curve(mask_for_curve, control_points, 1, thickness)
    mask = generate_curve(mask, control_points, 1, thickness)
    ex_mean = ex[mask_for_curve == 1].sum() / np.sum(mask_for_curve == 1)
    ex = generate_curve(ex.copy(), control_points, random_gray(ex_mean / 3), thickness)
    return ex, mask


def gen_diffs(ex, mask, diff_src, num_diff, H, W):
    big_cnt = 0
    for b in range(num_diff):
        # 随机确定左上角的点和长宽
        if np.random.randint(0, 30) == 0 and big_cnt == 0:
            x_l, y_l, w, h = np.random.randint(low=99, high=W - 300, size=None, dtype='l'), \
                np.random.randint(low=99, high=H - 300, size=None, dtype='l'), \
                np.random.randint(low=100, high=300, size=None, dtype='l'), \
                np.random.randint(low=100, high=300, size=None, dtype='l')
            big_cnt = 1
            big_flag = 1
        else:
            x_l, y_l, w, h = np.random.randint(low=49, high=W - 50, size=None, dtype='l'), \
                np.random.randint(low=49, high=H - 50, size=None, dtype='l'), \
                np.random.randint(low=10, high=50, size=None, dtype='l'), \
                np.random.randint(low=10, high=50, size=None, dtype='l')
            big_flag = 0

        # 在此方形区域内生成差异， 同时将空mask图片中的对应区域变为1（白色）作为标签
        points = [[x_l + w // 2, y_l + h // 2]]
        for i in range(x_l, x_l + w):
            for j in range(y_l, y_l + h):
                if big_flag == 1:
                    if np.random.randint(0, 5000) == 1:
                        points.append([i, j])
                else:
                    if np.random.randint(0, 100) == 1:
                        # if np.random.randint(0, 10) == 1:
                        points.append([i, j])
        random.shuffle(points)
        pts = np.asarray([points], dtype=np.int32)
        hull = get_hull(pts).transpose(1, 0, 2)
        mask = cv2.fillPoly(mask.copy(), hull, color=1)
        ex = cv2.fillPoly(ex.copy(), hull, color=(0, 0, 0))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask.copy(), kernel, 1)
    mask_filled = np.stack((mask, mask, mask), axis=2)
    mask_filled = cv2.GaussianBlur(mask_filled, (3, 3), 0, 0)
    ex = (255 * ((ex / 255) * (1 - mask_filled) + (diff_src / 255) * mask_filled)).astype(np.uint8)

    return ex, mask


def random_jitter(input, tx, ty):
    H = np.float32([[1, 0, tx], [0, 1, ty]])  # 定义平移矩阵
    rows, cols = input.shape[:2]  # 获取图像高宽(行列数)
    res = cv2.warpAffine(input, H, (cols, rows), cv2.INTER_NEAREST)
    return res


def get_hull(axis_list):
    hull = cv2.convexHull(axis_list, clockwise=True, returnPoints=True)
    return hull


def random_resize(img, right, up, fg, inter_type=cv2.INTER_NEAREST):
    H, W = img.shape[0], img.shape[1]
    scale = 1.0
    if fg == 0:
        if len(img.shape) == 3:
            img_croped = img[20:H - 20, 20:W - 20, :]
            if img.shape[2]==2:
                scale = W / img_croped.shape[0]
        else:
            img_croped = img[20:H - 20, 20:W - 20]
        img_resized = cv2.resize(img_croped, dsize=(W, H), interpolation=inter_type)
    elif fg == 1:
        img_resized = img
    else:
        img_paded = cv2.copyMakeBorder(img, int(up), int(up), int(right), int(right),
                                       cv2.BORDER_CONSTANT, value=0)
        if len(img.shape) == 3 and img.shape[2]==2:
            scale = W / img_paded.shape[0]
        img_resized = cv2.resize(img_paded, dsize=(W, H), interpolation=inter_type)
    return img_resized.copy() * scale


def rotation(img, angle):
    rows = img.shape[0]
    cols = img.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle=angle, scale=1)  # 向左旋转angle度并缩放为原来的scale倍
    img = cv2.warpAffine(img, M, (cols, rows), cv2.INTER_NEAREST)  # 第三个参数是输出图像的尺寸中心
    return img


def random_crop(img):
    sh, sw = img.shape[0], img.shape[1]
    x0, y0 = np.random.randint(0, sw // 2), np.random.randint(0, sh // 2)
    h, w = np.random.randint(sh // 4, sh // 2), np.random.randint(sw // 4, sw // 2)
    return img[y0:y0 + h, x0:x0 + w, :]


def black_edge_crop(img, H, W):
    if img.shape[2] == 2:
        img = img[30:H - 30, 30:W - 30, :]
        scale = H / img.shape[0]
        return cv2.resize(img, dsize=(W, H)) * scale
    return cv2.resize(img[30:H - 30, 30:W - 30, :], dsize=(W, H))


def random_pad(img):
    right = np.random.randint(100, 200)
    up = np.random.randint(100, 200)
    img = cv2.copyMakeBorder(img, int(up), int(up), int(right), int(right),
                             cv2.BORDER_CONSTANT, value=0)
    return img


def brightness_adjustment(img):
    blank = np.zeros_like(img)
    c = (1 + np.random.randint(low=-1, high=2, size=None, dtype='l') / 10.)
    img = cv2.addWeighted(img, c, blank, 1 - c, 0)
    return img


def random_color():
    b = np.random.randint(0, 127)
    g = np.random.randint(0, 127)
    r = np.random.randint(0, 127)
    return (b, g, r)


def random_gray(em):
    if em >= 128:
        B = np.random.randint(0, 30)
        G = np.random.randint(0, 30)
        R = np.random.randint(0, 30)
    else:
        B = np.random.randint(196, 226)
        G = np.random.randint(196, 226)
        R = np.random.randint(196, 226)
    return (B, G, R)


def change_channel(img, sd):
    b, g, r = cv2.split(img)
    if sd == 0:
        out = cv2.merge([b, r, g])
    elif sd == 1:
        out = cv2.merge([b, g, r])
    elif sd == 2:
        out = cv2.merge([r, g, b])
    elif sd == 3:
        out = cv2.merge([r, b, g])
    elif sd == 4:
        out = cv2.merge([g, b, r])
    elif sd == 5:
        out = cv2.merge([g, r, b])
    return out


def gasuss_noise(image, mu=0.0, sigma=0.01):
    """
	 添加高斯噪声
	:param image: 输入的图像
	:param mu: 均值
	:param sigma: 标准差
	:return: 含有高斯噪声的图像
	"""
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mu, sigma, image.shape)
    gauss_noise = image + noise
    if gauss_noise.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    gauss_noise = np.clip(gauss_noise, low_clip, 1.0)
    gauss_noise = np.uint8(gauss_noise * 255)
    return gauss_noise


def add_time_noise(image):
    mu = np.random.randint(0, 30) / 10.
    sigma = np.random.randint(0, 30) / 10.
    noise = np.random.normal(mu, sigma, image.shape)
    noisy_image = (image + noise) if np.random.randint(0, 2) == 0 else (image - noise)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


def noisy(noise_typ, image):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.001
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = (image / 255 + gauss) if np.random.randint(0, 2) == 1 else (image / 255 - gauss)
        noisy = cv2.normalize(noisy, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        noisy = np.uint8(noisy * 255)
        return noisy
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape[:2]]
        out[:, :, 0:1][tuple(coords)] = 255
        out[:, :, 1:2][tuple(coords)] = 255
        out[:, :, 2:3][tuple(coords)] = 255
        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape[:2]]
        out[:, :, 0:1][tuple(coords)] = 0
        out[:, :, 1:2][tuple(coords)] = 0
        out[:, :, 2:3][tuple(coords)] = 0
        return out


def polygon2mask(polygon, mask):
    points = np.array(polygon)

    # 使用多边形的点坐标创建一个包含多边形形状的路径
    path = points.reshape((-1, 1, 2))

    # 使用cv2.fillPoly()函数将路径填充到掩码中
    cv2.fillPoly(mask, [path], (1, 1, 1))

    # 返回生成的掩码
    return mask


def binary_pa(s, g):
    """
        calculate the pixel accuracy of two N-d volumes.
        s: the segmentation volume of numpy array
        g: the ground truth volume of numpy array
        """
    pa = ((s == g).sum()) / g.size
    return pa


def binary_iou(s, g):
    assert (len(s.shape) == len(g.shape))
    # 两者相乘值为1的部分为交集
    intersecion = np.multiply(s, g)
    # 两者相加，值大于0的部分为交集
    union = np.asarray(s + g > 0, np.float32)
    iou = intersecion.sum() / (union.sum() + 1e-10)
    return iou


def binary_evaluation(s, g):
    tp = np.asarray(s + g == 2, np.float32).sum()
    fn = np.asarray(g - s == 1, np.float32).sum()
    fp = np.asarray(s + g == 0, np.float32).sum()
    tn = np.asarray(g - s == -1, np.float32).sum()

    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1_score = 2 * precision * recall / (precision + recall)

    return recall, precision, F1_score, acc


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_list0(gj_area_file):
    path1 = "./DrawRect/biaozhun/labels/" + gj_area_file + ".txt"
    if not os.path.exists(path1):
        print("该型号标准位置文件缺失/或输入型号与对应标准文件名称不一致")
    file1 = open(path1, 'r')
    lines = file1.readlines()

    zb0, list0 = [], []
    for i in range(len(lines)):  # 取坐标
        if lines[i] != '(pt1,pt2):\n':
            zb0.append(lines[i][:-1])
    # print(zb0)
    for i in range(0, len(zb0)):  # 转换整数,获得多边形点列表
        zb0[i] = int(zb0[i])  # 多边形点列表
    # print(zb0)

    # 获得多边形最大矩形框点列表
    x_min, y_min, x_max, y_max = zb0[0], zb0[1], zb0[0], zb0[1]
    for i in range(0, len(zb0), 2):
        if x_min > zb0[i]:
            x_min = zb0[i]
        if x_max < zb0[i]:
            x_max = zb0[i]
        if y_min > zb0[i + 1]:
            y_min = zb0[i + 1]
        if y_max < zb0[i + 1]:
            y_max = zb0[i + 1]
    list0 = [x_min, y_min, x_max, y_max]  # 多边形最大矩形框点列表

    return zb0, list0


def get_bbox_from_mask_zoom(result, ex, biaoding_box):
    # 获取mask（灰度图）
    mask = cv2.normalize(result, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    mask = np.uint8(mask * 255)
    # 转换成二值图
    ret, mask = cv2.threshold(mask, mask.mean() * 0.1, 255, cv2.THRESH_BINARY)

    # blocksize = 31
    # C = -mask.mean()
    # mask = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, C)

    def mask_find_bboxs(mask):
        retval, labels, stats, centimgds = cv2.connectedComponentsWithStats(mask,
                                                                            connectivity=8)  # connectivity参数的默认值为8
        stats = stats[stats[:, 4].argsort()]
        return stats[:-1]  # 排除最外层的连通图

    bboxs = mask_find_bboxs(mask)
    width_scale = (biaoding_box[2] - biaoding_box[0]) / mask.shape[1]
    height_scale = (biaoding_box[3] - biaoding_box[1]) / mask.shape[0]
    S_threshold = (biaoding_box[3] - biaoding_box[1]) * (biaoding_box[2] - biaoding_box[0]) * 0.0001
    err_threshold = 0.1
    # print('err_thrashold:{}, S_thrashold:{}'.format(err_threshold, S_threshold))
    have_diff = False

    for b in bboxs:
        x0, y0 = b[0], b[1]
        x1 = b[0] + b[2]
        y1 = b[1] + b[3]
        start_point, end_point, bbox_width, bbox_height = (x0, y0), (x1, y1), x1 - x0, y1 - y0
        start_point = (int(x0 * width_scale + biaoding_box[0]), int(y0 * height_scale + biaoding_box[1]))
        end_point = (int(x1 * width_scale + biaoding_box[0]), int(y1 * height_scale + biaoding_box[1]))
        bbox_width = (x1 * width_scale + biaoding_box[0]) - (x0 * width_scale + biaoding_box[0])
        bbox_height = (y1 * height_scale + biaoding_box[1]) - (y0 * height_scale + biaoding_box[1])

        err = 0.01378330908618641 * max(bbox_height, bbox_width)
        S_box = bbox_height * bbox_width
        # print('err:{}, S_box:{}, is_box:{}'.format(err, S_box, (bbox_width / bbox_height) >= 0.33 and (
        #         bbox_width / bbox_height) <= 3.0))

        if (bbox_width / bbox_height) >= 0.1 and (
                bbox_width / bbox_height) <= 10.0 and S_box >= S_threshold and err >= err_threshold:
            have_diff = True
            color = (0, 0, 255)  # 边框颜色红
            thickness = 3  # 边框厚度1
            ex = np.ascontiguousarray(ex)
            ex = cv2.rectangle(ex, start_point, end_point, color, thickness)
            # pixel_err = max(int(8.55 * bbox_width), int(5.34 * bbox_height))
            # ex = cv2.putText(ex, str(err), (int(start_point[0]+bbox_width/2), int(start_point[1]+bbox_height/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)
    if have_diff:
        print('有异物')
    else:
        print('无异物')
    return ex


def calculate_mean_and_std(image):
    # 计算图片的均值和标准差
    mean = np.mean(image)
    std = np.std(image)
    return mean, std


def adjust_mean_and_std(source_image, target_image):
    # 计算原始图片和目标图片的均值和标准差
    source_mean, source_std = calculate_mean_and_std(source_image)
    target_mean, target_std = calculate_mean_and_std(target_image)

    # 调整目标图片的均值和标准差与原始图片相同
    adjusted_image = (target_image - target_mean) * (source_std / target_std) + source_mean
    return np.clip(adjusted_image, 0, 255).astype(np.uint8)


def extract_red_region(frame):
    h, w, _ = frame.shape
    mask1 = np.zeros((h, w))
    mask2 = np.zeros((h, w))
    mask3 = np.zeros((h, w))
    b, g, r = cv2.split(frame)
    mask1[np.where(r > b)] = 1
    mask2[np.where(r > g)] = 1
    mask3[np.where(g < b)] = 1
    res = cv2.bitwise_and(mask1, mask2, mask3)
    frame = cv2.merge([b * res, g * res, r * res]).astype(np.uint8)
    cv2.namedWindow('res', cv2.WINDOW_NORMAL)
    cv2.imshow('res', res)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    cv2.waitKey(500)
    return frame


def extract_region(frame):
    h, w, _ = frame.shape
    # frame = cv2.resize(frame, dsize=(0, 0), fx=0.1, fy=0.1)
    # frame = cv2.medianBlur(frame, 5)
    frame = cv2.bilateralFilter(frame, 101, 75, 75)
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    mask = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    cv2.imshow('mask', mask)
    cv2.waitKey(500)
    mask = cv2.resize(mask, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.merge([mask, mask, mask])


def extract_blue_region(frame):
    h, w, _ = frame.shape
    mask1 = np.zeros((h, w))
    mask2 = np.zeros((h, w))
    mask3 = np.zeros((h, w))
    b, g, r = cv2.split(frame)
    mask1[np.where(b > r)] = 1
    mask2[np.where(b > g)] = 1
    mask3[np.where(r < g)] = 1
    res = cv2.bitwise_and(mask1, mask2, mask3)
    contours, hierarchy = cv2.findContours(res.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts1 = sorted(contours, key=cv2.contourArea, reverse=True)
    frame = cv2.drawContours(frame, cnts1, 0, (255, 0, 0), -1)

    cv2.namedWindow('res', cv2.WINDOW_NORMAL)

    cv2.imshow('res', res)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    cv2.waitKey()
    return res


from shapely.geometry import Polygon, box


def is_intersect_poly_rect(poly_coords, rect_coords):
    # 将多边形的顶点列表转换为多边形对象
    poly = Polygon([(poly_coords[i], poly_coords[i + 1]) for i in range(0, len(poly_coords), 2)])
    # 将矩形框的坐标列表转换为矩形对象
    rect = box(rect_coords[0], rect_coords[1], rect_coords[2], rect_coords[3])
    # 判断多边形和矩形是否相交
    if not poly.intersects(rect):
        return False
    # 计算它们的相交面积
    intersect_area = poly.intersection(rect).area
    # 判断相交面积是否大于矩形面积的十分之一
    if intersect_area > rect.area / 2:
        return True
    else:
        return False


def get_iou(bbox1, bbox2):
    """
        Calculates the intersection-over-union of two bounding boxes.
        """
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]
    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2
    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)
    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0.0
    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection
    return size_intersection / size_union


def get_bbox_from_mask_zoom_pcb(result, ex, biaoding_box, biaoding_box_src, polygon, p=0.1):
    
    mask = cv2.normalize(result, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    mask = np.uint8(mask * 255)
    ret, mask = cv2.threshold(mask, 255 * p, 255, cv2.THRESH_BINARY)  # pcb

    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    width_scale = (biaoding_box[2] - biaoding_box[0]) / mask.shape[1]
    height_scale = (biaoding_box[3] - biaoding_box[1]) / mask.shape[0]
    S_threshold = (biaoding_box[3] - biaoding_box[1]) * (biaoding_box[2] - biaoding_box[0]) * 0.0001
    err_threshold = 0.1
    have_diff = False
    count_diff = 0
    pre_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)  # 获取轮廓顶点及边长
        start_point, end_point, bbox_width, bbox_height = (x, y), (x + w, y + h), w, h
        start_point = (int(x * width_scale + biaoding_box[0]), int(y * height_scale + biaoding_box[1]))
        end_point = (int((x + w) * width_scale + biaoding_box[0]), int((y + h) * height_scale + biaoding_box[1]))
        bbox_width = ((x + w) * width_scale + biaoding_box[0]) - (x * width_scale + biaoding_box[0])
        bbox_height = ((y + h) * height_scale + biaoding_box[1]) - (y * height_scale + biaoding_box[1])
        err = 0.01378330908618641 * max(bbox_height, bbox_width)
        S_box = bbox_height * bbox_width

        if (bbox_width / bbox_height) >= 0.1 and (
                bbox_width / bbox_height) <= 10.0 and S_box >= S_threshold and err >= err_threshold:
            x0, y0, x1, y1 = start_point[0], start_point[1], end_point[0], end_point[1]
            if is_intersect_poly_rect(polygon, [x0, y0, x1, y1]):
                pre_boxes.append([x0, y0, x1, y1])
                have_diff = True
                count_diff += 1
                color = (0, 0, 255)  # 边框颜色红
                thickness = 3  # 边框厚度1
                ex = np.ascontiguousarray(ex)
                ex = cv2.rectangle(ex, start_point, end_point, color, thickness)

    if have_diff:
        # print('有异物:', count_diff)
        pass
    else:
        # print('无异物')
        pass
    return ex, pre_boxes


def get_bbox_from_mask_connector(result, imgs, biaoding_box, hard_mask, polygon, cls, input_mask):
    # 获取mask（灰度图）
    ex, src = imgs

    mask = cv2.normalize(result, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    mask = np.uint8(mask * 255)
    ts = mask.mean()
    diff_confidence = 0.1
    diff_thresh = ts + (255 - ts) * diff_confidence
    ret, mask = cv2.threshold(mask, diff_thresh, 255, cv2.THRESH_BINARY)

    if len(hard_mask) == 2:
        mask = np.bitwise_or(mask, hard_mask[1])

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    width_scale = (biaoding_box[2] - biaoding_box[0]) / mask.shape[1]
    height_scale = (biaoding_box[3] - biaoding_box[1]) / mask.shape[0]
    S_threshold = (biaoding_box[3] - biaoding_box[1]) * (biaoding_box[2] - biaoding_box[0]) * 0.0001
    err_threshold = 0.1
    have_diff = False
    count_diff = 0
    pre_boxes = []
    show_boxes = []
    for cnt in contours:

        x, y, w, h = cv2.boundingRect(cnt)  # 获取轮廓顶点及边长
        start_point = (int(x * width_scale + biaoding_box[0]), int(y * height_scale + biaoding_box[1]))
        end_point = (int((x + w) * width_scale + biaoding_box[0]), int((y + h) * height_scale + biaoding_box[1]))
        bbox_width = ((x + w) * width_scale + biaoding_box[0]) - (x * width_scale + biaoding_box[0])
        bbox_height = ((y + h) * height_scale + biaoding_box[1]) - (y * height_scale + biaoding_box[1])
        err = 0.01378330908618641 * max(bbox_height, bbox_width)
        S_box = bbox_height * bbox_width

        if (bbox_width / bbox_height) >= 0.1 and (
                bbox_width / bbox_height) <= 10.0 and S_box >= S_threshold and err >= err_threshold:
            x0, y0, x1, y1 = start_point[0], start_point[1], end_point[0], end_point[1]
            if is_intersect_poly_rect(polygon, [x0, y0, x1, y1]):
                pre_boxes.append([x0, y0, x1, y1])
                show_boxes.append([x, y, x + w, y + h])
                have_diff = True
                count_diff += 1
                color = (0, 0, 255)  # 边框颜色红
                thickness = 2  # 边框厚度1
                ex = np.ascontiguousarray(ex)
                ex = cv2.rectangle(ex, start_point, end_point, color, thickness)
                # ex = cv2.drawContours(ex, drawcnt, -1, (0, 255, 0), 1)

    if have_diff:
        print('有异物:', count_diff)
    else:
        print('无异物')
    return ex, pre_boxes, show_boxes


def get_bbox_from_mask_zoom_connector(result, imgs, biaoding_box, hard_mask, polygon, cls, input_mask):
    # 获取mask（灰度图）
    ex, src = imgs

    # mask = cv2.normalize(result, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    # mask = np.uint8(mask * 255)
    mask = result

    ts = mask.mean()
    diff_confidence = 0.9 if 'rectangle' in cls else 0.3
    # diff_confidence = 0.99 if 'rectangle' in cls else 0.3
    diff_thresh = ts + (1.0 - ts) * diff_confidence
    ret, mask = cv2.threshold(mask, diff_thresh, 1, cv2.THRESH_BINARY)

    mask = mask.astype(np.uint8)
    # var = np.var(mask[input_mask[:, :, 1] == 1])
    # cv2.imshow('input_mask', input_mask * 255)
    # print(var)
    #
    # # 转换成二值图
    # # ts = mask.mean()
    # ts = mask[input_mask[:, :, 1] == 1].sum() / np.sum(input_mask[:, :, 1] == 1)
    # diff_confidence = 0.0
    # if cls == 'small_circle':
    #     diff_confidence = diff_confidence + 0.1 if diff_confidence + 0.1 <= 1.0 else 1.0
    #     if var <= 5.0:
    #         diff_confidence = diff_confidence + 0.1 if diff_confidence + 0.1 <= 1.0 else 1.0
    # if cls == 'normal_circle':
    #     diff_confidence = diff_confidence + 0.1 if diff_confidence + 0.1 <= 1.0 else 1.0
    #     if var <= 5.0:
    #         diff_confidence += 0.1
    # if 'rectangle' in cls:
    #     diff_confidence = diff_confidence + 0.6 if diff_confidence + 0.6 <= 1.0 else 1.0
    #     if var <= 5.0:
    #         diff_confidence += 0.1
    # diff_thresh = ts + (255 - ts) * diff_confidence
    # ret, mask = cv2.threshold(mask * input_mask[:, :, 1], diff_thresh, 255, cv2.THRESH_BINARY)  # diff

    # mask = np.uint8(result * 255)
    if len(hard_mask) == 2:
        mask = np.bitwise_or(mask, hard_mask[1])

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    width_scale = (biaoding_box[2] - biaoding_box[0]) / mask.shape[1]
    height_scale = (biaoding_box[3] - biaoding_box[1]) / mask.shape[0]
    S_threshold = (biaoding_box[3] - biaoding_box[1]) * (biaoding_box[2] - biaoding_box[0]) * 0.0001
    # S_threshold = 400 if 'rectangle' in cls else 44
    err_threshold = 0.1
    have_diff = False
    count_diff = 0
    pre_boxes = []
    show_boxes = []
    for cnt in contours:

        x, y, w, h = cv2.boundingRect(cnt)  # 获取轮廓顶点及边长
        start_point = (int(x * width_scale + biaoding_box[0]), int(y * height_scale + biaoding_box[1]))
        end_point = (int((x + w) * width_scale + biaoding_box[0]), int((y + h) * height_scale + biaoding_box[1]))
        bbox_width = ((x + w) * width_scale + biaoding_box[0]) - (x * width_scale + biaoding_box[0])
        bbox_height = ((y + h) * height_scale + biaoding_box[1]) - (y * height_scale + biaoding_box[1])
        err = 0.01378330908618641 * max(bbox_height, bbox_width)
        S_box = bbox_height * bbox_width

        if (bbox_width / bbox_height) >= 0.1 and (
                bbox_width / bbox_height) <= 10.0 and S_box >= S_threshold and err >= err_threshold:
            # if (bbox_width / bbox_height) >= 0.1 and (
            #         bbox_width / bbox_height) <= 10.0 and bbox_width * bbox_width >= 49:
            x0, y0, x1, y1 = start_point[0], start_point[1], end_point[0], end_point[1]
            if is_intersect_poly_rect(polygon, [x0, y0, x1, y1]):
                pre_boxes.append([x0, y0, x1, y1])
                show_boxes.append([x, y, x + w, y + h])
                have_diff = True
                count_diff += 1
                color = (0, 0, 255)  # 边框颜色红
                thickness = 2  # 边框厚度1
                ex = np.ascontiguousarray(ex)
                ex = cv2.rectangle(ex, start_point, end_point, color, thickness)
                # ex = cv2.drawContours(ex, drawcnt, -1, (0, 255, 0), 1)

    if have_diff:
        print('有异物:', count_diff)
    else:
        print('无异物')
    return ex, pre_boxes, show_boxes


def get_bbox_from_mask_diff(result, ex):
    # 获取mask（灰度图）
    mask = cv2.normalize(result, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    mask = np.uint8(mask * 255)

    # 转换成二值图
    ret, mask = cv2.threshold(mask, 255 * 0.1, 255, cv2.THRESH_BINARY)  # diff
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    have_diff = False
    count_diff = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)  # 获取轮廓顶点及边长
        start_point, end_point, bbox_width, bbox_height = (x, y), (x + w, y + h), w, h
        if w * h >= 40:
            have_diff = True
            count_diff += 1
            color = (0, 0, 255)  # 边框颜色红
            thickness = 2  # 边框厚度1
            ex = np.ascontiguousarray(ex)
            ex = cv2.rectangle(ex, start_point, end_point, color, thickness)
    # if have_diff:
    #     print('有异物:', count_diff)
    # else:
    #     print('无异物')
    return ex


def get_demo_input_zoom(defection_img_path, normal_img_path, biaoding_box, polygon, H, W):
    img1 = cv2.imread(defection_img_path)
    img2 = cv2.imread(normal_img_path)

    assert img1.shape == img2.shape
    hard_mask = [np.zeros_like(img1)]

    # cv2.namedWindow('diff', cv2.WINDOW_NORMAL)
    # cv2.imshow('diff', img1 - img2)
    # cv2.imwrite('chafen.png', img1 - img2)
    # cv2.waitKey()

    roi1 = img1[biaoding_box[1]:biaoding_box[3], biaoding_box[0]:biaoding_box[2], :]
    roi2 = img2[biaoding_box[1]:biaoding_box[3], biaoding_box[0]:biaoding_box[2], :]

    roi1, roi2, biaoding_box, hard_mask[0], cls = roi_dectection_by_yolo(roi1, roi2, biaoding_box, hard_mask[0])
    polygon = [biaoding_box[0], biaoding_box[1], biaoding_box[2], biaoding_box[1],
               biaoding_box[2], biaoding_box[3], biaoding_box[0], biaoding_box[3]]

    print(cls)

    biaoding_box_zoom = copy.deepcopy(biaoding_box)
    biaoding_box_center_y = (biaoding_box[3] + biaoding_box[1]) / 2.
    biaoding_box_center_x = (biaoding_box[2] + biaoding_box[0]) / 2.
    biaoding_box_width = biaoding_box[2] - biaoding_box[0]
    biaoding_box_height = biaoding_box[3] - biaoding_box[1]
    biaoding_box_scale_y = biaoding_box_height / img1.shape[0]
    biaoding_box_scale_x = biaoding_box_width / img1.shape[1]
    if cls == 'normal_circle':
        zs = 0.1
    elif cls == 'small_circle':
        zs = 0.3
    else:
        zs = 0.5
    zoom_scale_x = (1.0 / biaoding_box_scale_x) ** zs
    zoom_scale_y = (1.0 / biaoding_box_scale_y) ** zs

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

    # pad_color1 = (int(roi1[0, 0, 0]), int(roi1[0, 0, 1]), int(roi1[0, 0, 2]))
    # pad_color2 = (int(roi2[0, 0, 0]), int(roi2[0, 0, 1]), int(roi2[0, 0, 2]))
    # print(pad_color1, pad_color2)
    pad_color1, pad_color2 = (0, 0, 0), (0, 0, 0)
    roi1_pad = cv2.copyMakeBorder(roi1, int(top_size), int(bottom_size), int(left_size), int(right_size),
                                  cv2.BORDER_CONSTANT, value=pad_color1)
    roi2_pad = cv2.copyMakeBorder(roi2, int(top_size), int(bottom_size), int(left_size), int(right_size),
                                  cv2.BORDER_CONSTANT, value=pad_color2)

    roi1 = cv2.resize(roi1_pad, dsize=(W, H))  # 输入大小resize到640x480
    roi2 = cv2.resize(roi2_pad, dsize=(W, H))

    if cls == 'normal_circle' or cls == 'small_circle':

        Gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV)[:, :, 2]
        Gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)[:, :, 2]

        mask1 = cv2.threshold(Gray1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        mask2 = cv2.threshold(Gray2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # cv2.imshow('1', mask1)
        # cv2.imshow('2', mask2)
        # cv2.waitKey()

        k = np.ones((15, 15), np.uint8)
        mask_bd = mask2 - mask1
        mask_bd = cv2.erode(mask_bd, k, 20)

        number_of_white_pix = np.sum(mask_bd == 255)
        if number_of_white_pix >= 2000:
            print('big diff')
            hard_mask.append(mask_bd)

    roi1 = torch.FloatTensor(roi1).permute(2, 0, 1).unsqueeze(0)
    roi2 = torch.FloatTensor(roi2).permute(2, 0, 1).unsqueeze(0)
    data = torch.cat((roi1, roi2), dim=1)

    # if cls != 0:
    #     hard_mask = polygon2mask(polygon=polygon, mask=hard_mask)

    return data, hard_mask, [img1, img2], biaoding_box_zoom, polygon, cls


import time


def get_demo_input_zoom_connector(defection_img_path, normal_img_path, biaoding_box, polygon, H, W):
    img1 = cv2.imread(defection_img_path)
    img2 = cv2.imread(normal_img_path)

    assert img1.shape == img2.shape
    hard_mask = [np.zeros_like(img1)]

    roi1 = img1[biaoding_box[1]:biaoding_box[3], biaoding_box[0]:biaoding_box[2], :]
    roi2 = img2[biaoding_box[1]:biaoding_box[3], biaoding_box[0]:biaoding_box[2], :]

    time1 = time.time()
    roi1, roi2, biaoding_box, hard_mask[0], cls, input_mask = roi_dectection_by_yolo(roi1, roi2, biaoding_box,
                                                                                     hard_mask[0])
    time2 = time.time()

    print('use time:', time2 - time1)

    polygon = [biaoding_box[0], biaoding_box[1], biaoding_box[2], biaoding_box[1],
               biaoding_box[2], biaoding_box[3], biaoding_box[0], biaoding_box[3]]

    print(cls)

    biaoding_box_zoom = copy.deepcopy(biaoding_box)
    biaoding_box_center_y = (biaoding_box[3] + biaoding_box[1]) / 2.
    biaoding_box_center_x = (biaoding_box[2] + biaoding_box[0]) / 2.
    biaoding_box_width = biaoding_box[2] - biaoding_box[0]
    biaoding_box_height = biaoding_box[3] - biaoding_box[1]
    biaoding_box_scale_y = biaoding_box_height / img1.shape[0]
    biaoding_box_scale_x = biaoding_box_width / img1.shape[1]
    if cls == 'normal_circle':
        zs = 0.1
    elif cls == 'small_circle':
        zs = 0.3
    else:
        zs = 0.5
    zoom_scale_x = (1.0 / biaoding_box_scale_x) ** zs
    zoom_scale_y = (1.0 / biaoding_box_scale_y) ** zs

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

    roi1_pad = cv2.copyMakeBorder(roi1, int(top_size), int(bottom_size), int(left_size), int(right_size),
                                  cv2.BORDER_CONSTANT, value=0)
    roi2_pad = cv2.copyMakeBorder(roi2, int(top_size), int(bottom_size), int(left_size), int(right_size),
                                  cv2.BORDER_CONSTANT, value=0)
    input_mask_pad = cv2.copyMakeBorder(input_mask, int(top_size), int(bottom_size), int(left_size), int(right_size),
                                        cv2.BORDER_CONSTANT, value=0)

    roi1 = cv2.resize(roi1_pad, dsize=(W, H))  # 输入大小resize到640x480
    roi2 = cv2.resize(roi2_pad, dsize=(W, H))
    input_mask = cv2.resize(input_mask_pad, dsize=(W, H), interpolation=cv2.INTER_NEAREST)

    # if cls == 'normal_circle' or cls == 'small_circle':
    #
    #     Gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV)[:, :, 2]
    #     Gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)[:, :, 2]
    #
    #     mask1 = cv2.threshold(Gray1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #     mask2 = cv2.threshold(Gray2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #
    #     cv2.imshow('1', mask1)
    #     cv2.imshow('2', mask2)
    #     # cv2.waitKey()
    #
    #     k = np.ones((15, 15), np.uint8)
    #     mask_bd = mask2 - mask1
    #     mask_bd = cv2.erode(mask_bd, k, 20)
    #
    #     number_of_white_pix = np.sum(mask_bd == 255)
    #     if number_of_white_pix >= 2000:
    #         print('big diff')
    #         hard_mask.append(mask_bd)
    # roi1 = rgb_norm(roi1)
    # roi2 = rgb_norm(roi2)

    roi1 = torch.FloatTensor(roi1).permute(2, 0, 1).unsqueeze(0)
    roi2 = torch.FloatTensor(roi2).permute(2, 0, 1).unsqueeze(0)
    data = torch.cat((roi1, roi2), dim=1)

    data = rgb_norm_torch(data)
    # if cls != 0:
    #     hard_mask = polygon2mask(polygon=polygon, mask=hard_mask)

    return data, hard_mask, [img1, img2], biaoding_box_zoom, polygon, cls, input_mask


def get_demo_input_diff(defection_img_path, normal_img_path, W, H):
    img1_src = cv2.imread(defection_img_path)
    img2_src = cv2.imread(normal_img_path)

    # img1_src[img1_src == 0] = 255
    # img2_src[img2_src == 0] = 255

    img1_resize = cv2.resize(img1_src, dsize=(W, H))  # 输入大小resize到640x480
    img2_resize = cv2.resize(img2_src, dsize=(W, H))
    # img1_resize = random_jitter(img1_resize, 10,10)

    ex = img1_resize
    src = img2_resize

    # img1_resize = rgb_norm(img1_resize)
    # img2_resize = rgb_norm(img2_resize)
    # img2_resize = np.zeros_like(img1_resize)

    img1 = torch.cuda.FloatTensor(img1_resize).permute(2, 0, 1).unsqueeze(0)
    img2 = torch.cuda.FloatTensor(img2_resize).permute(2, 0, 1).unsqueeze(0)

    data = torch.cat((img1, img2), dim=1)
    data = rgb_norm_torch(data)

    return data, ex, src


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y


def getGrayDiff(img, currentPoint, tmpPoint):
    return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))


from bisect import bisect_right


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=1 / 3,
            warmup_iters=100,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        pre = torch.sigmoid(predict).view(num, -1)
        tar = target.view(num, -1)

        intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


def un_label_smooth(mask):
    un_smooth = copy.deepcopy(mask)
    un_smooth[un_smooth >= 0.5] = 1
    un_smooth[un_smooth <= 0.5] = 0
    return un_smooth


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        l = max_val - min_val
    else:
        l = val_range

    padd = window_size // 2
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    c1 = (0.01 * l) ** 2
    c2 = (0.03 * l) ** 2

    v1 = 2.0 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret, ssim_map


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average  # 对batch里面的数据取均值/求和

    def forward(self, inputs, targets):
        ce_loss = nn.BCEWithLogitsLoss()(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size).cuda()

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        s_score, ssim_map = ssim(img1, img2, window=window, window_size=self.window_size,
                                 size_average=self.size_average)
        return 1.0 - s_score


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # 计算Hinge Loss
        loss = F.relu(1 - y_true * y_pred)
        # 计算平均Hinge Loss
        mean_loss = torch.mean(loss)
        return mean_loss


class MS_SSIM_L1_LOSS(nn.Module):
    # Have to use cuda, otherwise the speed is too slow.
    def __init__(self, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
                 data_range=1.0,
                 K=(0.01, 0.03),
                 alpha=0.025,
                 compensation=200.0,
                 cuda_dev=0, ):
        super(MS_SSIM_L1_LOSS, self).__init__()
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation = compensation
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros((3 * len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
            g_masks[3 * idx + 0, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3 * idx + 1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3 * idx + 2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
        self.g_masks = g_masks.cuda(cuda_dev)

    def _fspecial_gauss_1d(self, size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)

    def forward(self, x, y):
        b, c, h, w = x.shape
        mux = F.conv2d(x, self.g_masks, groups=3, padding=self.pad)
        muy = F.conv2d(y, self.g_masks, groups=3, padding=self.pad)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=3, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=3, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=3, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l = (2 * muxy + self.C1) / (mux2 + muy2 + self.C1)  # [B, 15, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

        lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]
        PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM * PIcs  # [B, H, W]

        loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, 3, H, W]
        # average l1 loss in 3 channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-3, length=3),
                               groups=3, padding=self.pad).mean(1)  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss_mix = self.compensation * loss_mix

        return loss_mix.mean()


def random_h(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    region_mask1 = np.where(img[:, :, 0] < 175, 1, 0)
    region_mask2 = np.where(img[:, :, 0] > 5, 1, 0)
    region_mask = np.bitwise_and(region_mask1, region_mask2)
    hue_t = np.ones_like(img[:, :, 0]) * np.random.randint(-5, 6) * region_mask
    img[:, :, 0] = img[:, :, 0] + hue_t
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def random_v(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    region_mask1 = np.where(img[:, :, 2] < 250, 1, 0)
    region_mask2 = np.where(img[:, :, 2] > 5, 1, 0)
    region_mask = np.bitwise_and(region_mask1, region_mask2)
    value_t = np.ones_like(img[:, :, 2]) * np.random.randint(-5, 6) * region_mask
    img[:, :, 2] = img[:, :, 2] + value_t
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def virtual_light(img):
    # 获取图像行和列
    rows, cols = img.shape[:2]
    # 设置中心点和光照半径
    centerX = np.random.randint(50, cols - 50)
    centerY = np.random.randint(50, rows - 50)
    radius = min(centerX, centerY)
    # radius = np.random.randint(50, cols // 4)
    # 设置光照强度
    strength = 0 + np.random.randint(-20, 20)
    x = 1 if np.random.randint(0, 2) == 1 else -1
    # 新建目标图像
    distance = (centerY - np.arange(rows)[:, np.newaxis] - 0.5) ** 2 + \
               (centerX - np.arange(cols)[np.newaxis, :] - 0.5) ** 2

    # 计算结果矩阵
    result = strength * (1 - np.sqrt(distance) / radius)
    result[distance >= radius ** 2] = 0
    result = np.clip(result * x, -255, 255).astype("int32")

    # 添加结果
    dst = np.clip(img + result[..., np.newaxis], 0, 255).astype("uint8")

    return dst


def img_merge(img1, img2):
    h, w = img1.shape[0], img1.shape[1]
    img = np.zeros((h * 2, w * 2, 3))
    img[::2, ::2, :] = img1
    img[1::2, 1::2, :] = img2
    return img


# from yolov5_circle.detect import single_detect
# from yolov5_circle.segment.predict import rec_mask_detect


def color_squeeze_l_h(img, c, l, h):
    a = copy.deepcopy(img)
    if c == 'r':
        r = cv2.normalize(a[:, :, 2], None, l, h, cv2.NORM_MINMAX, cv2.CV_8U)
        g = cv2.normalize(a[:, :, 1], None, 0, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        b = cv2.normalize(a[:, :, 0], None, 0, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    elif c == 'g':
        r = cv2.normalize(a[:, :, 2], None, 0, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        g = cv2.normalize(a[:, :, 1], None, l, h, cv2.NORM_MINMAX, cv2.CV_8U)
        b = cv2.normalize(a[:, :, 0], None, 0, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    else:
        r = cv2.normalize(a[:, :, 2], None, 0, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        g = cv2.normalize(a[:, :, 1], None, 0, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        b = cv2.normalize(a[:, :, 0], None, l, h, cv2.NORM_MINMAX, cv2.CV_8U)
    a[:, :, 2] = r
    a[:, :, 1] = g
    a[:, :, 0] = b
    return a


def get_rec_mask(roi, x, y, w, h):
    extend_h, extend_w = 30, 30
    if min(x, roi.shape[1] - x - w) < extend_w:
        extend_w = min(x, roi.shape[1] - w - x)
    if min(y, roi.shape[0] - y - h) < extend_h:
        extend_h = min(y, roi.shape[0] - h - y)

    print(extend_h, extend_w)
    extend = roi[y - extend_h:y + h + extend_h, x - extend_w:x + w + extend_w, :]

    extend = cv2.GaussianBlur(extend, (5, 5), 0, 0)
    extend = color_squeeze_l_h(extend, 'r', 0, 100)

    extend = cv2.cvtColor(extend, cv2.COLOR_BGR2HSV)

    im1Gray = extend[:, :, 2]
    mask = cv2.threshold(im1Gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    number_of_white_pix = np.sum(mask == 255)
    number_of_black_pix = np.sum(mask == 0)
    if number_of_white_pix <= number_of_black_pix:
        mask = cv2.threshold(im1Gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    k = np.ones((11, 11), np.uint8)  # 创建核
    mask = cv2.erode(mask, k, 20)

    a1 = np.zeros_like(extend)
    contour1, hierarchy1 = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts1 = sorted(contour1, key=cv2.contourArea, reverse=True)
    result1 = cv2.drawContours(a1, cnts1, -1, (1, 1, 1), -1)

    if extend_w == 0 and extend_h != 0:
        return result1[extend_h:-extend_h, :, :]
    if extend_h == 0 and extend_w != 0:
        return result1[:, extend_w:-extend_w, :]
    if extend_h == 0 and extend_w == 0:
        return result1
    return result1[extend_h:-extend_h, extend_w:-extend_w, :]


def grabcut(img1Gray, x1, y1, w1, h1):
    mask = np.zeros(img1Gray.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (x1, y1, w1, h1)
    cv2.grabCut(img1Gray, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask_fg = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    mask_fg = np.expand_dims(mask_fg, axis=2)
    mask_fg = np.concatenate((mask_fg, mask_fg, mask_fg), axis=2)
    return mask_fg[y1:y1 + h1, x1:x1 + w1, :]


def choose_best_mask(mask1s, mask2s):
    max_iou = 0
    mask1_choosed, mask2_choosed = mask1s[0], mask2s[0]
    for mask1 in mask1s:
        for mask2 in mask2s:
            h1, w1, _ = mask1.shape
            h2, w2, _ = mask2.shape
            h, w = max(h1, h2), max(w1, w2)
            mask1_ = cv2.resize(mask1, dsize=(w, h))
            mask2_ = cv2.resize(mask2, dsize=(w, h))
            iou = binary_iou(mask1_[:, :, 0], mask2_[:, :, 0])
            if iou >= max_iou:
                max_iou = iou
                mask1_choosed, mask2_choosed = mask1, mask2
    return mask1_choosed, mask2_choosed


def keep_largest_connected_component(mask):
    # 进行连通组件分析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask[:, :, 0], connectivity=8)

    # 找到面积最大的连通组件
    largest_component_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

    # 创建一个新的二值掩码，将除了最大连通组件之外的所有像素设为0
    largest_component_mask = np.zeros_like(mask)
    largest_component_mask[labels == largest_component_label] = 1

    return largest_component_mask


def extract_color_region_in_hsv(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([35, 43, 0])  # green
    upper = np.array([99, 255, 255])
    mask = cv2.inRange(img_hsv, lower, upper) / 255
    mask_show = np.zeros_like(mask)

    contour1, hierarchy1 = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts1 = sorted(contour1, key=cv2.contourArea, reverse=True)
    mask_show = cv2.drawContours(mask_show, cnts1, 0, (1), -1)

    k = np.ones((51, 51), np.uint8)  # 创建核
    mask_show = cv2.morphologyEx(mask_show, cv2.MORPH_OPEN, k)

    # cv2.namedWindow('mask_show', cv2.WINDOW_NORMAL)
    # cv2.imshow('mask_show', 255*mask.astype(np.uint8))
    # cv2.namedWindow('extracted_region', cv2.WINDOW_NORMAL)
    # cv2.imshow('extracted_region', (img*mask[:, :, np.newaxis]).astype(np.uint8))
    # cv2.waitKey()

    contour2, hierarchy2 = cv2.findContours(mask_show.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    x, y, w, h = cv2.boundingRect(contour2[0])  # 获取轮廓顶点及边长

    return x, y, w, h


def roi_dectection_by_yolo(roi1, roi2, biaoding_box, hard_mask):
    sh, sw = roi1.shape[0], roi1.shape[1]

    # cls, x1, y1, w1, h1, mask1 = rec_mask_detect(roi1)
    # cls, x2, y2, w2, h2, mask2 = rec_mask_detect(roi2)

    cls, x1, y1, w1, h1 = single_detect(roi1)
    cls, x2, y2, w2, h2 = single_detect(roi2)

    print('before', cls, x2, y2, w2, h2)
    if abs(w2 - h2) / ((h2 + w2) / 2) < 0.2 and min(h2, w2) >= 400:
        x1, y1, w1, h1 = extract_color_region_in_hsv(roi1)
        x2, y2, w2, h2 = extract_color_region_in_hsv(roi2)
        cls = 0
        print('after', cls, x2, y2, w2, h2)

    # if cls == 0 or cls == 1 or cls == 4:
    if cls == 0:
        # cv2.imshow('roi', roi1)

        wm, hm = (w1 + w2) / 2, (h1 + h2) / 2
        x1c, y1c, x2c, y2c = x1 + w1 / 2, y1 + h1 / 2, x2 + w2 / 2, y2 + h2 / 2
        x1, y1, w1, h1 = int(x1c - wm / 2), int(y1c - hm / 2), int(wm), int(hm)
        x2, y2, w2, h2 = int(x2c - wm / 2), int(y2c - hm / 2), int(wm), int(hm)

        smaller = 6
        # print(min(w1, h1))
        if 120 < min(w1, h1) <= 160:
            smaller = 8
        if min(w1, h1) <= 120:
            smaller = 4.5
        if min(w1, w2) >= 400 and min(h1, h2) >= 400:
            cls = 4

        roi1 = roi1[y1:y1 + h1, x1:x1 + w1, :]
        roi2 = roi2[y2:y2 + h2, x2:x2 + w2, :]
        # cv2.imshow('roi_in_box', roi1)
        circle_mask1 = np.zeros_like(roi1)
        cv2.circle(circle_mask1, (int(w1 / 2), int(h1 / 2)), int(min(w1, h1) / 2), (1, 1, 1), -1)
        if cls != 4:
            cv2.circle(circle_mask1, (int(w1 / 2), int(h1 / 2)), int(min(w1, h1) / smaller), (0, 0, 0), -1)
        # cv2.imshow('circle_mask', circle_mask1*255)
        roi1 = roi1 * circle_mask1
        # roi1[circle_mask1 == 0] = roi1[circle_mask1 != 0].sum() / np.sum(circle_mask1 != 0)

        # cv2.imshow('masked_roi', roi1)
        circle_mask2 = np.zeros_like(roi2)
        cv2.circle(circle_mask2, (int(w2 / 2), int(h2 / 2)), int(min(w2, h2) / 2), (1, 1, 1), -1)
        if cls != 4:
            cv2.circle(circle_mask2, (int(w2 / 2), int(h2 / 2)), int(min(w2, h2) / smaller), (0, 0, 0), -1)

        roi2 = roi2 * circle_mask2
        # roi2[circle_mask2 == 0] = roi2[circle_mask1 != 0].sum() / np.sum(circle_mask2 != 0)

        biaoding_box[0] = biaoding_box[0] + x1
        biaoding_box[1] = biaoding_box[1] + y1
        biaoding_box[2] = biaoding_box[2] - (sw - w1 - x1)
        biaoding_box[3] = biaoding_box[3] - (sh - h1 - y1)

        mask_center = (int((biaoding_box[0] + biaoding_box[2]) / 2), int((biaoding_box[1] + biaoding_box[3]) / 2))
        r1 = int(min((biaoding_box[2] - biaoding_box[0], biaoding_box[3] - biaoding_box[1])) / 2)
        r2 = int(min((biaoding_box[2] - biaoding_box[0], biaoding_box[3] - biaoding_box[1])) / smaller)
        cv2.circle(hard_mask, mask_center, r1, (1, 1, 1), -1)
        if cls != 4:
            cv2.circle(hard_mask, mask_center, r2, (0, 0, 0), -1)

        circle_type = 'normal_circle'
        if smaller == 4.5:
            circle_type = 'small_circle'
        if cls == 4:
            circle_type = 'large_circle'

        return roi1, roi2, biaoding_box, hard_mask, circle_type, circle_mask1

    # elif cls == 2 or cls == 3:
    elif cls == 1:
        # print(x1, y1, w1, h1, x2, y2, w2, h2)

        x0, y0, xw, yh = min(x1, x2), min(y1, y2), max(x1 + w1, x2 + w2), max(y1 + h1, y2 + h2)
        if abs(x1 - x2) >= 50:
            x1, x2 = x0, x0
        if abs(y1 - y2) >= 50:
            y1, y2 = y0, y0
        if abs(w1 - w2) >= 50:
            w1, w2 = xw - x1, xw - x2
        if abs(h1 - h2) >= 50:
            h1, h2 = yh - y1, yh - y2

        # mask1s, mask2s = [None, None], [None, None]

        # mask1s[1] = get_rec_mask(copy.deepcopy(roi1), x1, y1, w1, h1)
        # mask2s[1] = get_rec_mask(copy.deepcopy(roi2), x2, y2, w2, h2)

        # mask1, mask2 = choose_best_mask(mask1s, mask2s)

        mask1 = get_rec_mask(copy.deepcopy(roi1), x1, y1, w1, h1)
        mask2 = get_rec_mask(copy.deepcopy(roi2), x2, y2, w2, h2)

        roi1 = roi1[y1:y1 + h1, x1:x1 + w1, :]
        roi2 = roi2[y2:y2 + h2, x2:x2 + w2, :]
        # mask1 = mask1[y1:y1 + h1, x1:x1 + w1]
        # mask2 = mask2[y2:y2 + h2, x2:x2 + w2]
        #
        # mask1 = np.expand_dims(mask1, axis=2)
        # mask1 = np.concatenate((mask1, mask1, mask1), axis=2)
        # mask2 = np.expand_dims(mask2, axis=2)
        # mask2 = np.concatenate((mask2, mask2, mask2), axis=2)

        mask1 = keep_largest_connected_component(mask1)
        mask2 = keep_largest_connected_component(mask2)

        if mask1.sum() <= 0.8 * mask2.sum():
            mask1 = cv2.resize(mask2, dsize=(mask1.shape[1], mask1.shape[0]))

        if mask2.sum() <= 0.8 * mask1.sum():
            mask2 = cv2.resize(mask1, dsize=(mask2.shape[1], mask2.shape[0]))

        # cv2.namedWindow('roimask1', cv2.WINDOW_NORMAL)
        # cv2.namedWindow('roimask2', cv2.WINDOW_NORMAL)
        # cv2.imshow('roimask1', np.hstack((roi1, mask1*255)))
        # cv2.imshow('roimask2', np.hstack((roi2, mask2*255)))
        # cv2.waitKey()

        roi1 = roi1 * mask1
        roi2 = roi2 * mask2

        biaoding_box[0] = biaoding_box[0] + x1
        biaoding_box[1] = biaoding_box[1] + y1
        biaoding_box[2] = biaoding_box[2] - (sw - w1 - x1)
        biaoding_box[3] = biaoding_box[3] - (sh - h1 - y1)

        hard_mask[biaoding_box[1]:biaoding_box[3], biaoding_box[0]:biaoding_box[2], :] = mask1

        rectangle_type = 'blue_rectangle'
        if cls == 3:
            rectangle_type = 'red_rectangle'

        return roi1, roi2, biaoding_box, hard_mask, rectangle_type, mask1.astype(np.uint8)
    else:
        return roi1, roi2, biaoding_box, hard_mask, 'None', np.ones_like((roi1.shape[0], roi1.shape[1]))


def get_M(before_points, after_points):
    pts_std = np.float32(after_points)
    points = np.float32(before_points)
    M = cv2.getPerspectiveTransform(points, pts_std)
    return M


def get_coord(pos, cvt_M):
    u = pos[0]
    v = pos[1]
    x = (cvt_M[0][0] * u + cvt_M[0][1] * v + cvt_M[0][2]) / (cvt_M[2][0] * u + cvt_M[2][1] * v + cvt_M[2][2])
    y = (cvt_M[1][0] * u + cvt_M[1][1] * v + cvt_M[1][2]) / (cvt_M[2][0] * u + cvt_M[2][1] * v + cvt_M[2][2])
    return (int(x), int(y))


def get_square_points(point1, point2):
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]
    # 逆时针旋转45度
    p3_x = (x1 - x2) * math.cos(math.radians(45)) - (y1 - y2) * math.sin(math.radians(45)) + x2
    p3_y = (x1 - x2) * math.sin(math.radians(45)) + (y1 - y2) * math.cos(math.radians(45)) + y2
    # 顺时针旋转45度
    p4_x = (x1 - x2) * math.cos(math.radians(-45)) - (y1 - y2) * math.sin(math.radians(-45)) + x2
    p4_y = (x1 - x2) * math.sin(math.radians(-45)) + (y1 - y2) * math.cos(math.radians(-45)) + y2
    return [p3_x, p3_y], [p4_x, p4_y]


def generate_transformed_mask(before_keypoints, after_keypoints, standar_mask, target_img):
    if class_type == 'blue_rectangle':
        p1, p2 = get_square_points(before_keypoints[0], before_keypoints[1])
        p3, p4 = get_square_points(before_keypoints[3], before_keypoints[2])
        before_points = [p1, p2, p3, p4]
        p1, p2 = get_square_points(after_keypoints[0], after_keypoints[1])
        p3, p4 = get_square_points(after_keypoints[3], after_keypoints[2])
        after_points = [p1, p2, p3, p4]
    M = get_M(before_keypoints, after_keypoints)
    cnts_src, hierarchy1 = cv2.findContours(standar_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts_target = []
    for each in cnts_src[0]:
        before_coord = (each[0][0], each[0][1])
        after_coord = get_coord(before_coord, M)
        cnts_target.append(after_coord)
    cnts_target = np.asarray(cnts_target, dtype=np.int32)
    hull = get_hull(cnts_target).transpose(1, 0, 2)
    bg = np.zeros_like(target_img)
    transformed_mask = cv2.fillPoly(bg.copy(), hull, color=(1, 1, 1))
    return transformed_mask

import glob
import copy
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, constant_
from util_ import conv, deconv, predict_mask, predict_image, predict_flow, upconv, NonLocalBlock, flow_warp_torch, flow_warp_torch_v2, crop_like
from torch.nn import init
from collections import OrderedDict



class Net3(nn.Module):
    def __init__(self, Norm='instance'):
        super(Net3, self).__init__()

        self.Norm = Norm
        self.conv1 = conv(self.Norm, 6, 64, kernel_size=15, stride=2)
        self.conv2 = conv(self.Norm, 64, 128, kernel_size=11, stride=2)
        self.conv3 = conv(self.Norm, 128, 256, kernel_size=7, stride=2)
        self.conv3_1 = conv(self.Norm, 256, 256)
        self.conv4 = conv(self.Norm, 256, 512, stride=2)
        self.conv4_1 = conv(self.Norm, 512, 512)
        self.conv5 = conv(self.Norm, 512, 512, stride=2)
        self.conv5_1 = conv(self.Norm, 512, 512)
        self.conv6 = conv(self.Norm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.Norm, 1024, 1024)

        self.deconv5_fst = deconv(1024, 512)
        self.deconv4_fst = deconv(1025, 256)
        self.deconv3_fst = deconv(769, 128)
        self.deconv2_fst = deconv(385, 64)

        self.predict_mask6 = predict_mask(1024)
        self.predict_mask5 = predict_mask(1025)
        self.predict_mask4 = predict_mask(769)
        self.predict_mask3 = predict_mask(385)
        self.predict_mask2 = predict_mask(193)

        self.upsampled_mask6_to_5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_mask5_to_4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_mask4_to_3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_mask3_to_2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_mask2_to_1 = nn.ConvTranspose2d(1, 1, 8, 4, 1, bias=False)

        self.deconv5_sec = deconv(1024, 512)
        self.deconv4_sec = deconv(1027, 256)
        self.deconv3_sec = deconv(771, 128)
        self.deconv2_sec = deconv(387, 64)

        self.predict_image6 = predict_image(1024)
        self.predict_image5 = predict_image(1027)
        self.predict_image4 = predict_image(771)
        self.predict_image3 = predict_image(387)
        self.predict_image2 = predict_image(195)

        self.upsampled_image6_to_5 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False)
        self.upsampled_image5_to_4 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False)
        self.upsampled_image4_to_3 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False)
        self.upsampled_image3_to_2 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False)
        self.upsampled_image2_to_1 = nn.ConvTranspose2d(3, 3, 8, 4, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):

        x_fst = torch.cat((x[:, :3, :, :], x[:, :3, :, :] - x[:, 3:, :, :]), dim=1)

        out_conv2 = self.conv2(self.conv1(x_fst))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        image6 = self.predict_image6(out_conv6)
        image6_up = crop_like(self.upsampled_image6_to_5(image6), out_conv5)
        out_deconv5 = crop_like(self.deconv5_sec(out_conv6), out_conv5)

        concat5 = torch.cat((out_conv5, out_deconv5, image6_up), 1)
        image5 = self.predict_image5(concat5)
        image5_up = crop_like(self.upsampled_image5_to_4(image5), out_conv4)
        out_deconv4 = crop_like(self.deconv4_sec(concat5), out_conv4)

        concat4 = torch.cat((out_conv4, out_deconv4, image5_up), 1)
        image4 = self.predict_image4(concat4)
        image4_up = crop_like(self.upsampled_image4_to_3(image4), out_conv3)
        out_deconv3 = crop_like(self.deconv3_sec(concat4), out_conv3)

        concat3 = torch.cat((out_conv3, out_deconv3, image4_up), 1)
        image3 = self.predict_image3(concat3)
        image3_up = crop_like(self.upsampled_image3_to_2(image3), out_conv2)
        out_deconv2 = crop_like(self.deconv2_sec(concat3), out_conv2)

        concat2 = torch.cat((out_conv2, out_deconv2, image3_up), 1)
        image2 = self.predict_image2(concat2)
        image2_up = crop_like(self.upsampled_image2_to_1(image2), x)

        x_sec = torch.cat((x[:, :3, :, :], image2_up), dim=1)
        out_conv2 = self.conv2(self.conv1(x_sec))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        mask6 = self.predict_mask6(out_conv6)
        mask6_up = crop_like(self.upsampled_mask6_to_5(mask6), out_conv5)
        out_deconv5 = crop_like(self.deconv5_fst(out_conv6), out_conv5)

        concat5 = torch.cat((out_conv5, out_deconv5, mask6_up), 1)
        mask5 = self.predict_mask5(concat5)
        mask5_up = crop_like(self.upsampled_mask5_to_4(mask5), out_conv4)
        out_deconv4 = crop_like(self.deconv4_fst(concat5), out_conv4)

        concat4 = torch.cat((out_conv4, out_deconv4, mask5_up), 1)
        mask4 = self.predict_mask4(concat4)
        mask4_up = crop_like(self.upsampled_mask4_to_3(mask4), out_conv3)
        out_deconv3 = crop_like(self.deconv3_fst(concat4), out_conv3)

        concat3 = torch.cat((out_conv3, out_deconv3, mask4_up), 1)
        mask3 = self.predict_mask3(concat3)
        mask3_up = crop_like(self.upsampled_mask3_to_2(mask3), out_conv2)
        out_deconv2 = crop_like(self.deconv2_fst(concat3), out_conv2)

        concat2 = torch.cat((out_conv2, out_deconv2, mask3_up), 1)
        mask2 = self.predict_mask2(concat2)
        mask2_up = crop_like(self.upsampled_mask2_to_1(mask2), x)

        if self.training:
            return [image2_up, mask2_up]
        else:
            return [image2_up, nn.Sigmoid()(mask2_up)]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def other_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' not in name and 'weight' not in name]

class SharedEncoder(nn.Module):
    def __init__(self, Norm='instance'):
        super(SharedEncoder, self).__init__()

        self.Norm = Norm
        self.conv1 = conv(self.Norm, 6, 64, kernel_size=15, stride=2)
        self.conv2 = conv(self.Norm, 64, 128, kernel_size=11, stride=2)
        self.conv3 = conv(self.Norm, 128, 256, kernel_size=7, stride=2)
        self.conv3_1 = conv(self.Norm, 256, 256)
        self.conv4 = conv(self.Norm, 256, 512, stride=2)
        self.conv4_1 = conv(self.Norm, 512, 512)
        self.conv5 = conv(self.Norm, 512, 512, stride=2)
        self.conv5_1 = conv(self.Norm, 512, 512)
        self.conv6 = conv(self.Norm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.Norm, 1024, 1024)

    def forward(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        return {"out_conv2": out_conv2,
                "out_conv3": out_conv3,
                "out_conv4": out_conv4,
                "out_conv5": out_conv5,
                "out_conv6": out_conv6}


class MaskDecoder(nn.Module):
    def __init__(self):
        super(MaskDecoder, self).__init__()

        self.deconv5 = upconv(1024, 512, act=True)
        self.deconv4 = upconv(1025, 256, act=True)
        self.deconv3 = upconv(769, 128, act=True)
        self.deconv2 = upconv(385, 64, act=True)

        self.predict_mask6 = predict_mask(1024)
        self.predict_mask5 = predict_mask(1025)
        self.predict_mask4 = predict_mask(769)
        self.predict_mask3 = predict_mask(385)
        self.predict_mask2 = predict_mask(193)

        self.upsampled_mask6_to_5 = upconv(1, 1)
        self.upsampled_mask5_to_4 = upconv(1, 1)
        self.upsampled_mask4_to_3 = upconv(1, 1)
        self.upsampled_mask3_to_2 = upconv(1, 1)
        self.upsampled_mask2_to_1 = upconv(1, 1, 4)

    def forward(self, x):
        out_conv2, out_conv3, out_conv4, out_conv5, out_conv6 = x["out_conv2"], x["out_conv3"], x["out_conv4"], x[
            "out_conv5"], x["out_conv6"]

        mask6 = self.predict_mask6(out_conv6)
        mask6_up = self.upsampled_mask6_to_5(mask6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5, out_deconv5, mask6_up), 1)
        mask5 = self.predict_mask5(concat5)
        mask5_up = self.upsampled_mask5_to_4(mask5)
        out_deconv4 = self.deconv4(concat5)

        concat4 = torch.cat((out_conv4, out_deconv4, mask5_up), 1)
        mask4 = self.predict_mask4(concat4)
        mask4_up = self.upsampled_mask4_to_3(mask4)
        out_deconv3 = self.deconv3(concat4)

        concat3 = torch.cat((out_conv3, out_deconv3, mask4_up), 1)
        mask3 = self.predict_mask3(concat3)
        mask3_up = self.upsampled_mask3_to_2(mask3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2, out_deconv2, mask3_up), 1)
        mask2 = self.predict_mask2(concat2)
        mask2_up = self.upsampled_mask2_to_1(mask2)

        return mask2_up


class ImageDecoder(nn.Module):
    def __init__(self):
        super(ImageDecoder, self).__init__()

        self.deconv5 = upconv(1024, 512, act=True)
        self.deconv4 = upconv(1027, 256, act=True)
        self.deconv3 = upconv(771, 128, act=True)
        self.deconv2 = upconv(387, 64, act=True)

        self.predict_image6 = predict_image(1024)
        self.predict_image5 = predict_image(1027)
        self.predict_image4 = predict_image(771)
        self.predict_image3 = predict_image(387)
        self.predict_image2 = predict_image(195)

        self.upsampled_image6_to_5 = upconv(3, 3)
        self.upsampled_image5_to_4 = upconv(3, 3)
        self.upsampled_image4_to_3 = upconv(3, 3)
        self.upsampled_image3_to_2 = upconv(3, 3)
        self.upsampled_image2_to_1 = upconv(3, 3, 4)

    def forward(self, x):
        out_conv2, out_conv3, out_conv4, out_conv5, out_conv6 = x["out_conv2"], x["out_conv3"], x["out_conv4"], x[
            "out_conv5"], x["out_conv6"]

        image6 = self.predict_image6(out_conv6)
        image6_up = self.upsampled_image6_to_5(image6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5, out_deconv5, image6_up), 1)
        image5 = self.predict_image5(concat5)
        image5_up = self.upsampled_image5_to_4(image5)
        out_deconv4 = self.deconv4(concat5)

        concat4 = torch.cat((out_conv4, out_deconv4, image5_up), 1)
        image4 = self.predict_image4(concat4)
        image4_up = self.upsampled_image4_to_3(image4)
        out_deconv3 = self.deconv3(concat4)

        concat3 = torch.cat((out_conv3, out_deconv3, image4_up), 1)
        image3 = self.predict_image3(concat3)
        image3_up = self.upsampled_image3_to_2(image3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2, out_deconv2, image3_up), 1)
        image2 = self.predict_image2(concat2)
        image2_up = self.upsampled_image2_to_1(image2)

        return image2_up


class FlowDecoder(nn.Module):
    def __init__(self):
        super(FlowDecoder, self).__init__()

        self.deconv5 = upconv(1024, 512, act=True)
        self.deconv4 = upconv(1026, 256, act=True)
        self.deconv3 = upconv(770, 128, act=True)
        self.deconv2 = upconv(386, 64, act=True)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = upconv(2, 2)
        self.upsampled_flow5_to_4 = upconv(2, 2)
        self.upsampled_flow4_to_3 = upconv(2, 2)
        self.upsampled_flow3_to_2 = upconv(2, 2)
        self.upsampled_flow2_to_1 = upconv(2, 2, 4)

    def forward(self, x):
        out_conv2, out_conv3, out_conv4, out_conv5, out_conv6 = x["out_conv2"], x["out_conv3"], x["out_conv4"], x[
            "out_conv5"], x["out_conv6"]

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)
        flow2_up = self.upsampled_flow2_to_1(flow2)

        return flow2_up


class Net5(nn.Module):
    def __init__(self, Norm='instance', mode_type="rst_flow"):
        super(Net5, self).__init__()

        self.mode_type = mode_type
        self.Norm = Norm
        self.encoder = SharedEncoder(self.Norm)
        self.mask_decoder = MaskDecoder()
        if "rst" in mode_type:
            self.image_decoder = ImageDecoder()
        if "flow" in mode_type:
            self.flow_decoder = FlowDecoder()

    def forward(self, x):
        f1, f2 = x[:, 0:3, :, :], x[:, 3:6, :, :]

        if "flow" in self.mode_type:
            # flow estimation
            x1 = torch.cat((f1, f2), dim=1)
            x1 = self.encoder(x1)
            flow = self.flow_decoder(x1)
        else:
            flow = None

        if "rst" in self.mode_type:
            # image reconstruction
            warped_f2 = flow_warp_torch(f2, -flow) if flow is not None else f2
            x2 = torch.cat((f1, f1 - warped_f2), dim=1)
            x2 = self.encoder(x2)
            image = self.image_decoder(x2)
        else:
            if "warp" in self.mode_type:
                image = flow_warp_torch(f2, -flow) if flow is not None else f2
            else:
                image = f2

        # mask prediction
        x3 = torch.cat((f1, image), dim=1)
        x3 = self.encoder(x3)
        mask = self.mask_decoder(x3)
        
        if not self.training:
            mask = nn.Sigmoid()(mask)

        return {"mask": mask, "image": image, "flow": flow}


if __name__ == "__main__":
    net = Net5()
    x = torch.randn(1, 6, 640, 640)
    out = net(x)
    print(out["mask"].size())
    print(out["image"].size())
    # print(out["flow"].size())

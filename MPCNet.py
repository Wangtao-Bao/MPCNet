import torch
import torch.nn as nn
from thop import profile
import torch.nn.functional as F


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class activation(nn.ReLU):
    def __init__(self, dim, act_num=3, deploy=False):
        super(activation, self).__init__()
        self.deploy = deploy
        self.weight = torch.nn.Parameter(torch.randn(dim, 1, act_num * 2 + 1, act_num * 2 + 1))
        self.bias = None
        self.bn = nn.BatchNorm2d(dim, eps=1e-6)
        self.dim = dim
        self.act_num = act_num

    def forward(self, x):
        if self.deploy:
            return torch.nn.functional.conv2d(
                super(activation, self).forward(x),
                self.weight, self.bias, padding=(self.act_num * 2 + 1) // 2, groups=self.dim)
        else:
            return self.bn(torch.nn.functional.conv2d(
                super(activation, self).forward(x),
                self.weight, padding=(self.act_num * 2 + 1) // 2, groups=self.dim))

    def _fuse_bn_tensor(self, weight, bn):
        kernel = weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (0 - running_mean) * gamma / std

    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.weight, self.bn)
        self.weight.data = kernel
        self.bias = torch.nn.Parameter(torch.zeros(self.dim))
        self.bias.data = bias
        self.__delattr__('bn')
        self.deploy = True

#Downsampling
class FPD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_fusion = nn.Conv2d(in_channels * 5, out_channels, kernel_size=1, stride=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self, x):
        x0 = x[:, :, 0::2, 0::2]  # x = [B, C, H/2, W/2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3, self.maxpool(x)], dim=1)  # x = [B, 4*C, H/2, W/2]
        x = self.act(self.batch_norm(self.conv_fusion(x)))
        return x

#Conv+BN+ReLU+Downsampling

class CBRD(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = activation(c2, act_num=3)
        self.downsample = FPD(c2, c2)
        # self.act = self.default_act if ahaoct is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.downsample(self.act(self.bn(self.conv(x))))

    def forward_fuse(self, x):
        return self.downsample(self.act((self.conv(x))))

#Conv+BN+ReLU
class CBR(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = activation(c2, act_num=3)
        # self.act = self.default_act if ahaoct is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttiton(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAttiton, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class MPM(nn.Module):
    def __init__(self, in_channel):
        super(MPM, self).__init__()

        self.Conv_3 = nn.Sequential(
            nn.Conv2d(in_channel // 4, in_channel // 4, 3, 1, padding=1),
            nn.BatchNorm2d(in_channel // 4),
            nn.ReLU()
        )

        self.Conv_5 = nn.Sequential(
            nn.Conv2d(in_channel // 4, in_channel // 4, 5, 1, padding=2),
            nn.BatchNorm2d(in_channel // 4),
            nn.ReLU()
        )

        self.Conv_7 = nn.Sequential(
            nn.Conv2d(in_channel // 4, in_channel // 4, 7, 1, padding=3),
            nn.BatchNorm2d(in_channel // 4),
            nn.ReLU()
        )


        self.Conv = CBR(in_channel * 2, in_channel, 1)                              
        self.ca = CoordAttiton(in_channel,in_channel)

    def forward(self, x):
        b, c, h, w = x.size()
        x_1 = x[:, :(c // 4), :, :]
        x_2 = x[:, (c // 4):(c // 4) * 2, :, :]
        x_3 = x[:, (c // 4) * 2:(c // 4) * 3, :, :]
        x_4 = x[:, (c // 4) * 3:, :, :]

        x_4_7 = self.Conv_7(x_4)
        x_3_5 = self.Conv_5(x_3)
        x_2_3 = self.Conv_3(x_2)
        x_1_1 = self.Conv_7(x_1)

        out = self.ca(self.Conv(torch.cat((x_1_1, x_2_3, x_3_5, x_4_7, x), 1)))

        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_source = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return self.sigmoid(x) * x_source

class SCFM(nn.Module):
    def __init__(self, low_channels, high_channels, out_channels, kernel_size):
        super(SCFM, self).__init__()

        self.low_saliency_block = nn.Sequential(
            CBR(low_channels, low_channels // 16, kernel_size),
            nn.Conv2d(low_channels // 16, 1, 1, padding=0),
            nn.Sigmoid()
        )

        self.high_saliency_block = nn.Sequential(
            CBR(high_channels, high_channels // 16, kernel_size),
            nn.Conv2d(high_channels // 16, 1, 1, padding=0),
            nn.Sigmoid()
        )

        self.low_proj = CBR(low_channels, out_channels, 1)
        self.high_proj = CBR(high_channels, out_channels, 1)
        self.mix_proj = CBR(low_channels + high_channels, out_channels, 1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.low_spatial_att = SpatialAttention()
        self.high_spatial_att = SpatialAttention()
        self.coord_att = CoordAttiton(out_channels, out_channels)
        self.fusion_conv = CBR(out_channels * 2, out_channels, 1)

    def forward(self, low_feat, high_feat):
        b1, c1, h1, w1 = low_feat.size()
        b2, c2, h2, w2 = high_feat.size()
        if (h1, w1) != (h2, w2):
            high_feat = self.upsample(high_feat)
        low_identity = low_feat
        high_identity = high_feat
        low_att_feat = self.low_spatial_att(low_feat)
        high_att_feat = self.high_spatial_att(high_feat)
        low_saliency = self.low_saliency_block(low_att_feat)
        high_saliency = self.high_saliency_block(high_att_feat)
        mixed_feat = torch.cat(
            [
                low_identity * high_saliency,
                high_identity * low_saliency
            ],
            dim=1
        )

        cross_att = torch.sigmoid(self.coord_att(self.mix_proj(mixed_feat)))

        low_out = cross_att * self.low_proj(low_identity + low_att_feat)
        high_out = cross_att * self.high_proj(high_identity + high_att_feat)

        out = self.fusion_conv(torch.cat([low_out, high_out], dim=1))
        return out


class GSFM(nn.Module):
    def __init__(self, low_C, high_C, out_C, size):
        super(GSFM, self).__init__()
        self.size = size // 8

        self.LOW_K = CBR(high_C, out_C, 3, 1, 1)
        self.LOW_V = CBR(high_C, out_C, 3, 1, 1)
        self.LOW_Q = CBR(high_C, out_C, 3, 1, 1)

        self.HIGH_K = CBR(high_C, out_C, 3, 1, 1)
        self.HIGH_V = CBR(high_C, out_C, 3, 1, 1)
        self.HIGH_Q = CBR(high_C, out_C, 3, 1, 1)

        self.REDUCE = CBR(out_C * 4, out_C, 3, 1, 1)

        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))
        self.gamma4 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.conv1 = nn.Conv2d(low_C, high_C, 1, 1)

    def forward(self, low_feat, high_feat):
        m_batchsize, c, h, w = high_feat.shape

        low_feat = F.interpolate(
            low_feat, size=(h, w), mode='bilinear', align_corners=True
        )
        low_feat = self.conv1(low_feat)

        low_re = F.interpolate(
            low_feat, size=(self.size, self.size), mode='bilinear', align_corners=True
        )
        high_re = F.interpolate(
            high_feat, size=(self.size, self.size), mode='bilinear', align_corners=True
        )

        LOW_K = self.LOW_K(low_re)
        LOW_V = self.LOW_V(low_re)
        LOW_Q = self.LOW_Q(low_re)

        HIGH_K = self.HIGH_K(high_re)
        HIGH_V = self.HIGH_V(high_re)
        HIGH_Q = self.HIGH_Q(high_re)

        DUAL_V = LOW_V + HIGH_V

        LOW_V = LOW_V.view(m_batchsize, -1, self.size * self.size)
        LOW_K = LOW_K.view(m_batchsize, -1, self.size * self.size).permute(0, 2, 1)
        LOW_Q = LOW_Q.view(m_batchsize, -1, self.size * self.size)

        HIGH_V = HIGH_V.view(m_batchsize, -1, self.size * self.size)
        HIGH_K = HIGH_K.view(m_batchsize, -1, self.size * self.size).permute(0, 2, 1)
        HIGH_Q = HIGH_Q.view(m_batchsize, -1, self.size * self.size)

        DUAL_V = DUAL_V.view(m_batchsize, -1, self.size * self.size)

        LOW_mask = torch.bmm(LOW_K, LOW_Q)
        LOW_mask = self.softmax(LOW_mask)
        LOW_refine = torch.bmm(DUAL_V, LOW_mask.permute(0, 2, 1))
        LOW_refine = LOW_refine.view(m_batchsize, -1, self.size, self.size)
        LOW_refine = self.gamma1 * LOW_refine
        LOW_refine = F.interpolate(
            LOW_refine, size=(h, w), mode='bilinear', align_corners=True
        ) + low_feat

        HIGH_mask = torch.bmm(HIGH_K, HIGH_Q)
        HIGH_mask = self.softmax(HIGH_mask)
        HIGH_refine = torch.bmm(DUAL_V, HIGH_mask.permute(0, 2, 1))
        HIGH_refine = HIGH_refine.view(m_batchsize, -1, self.size, self.size)
        HIGH_refine = self.gamma2 * HIGH_refine
        HIGH_refine = F.interpolate(
            HIGH_refine, size=(h, w), mode='bilinear', align_corners=True
        ) + high_feat

        LOW_HIGH_mask = torch.bmm(LOW_K, HIGH_Q)
        LOW_HIGH_mask = self.softmax(LOW_HIGH_mask)
        LOW_HIGH_refine = torch.bmm(LOW_V, LOW_HIGH_mask.permute(0, 2, 1))
        LOW_HIGH_refine = LOW_HIGH_refine.view(m_batchsize, -1, self.size, self.size)
        LOW_HIGH_refine = self.gamma3 * LOW_HIGH_refine
        LOW_HIGH_refine = F.interpolate(
            LOW_HIGH_refine, size=(h, w), mode='bilinear', align_corners=True
        ) + high_feat

        HIGH_LOW_mask = torch.bmm(HIGH_K, LOW_Q)
        HIGH_LOW_mask = self.softmax(HIGH_LOW_mask)
        HIGH_LOW_refine = torch.bmm(HIGH_V, HIGH_LOW_mask.permute(0, 2, 1))
        HIGH_LOW_refine = HIGH_LOW_refine.view(m_batchsize, -1, self.size, self.size)
        HIGH_LOW_refine = self.gamma4 * HIGH_LOW_refine + low_re
        HIGH_LOW_refine = F.interpolate(
            HIGH_LOW_refine, size=(h, w), mode='bilinear', align_corners=True
        ) + low_feat

        GLOBAL_ATT = self.REDUCE(
            torch.cat(
                (LOW_refine, HIGH_refine, LOW_HIGH_refine, HIGH_LOW_refine), dim=1
            )
        )

        return GLOBAL_ATT


class MPCNet(nn.Module):
    def __init__(self, Train=False,size=256):
        super(MPCNet, self).__init__()
        self.Train = Train
        self.Backbone_1 = CBR(1,32,3)
        self.Backbone_2 = CBRD(32, 64, 3)
        self.Backbone_3 = CBRD(64, 128, 3)

        self.MPM_1 = MPM(32)
        self.MPM_2 = MPM(64)
        self.MPM_3 = MPM(128)

        self.GSFM_1 = GSFM(32,64,64,size)
        self.GSFM_2 = GSFM(64,128,128,size)

        self.SCFM_3 = SCFM(128,128,64,1)
        self.SCFM_2 = SCFM(64,64,32,1)
        self.SCFM_1 = SCFM(32,32,32,3)

        self.final_1 = nn.Sequential(
            CBR(32, 1,1))
            #nn.Conv2d(32, 1, 1, 1))

        self.final_2 = nn.Sequential(
            CBR(32, 1, 1))
            #nn.Conv2d(32, 1, 1, 1))

        self.final_3 = nn.Sequential(
            CBR(64, 1, 1))
            #nn.Conv2d(64, 1, 1, 1))


    def forward(self, x):
        x_e1 = self.Backbone_1(x)
        MSF_1 = self.MPM_1(x_e1)

        x_e2 = self.Backbone_2(MSF_1)
        mix_1 = self.GSFM_1(x_e1,x_e2)
        MSF_2 = self.MPM_2(mix_1)

        x_e3 = self.Backbone_3(MSF_2)
        mix_2 = self.GSFM_2(x_e2, x_e3)
        MSF_3 = self.MPM_3(mix_2)

        x_d3 = self.SCFM_3(mix_2,MSF_3)
        x_d2 = self.SCFM_2(MSF_2,x_d3)
        x_d1 = self.SCFM_1(MSF_1, x_d2)

        output1 = self.final_1(x_d1)
        output2 = self.final_2(x_d2)
        output3 = self.final_3(x_d3)

        output2 = F.interpolate(output2, scale_factor=2, mode='bilinear', align_corners=True)
        output3 = F.interpolate(output3, scale_factor=4, mode='bilinear', align_corners=True)
        if self.Train:
            return [torch.sigmoid(output1), torch.sigmoid(output2), torch.sigmoid(output3)]
        else:
            return torch.sigmoid(output1)



if __name__ == '__main__':

    model = MPCNet(Train=False)
    x = torch.randn(1, 1, 256, 256)
    output = model(x)
    flops, params = profile(model, (x,))

    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')

    if len(output)>1:
        print("Output shape:", output[0].shape, output[1].shape, output[2].shape)
    else:
        print("Output shape:", output.shape)

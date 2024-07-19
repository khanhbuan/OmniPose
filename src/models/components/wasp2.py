import torch
import torch.nn as nn
import torch.nn.functional as F

class SepConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, bias=True, padding_mode='zeros', depth_multiplier=1):
        super(SepConv2d, self).__init__()

        intermediate_channels = in_channels * depth_multiplier

        self.spatialConv = nn.Conv2d(in_channels=in_channels, out_channels=intermediate_channels,
                                     kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, 
                                     groups=in_channels, bias=bias, padding_mode=padding_mode)

        self.pointConv = nn.Conv2d(in_channels=intermediate_channels, out_channels=out_channels,
                                   kernel_size=1, stride=1, padding=0, dilation=1, 
                                   bias=bias, padding_mode=padding_mode)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.spatialConv(x)
        x = self.relu(x)
        x = self.pointConv(x)

        return x

class _AtrousModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_AtrousModule, self).__init__()
        self.atrous_conv = SepConv2d(in_channels=inplanes, out_channels=planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, 
                                     bias=False, padding_mode='zeros')
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)
    
class WASPv2(nn.Module):
    def __init__(self, inplanes, planes, n_classes = 14):
        super(WASPv2, self).__init__()

        dilations = [1, 6, 12, 18]
        reduction = planes // 8

        self.aspp1 = _AtrousModule(inplanes=inplanes, planes=planes, kernel_size=1, padding=0, dilation=dilations[0])
        self.aspp2 = _AtrousModule(inplanes=planes, planes=planes, kernel_size=3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _AtrousModule(inplanes=planes, planes=planes, kernel_size=3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _AtrousModule(inplanes=planes, planes=planes, kernel_size=3, padding=dilations[3], dilation=dilations[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                             nn.Conv2d(in_channels=planes, out_channels=planes, 
                                                       kernel_size=1, stride=1, bias=False),
                                             nn.BatchNorm2d(num_features=planes),
                                             nn.ReLU())
        
        self.conv1 = nn.Conv2d(in_channels=5*planes, out_channels=planes, 
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=planes)

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=reduction, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=reduction)

        self.last_conv = nn.Sequential(nn.Conv2d(in_channels=planes+reduction, out_channels=planes, kernel_size=3, 
                                                 stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(num_features=planes),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, 
                                                 stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(num_features=planes),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=planes, out_channels=n_classes, kernel_size=1, stride=1))
        
    def forward(self, x, low_level_features):
        # x.shape = (48, 96, 96)
        # low_level_features = (256, 96, 96)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x1)
        x3 = self.aspp3(x2)
        x4 = self.aspp4(x3)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        return x

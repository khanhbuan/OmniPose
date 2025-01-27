import torch
import math
import torch.nn as nn
from src.models.components.wasp2 import WASPv2

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, 
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        
        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=planes * self.expansion, 
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=0.1)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()

        self.num_inchannels = num_inchannels
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.num_inchannels[branch_index], 
                          out_channels=num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=0.1)
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        
        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))

        return nn.Sequential(*layers)
    
    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i and j - i == 1:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(in_channels=num_inchannels[j], out_channels=num_inchannels[i], 
                                  kernel_size=1, stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(num_inchannels[i]),
                        nn.ConvTranspose2d(in_channels=num_inchannels[i], out_channels=num_inchannels[i],
                                           kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                        nn.BatchNorm2d(num_inchannels[i], momentum=0.1),
                        nn.ReLU(inplace=True),
                        self.gaussian_filter(num_inchannels[i], 3, 3)))
                
                elif j > i and j - i == 2:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(in_channels=num_inchannels[j], out_channels=num_inchannels[i], 
                                  kernel_size=1, stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(num_features=num_inchannels[i]),
                        
                        nn.ConvTranspose2d(in_channels=num_inchannels[i], out_channels=num_inchannels[i],
                                           kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                        nn.BatchNorm2d(num_features=num_inchannels[i], momentum=0.1),
                        nn.ReLU(inplace=True),
                        
                        nn.ConvTranspose2d(in_channels=num_inchannels[i], out_channels=num_inchannels[i],
                                           kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                        nn.BatchNorm2d(num_features=num_inchannels[i], momentum=0.1),
                        nn.ReLU(inplace=True),
                        
                        self.gaussian_filter(num_inchannels[i], 3, 3)))
                
                elif j > i and j - i == 3:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(in_channels=num_inchannels[j], out_channels=num_inchannels[i],
                                  kernel_size=1, stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(num_features=num_inchannels[i]),
                        
                        nn.ConvTranspose2d(in_channels=num_inchannels[i], out_channels=num_inchannels[i],
                                           kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                        nn.BatchNorm2d(num_features=num_inchannels[i], momentum=0.1),
                        nn.ReLU(inplace=True),
                        
                        nn.ConvTranspose2d(in_channels=num_inchannels[i], out_channels=num_inchannels[i],
                                           kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                        nn.BatchNorm2d(num_features=num_inchannels[i], momentum=0.1),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(in_channels=num_inchannels[i], out_channels=num_inchannels[i],
                                           kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                        nn.BatchNorm2d(num_features=num_inchannels[i], momentum=0.1),
                        nn.ReLU(inplace=True),
                        
                        self.gaussian_filter(num_inchannels[i], 3, 3)))
                
                elif j == i:
                    fuse_layer.append(None)
                
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(in_channels=num_inchannels[j], out_channels=num_outchannels_conv3x3, 
                                              kernel_size=3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(num_features=num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        in_channels=num_inchannels[j], out_channels=num_outchannels_conv3x3,
                                        kernel_size=3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def gaussian_filter(self, channels, kernel_size, sigma):
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
        mean = (kernel_size - 1)/2

        gaussian_kernel = (1./(2.* math.pi * sigma**2)) * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2*sigma**2))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_fltr = nn.Conv2d(in_channels=channels, out_channels=channels,
                                  kernel_size=kernel_size, padding=int(kernel_size//2), groups=channels, bias=False)

        gaussian_fltr.weight.data = gaussian_kernel
        gaussian_fltr.weight.requires_grad = False

        return gaussian_fltr
    
    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse

class OmniPose(nn.Module):
    def __init__(self):
        self.inplanes = 64
        super(OmniPose, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, 
                               stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64, momentum=0.1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, 
                               stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=64, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, planes=64, blocks=4)

        num_channels = [48, 96]
        block = BasicBlock
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(num_modules=1, num_branches=2, num_blocks=[4, 4], 
                                                           num_channels=[48, 96], block=BasicBlock,num_inchannels=num_channels)
        
        num_channels = [48, 96, 192]
        block = BasicBlock
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(num_modules=4, num_branches=3, num_blocks=[4, 4, 4], 
                                                           num_channels=[48, 96, 192], block=BasicBlock, num_inchannels=num_channels)

        num_channels = [48, 96, 192, 384]
        block = BasicBlock
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(num_modules=3, num_branches=4, num_blocks=[4, 4, 4, 4], 
                                                           num_channels=[48, 96, 192, 384], block=BasicBlock, num_inchannels=num_channels)

        self.waspv2 = WASPv2(48, 48, 14)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.inplanes, out_channels=planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.1),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(in_channels=num_channels_pre_layer[i], out_channels=num_channels_cur_layer[i],
                                  kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(in_channels=inchannels, out_channels=outchannels, 
                                  kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)
                    ))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, num_modules, num_branches, num_blocks, num_channels, block, num_inchannels, multi_scale_output=True):
        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(HighResolutionModule(num_branches, block, num_blocks, num_inchannels, 
                                                num_channels, reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)

        low_level_feat = x

        x_list = []
        for i in range(2):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(3):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(4):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        x = self.waspv2(y_list[0], low_level_feat)
        
        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    model = OmniPose()
    model.eval()
    x = torch.rand(1, 3, 256, 256)
    y = model(x)
    print(y.shape)
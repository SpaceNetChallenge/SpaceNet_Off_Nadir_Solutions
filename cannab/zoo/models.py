import torch
from torch import nn
import torch.nn.functional as F

from .senet import se_resnext50_32x4d, senet154, se_resnext101_32x4d
from .dpn import dpn92

class ConvReluBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvReluBN, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)

class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvRelu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)

class SCSEModule(nn.Module):
    # according to https://arxiv.org/pdf/1808.08127.pdf concat is better
    def __init__(self, channels, reduction=16, concat=False):
        super(SCSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

        self.spatial_se = nn.Sequential(nn.Conv2d(channels, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())
        self.concat = concat

    def forward(self, x):
        module_input = x

        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        chn_se = self.sigmoid(x)
        chn_se = chn_se * module_input

        spa_se = self.spatial_se(module_input)
        spa_se = module_input * spa_se
        if self.concat:
            return torch.cat([chn_se, spa_se], dim=1)
        else:
            return chn_se + spa_se


class SeResNext50_9ch_Unet(nn.Module):
    def __init__(self, pretrained='imagenet', **kwargs):
        super(SeResNext50_9ch_Unet, self).__init__()
        
        encoder_filters = [64, 256, 512, 1024, 2048]
        decoder_filters = [64, 96, 128, 256, 512]

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = nn.Sequential(ConvRelu(decoder_filters[-1]+encoder_filters[-2] + 2 + 27 + 2, decoder_filters[-1]), SCSEModule(decoder_filters[-1], reduction=16, concat=True))
        self.conv7 = ConvRelu(decoder_filters[-1] * 2, decoder_filters[-2])
        self.conv7_2 = nn.Sequential(ConvRelu(decoder_filters[-2]+encoder_filters[-3] + 2 + 27 + 2, decoder_filters[-2]), SCSEModule(decoder_filters[-2], reduction=16, concat=True))
        self.conv8 = ConvRelu(decoder_filters[-2] * 2, decoder_filters[-3])
        self.conv8_2 = nn.Sequential(ConvRelu(decoder_filters[-3]+encoder_filters[-4] + 2 + 27 + 2, decoder_filters[-3]), SCSEModule(decoder_filters[-3], reduction=16, concat=True))
        self.conv9 = ConvRelu(decoder_filters[-3] * 2, decoder_filters[-4])
        self.conv9_2 = nn.Sequential(ConvRelu(decoder_filters[-4]+encoder_filters[-5] + 2 + 27 + 2, decoder_filters[-4]), SCSEModule(decoder_filters[-4], reduction=16, concat=True))
        self.conv10 = ConvRelu(decoder_filters[-4] * 2 + 2 + 2 + 27, decoder_filters[-5]) # + 27

        self.res = nn.Conv2d(decoder_filters[-5] + 2 + 2, 4, 1, stride=1, padding=0) #+ decoder_filters[-1] + decoder_filters[-2] + decoder_filters[-3]  + 27 

        self.off_nadir = nn.Sequential(nn.Linear(encoder_filters[-1], 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
        
        self._initialize_weights()

        encoder = se_resnext50_32x4d(pretrained=pretrained)

        conv1_new = nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        _w = encoder.layer0.conv1.state_dict()
        _w['weight'] = torch.cat([0.8 * _w['weight'], 0.1 * _w['weight'], 0.1 * _w['weight']], 1)
        conv1_new.load_state_dict(_w)
        self.conv1 = nn.Sequential(conv1_new, encoder.layer0.bn1, encoder.layer0.relu1)
        self.conv2 = nn.Sequential(encoder.pool, encoder.layer1)
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4


    def forward(self, x, y, cat_inp, coord_inp):
        batch_size, C, H, W = x.shape

        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)

        x1 = F.adaptive_avg_pool2d(enc5, output_size=1).view(batch_size, -1)
        x1 = F.dropout(x1, p=0.05, training=self.training)
        x1 = self.off_nadir(x1)

        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4,
                F.upsample(y.view(batch_size, -1, 1, 1,), scale_factor=H//16, mode='nearest'),
                F.upsample(x1.view(batch_size, -1, 1, 1,), scale_factor=H//16, mode='nearest'),
                F.upsample(cat_inp.view(batch_size, -1, 1, 1,), scale_factor=H//16, mode='nearest'),
                F.upsample(coord_inp.view(batch_size, -1, 1, 1,), scale_factor=H//16, mode='nearest')
                ], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3,
                F.upsample(y.view(batch_size, -1, 1, 1,), scale_factor=H//8, mode='nearest'),
                F.upsample(x1.view(batch_size, -1, 1, 1,), scale_factor=H//8, mode='nearest'),
                F.upsample(cat_inp.view(batch_size, -1, 1, 1,), scale_factor=H//8, mode='nearest'),
                F.upsample(coord_inp.view(batch_size, -1, 1, 1,), scale_factor=H//8, mode='nearest')
                ], 1))

        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2,
                F.upsample(y.view(batch_size, -1, 1, 1,), scale_factor=H//4, mode='nearest'),
                F.upsample(x1.view(batch_size, -1, 1, 1,), scale_factor=H//4, mode='nearest'),
                F.upsample(cat_inp.view(batch_size, -1, 1, 1,), scale_factor=H//4, mode='nearest'),
                F.upsample(coord_inp.view(batch_size, -1, 1, 1,), scale_factor=H//4, mode='nearest')
                ], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, 
                enc1,
                F.upsample(y.view(batch_size, -1, 1, 1,), scale_factor=H//2, mode='nearest'),
                F.upsample(x1.view(batch_size, -1, 1, 1,), scale_factor=H//2, mode='nearest'),
                F.upsample(cat_inp.view(batch_size, -1, 1, 1,), scale_factor=H//2, mode='nearest'),
                F.upsample(coord_inp.view(batch_size, -1, 1, 1,), scale_factor=H//2, mode='nearest')
                ], 1))

        dec10 = self.conv10(torch.cat([F.interpolate(dec9, scale_factor=2), 
                F.upsample(y.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest'),
                F.upsample(x1.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest'),
                F.upsample(cat_inp.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest'),
                F.upsample(coord_inp.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest')
                ], 1))

        dec10 = torch.cat([dec10,
                F.upsample(y.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest'),
                F.upsample(x1.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest'),
                # F.upsample(cat_inp.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest'),
                F.upsample(coord_inp.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest')
                ], 1)
        return self.res(dec10), x1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Dpn92_9ch_Unet(nn.Module):
    def __init__(self, pretrained='imagenet+5k', **kwargs):
        super(Dpn92_9ch_Unet, self).__init__()
        
        encoder_filters = [64, 336, 704, 1552, 2688]
        decoder_filters = [64, 96, 128, 256, 512]

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = nn.Sequential(ConvRelu(decoder_filters[-1]+encoder_filters[-2] + 2 + 27 + 2, decoder_filters[-1]), SCSEModule(decoder_filters[-1], reduction=16, concat=True))
        self.conv7 = ConvRelu(decoder_filters[-1] * 2, decoder_filters[-2])
        self.conv7_2 = nn.Sequential(ConvRelu(decoder_filters[-2]+encoder_filters[-3] + 2 + 27 + 2, decoder_filters[-2]), SCSEModule(decoder_filters[-2], reduction=16, concat=True))
        self.conv8 = ConvRelu(decoder_filters[-2] * 2, decoder_filters[-3])
        self.conv8_2 = nn.Sequential(ConvRelu(decoder_filters[-3]+encoder_filters[-4] + 2 + 27 + 2, decoder_filters[-3]), SCSEModule(decoder_filters[-3], reduction=16, concat=True))
        self.conv9 = ConvRelu(decoder_filters[-3] * 2, decoder_filters[-4])
        self.conv9_2 = nn.Sequential(ConvRelu(decoder_filters[-4]+encoder_filters[-5] + 2 + 27 + 2, decoder_filters[-4]), SCSEModule(decoder_filters[-4], reduction=16, concat=True))
        self.conv10 = ConvRelu(decoder_filters[-4] * 2 + 2 + 2 + 27, decoder_filters[-5]) # + 27

        self.res = nn.Conv2d(decoder_filters[-5] + 2 + 2, 4, 1, stride=1, padding=0) #+ decoder_filters[-1] + decoder_filters[-2] + decoder_filters[-3]  + 27 

        self.off_nadir = nn.Sequential(nn.Linear(encoder_filters[-1], 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
        
        self._initialize_weights()

        encoder = dpn92(pretrained=pretrained)

        conv1_new = nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        _w = encoder.blocks['conv1_1'].conv.state_dict()
        _w['weight'] = torch.cat([0.8 * _w['weight'], 0.1 * _w['weight'], 0.1 * _w['weight']], 1)
        conv1_new.load_state_dict(_w)
        
        self.conv1 = nn.Sequential(
                conv1_new,  # conv
                encoder.blocks['conv1_1'].bn,  # bn
                encoder.blocks['conv1_1'].act,  # relu
            )
        self.conv2 = nn.Sequential(
                encoder.blocks['conv1_1'].pool,  # maxpool
                *[b for k, b in encoder.blocks.items() if k.startswith('conv2_')]
            )
        self.conv3 = nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv3_')])
        self.conv4 = nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv4_')])
        self.conv5 = nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv5_')])

    def forward(self, x, y, cat_inp, coord_inp):
        batch_size, C, H, W = x.shape

        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)

        enc1 = (torch.cat(enc1, dim=1) if isinstance(enc1, tuple) else enc1)
        enc2 = (torch.cat(enc2, dim=1) if isinstance(enc2, tuple) else enc2)
        enc3 = (torch.cat(enc3, dim=1) if isinstance(enc3, tuple) else enc3)
        enc4 = (torch.cat(enc4, dim=1) if isinstance(enc4, tuple) else enc4)
        enc5 = (torch.cat(enc5, dim=1) if isinstance(enc5, tuple) else enc5)

        x1 = F.adaptive_avg_pool2d(enc5, output_size=1).view(batch_size, -1)
        x1 = F.dropout(x1, p=0.05, training=self.training)
        x1 = self.off_nadir(x1)

        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4,
                F.upsample(y.view(batch_size, -1, 1, 1,), scale_factor=H//16, mode='nearest'),
                F.upsample(x1.view(batch_size, -1, 1, 1,), scale_factor=H//16, mode='nearest'),
                F.upsample(cat_inp.view(batch_size, -1, 1, 1,), scale_factor=H//16, mode='nearest'),
                F.upsample(coord_inp.view(batch_size, -1, 1, 1,), scale_factor=H//16, mode='nearest')
                ], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3,
                F.upsample(y.view(batch_size, -1, 1, 1,), scale_factor=H//8, mode='nearest'),
                F.upsample(x1.view(batch_size, -1, 1, 1,), scale_factor=H//8, mode='nearest'),
                F.upsample(cat_inp.view(batch_size, -1, 1, 1,), scale_factor=H//8, mode='nearest'),
                F.upsample(coord_inp.view(batch_size, -1, 1, 1,), scale_factor=H//8, mode='nearest')
                ], 1))

        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2,
                F.upsample(y.view(batch_size, -1, 1, 1,), scale_factor=H//4, mode='nearest'),
                F.upsample(x1.view(batch_size, -1, 1, 1,), scale_factor=H//4, mode='nearest'),
                F.upsample(cat_inp.view(batch_size, -1, 1, 1,), scale_factor=H//4, mode='nearest'),
                F.upsample(coord_inp.view(batch_size, -1, 1, 1,), scale_factor=H//4, mode='nearest')
                ], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, 
                enc1,
                F.upsample(y.view(batch_size, -1, 1, 1,), scale_factor=H//2, mode='nearest'),
                F.upsample(x1.view(batch_size, -1, 1, 1,), scale_factor=H//2, mode='nearest'),
                F.upsample(cat_inp.view(batch_size, -1, 1, 1,), scale_factor=H//2, mode='nearest'),
                F.upsample(coord_inp.view(batch_size, -1, 1, 1,), scale_factor=H//2, mode='nearest')
                ], 1))

        dec10 = self.conv10(torch.cat([F.interpolate(dec9, scale_factor=2), 
                F.upsample(y.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest'),
                F.upsample(x1.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest'),
                F.upsample(cat_inp.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest'),
                F.upsample(coord_inp.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest')
                ], 1))

        dec10 = torch.cat([dec10,
                F.upsample(y.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest'),
                F.upsample(x1.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest'),
                # F.upsample(cat_inp.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest'),
                F.upsample(coord_inp.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest')
                ], 1)
        return self.res(dec10), x1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ScSeSenet154_9ch_Unet(nn.Module):
    def __init__(self, pretrained='imagenet', **kwargs):
        super(ScSeSenet154_9ch_Unet, self).__init__()
        
        encoder_filters = [128, 256, 512, 1024, 2048]
        decoder_filters = [96, 128, 160, 256, 512]

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(decoder_filters[-1]+encoder_filters[-2] + 2 + 27 + 2, decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(decoder_filters[-2]+encoder_filters[-3] + 2 + 27 + 2, decoder_filters[-2])
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(decoder_filters[-3]+encoder_filters[-4] + 2 + 27 + 2, decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvRelu(decoder_filters[-4]+encoder_filters[-5] + 2 + 27 + 2, decoder_filters[-4])
        self.conv10 = ConvRelu(decoder_filters[-4] + 2 + 2 + 27, decoder_filters[-5]) # + 27  * 2

        self.res = nn.Conv2d(decoder_filters[-5] + 2 + 2, 4, 1, stride=1, padding=0) #+ decoder_filters[-1] + decoder_filters[-2] + decoder_filters[-3]  + 27 

        self.off_nadir = nn.Sequential(nn.Linear(encoder_filters[-1], 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
        
        self._initialize_weights()

        encoder = senet154(pretrained=pretrained)

        conv1_new = nn.Conv2d(9, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        _w = encoder.layer0.conv1.state_dict()
        _w['weight'] = torch.cat([0.8 * _w['weight'], 0.1 * _w['weight'], 0.1 * _w['weight']], 1)
        conv1_new.load_state_dict(_w)
        self.conv1 = nn.Sequential(conv1_new, encoder.layer0.bn1, encoder.layer0.relu1, encoder.layer0.conv2, encoder.layer0.bn2, encoder.layer0.relu2, encoder.layer0.conv3, encoder.layer0.bn3, encoder.layer0.relu3)
        self.conv2 = nn.Sequential(encoder.pool, encoder.layer1)
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4

    def forward(self, x, y, cat_inp, coord_inp):
        batch_size, C, H, W = x.shape

        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)

        x1 = F.adaptive_avg_pool2d(enc5, output_size=1).view(batch_size, -1)
        x1 = F.dropout(x1, p=0.05, training=self.training)
        x1 = self.off_nadir(x1)

        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4,
                F.upsample(y.view(batch_size, -1, 1, 1,), scale_factor=H//16, mode='nearest'),
                F.upsample(x1.view(batch_size, -1, 1, 1,), scale_factor=H//16, mode='nearest'),
                F.upsample(cat_inp.view(batch_size, -1, 1, 1,), scale_factor=H//16, mode='nearest'),
                F.upsample(coord_inp.view(batch_size, -1, 1, 1,), scale_factor=H//16, mode='nearest')
                ], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3,
                F.upsample(y.view(batch_size, -1, 1, 1,), scale_factor=H//8, mode='nearest'),
                F.upsample(x1.view(batch_size, -1, 1, 1,), scale_factor=H//8, mode='nearest'),
                F.upsample(cat_inp.view(batch_size, -1, 1, 1,), scale_factor=H//8, mode='nearest'),
                F.upsample(coord_inp.view(batch_size, -1, 1, 1,), scale_factor=H//8, mode='nearest')
                ], 1))

        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2,
                F.upsample(y.view(batch_size, -1, 1, 1,), scale_factor=H//4, mode='nearest'),
                F.upsample(x1.view(batch_size, -1, 1, 1,), scale_factor=H//4, mode='nearest'),
                F.upsample(cat_inp.view(batch_size, -1, 1, 1,), scale_factor=H//4, mode='nearest'),
                F.upsample(coord_inp.view(batch_size, -1, 1, 1,), scale_factor=H//4, mode='nearest')
                ], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, 
                enc1,
                F.upsample(y.view(batch_size, -1, 1, 1,), scale_factor=H//2, mode='nearest'),
                F.upsample(x1.view(batch_size, -1, 1, 1,), scale_factor=H//2, mode='nearest'),
                F.upsample(cat_inp.view(batch_size, -1, 1, 1,), scale_factor=H//2, mode='nearest'),
                F.upsample(coord_inp.view(batch_size, -1, 1, 1,), scale_factor=H//2, mode='nearest')
                ], 1))

        dec10 = self.conv10(torch.cat([F.interpolate(dec9, scale_factor=2), 
                F.upsample(y.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest'),
                F.upsample(x1.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest'),
                F.upsample(cat_inp.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest'),
                F.upsample(coord_inp.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest')
                ], 1))

        dec10 = torch.cat([dec10,
                F.upsample(y.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest'),
                F.upsample(x1.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest'),
                # F.upsample(cat_inp.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest'),
                F.upsample(coord_inp.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest')
                ], 1)
        return self.res(dec10), x1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ScSeResNext101_9ch_Unet(nn.Module):
    def __init__(self, pretrained='imagenet', **kwargs):
        super(ScSeResNext101_9ch_Unet, self).__init__()
        
        encoder_filters = [64, 256, 512, 1024, 2048]
        decoder_filters = [64, 96, 128, 256, 512]

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = nn.Sequential(ConvRelu(decoder_filters[-1]+encoder_filters[-2] + 2 + 27 + 2, decoder_filters[-1]), SCSEModule(decoder_filters[-1], reduction=16, concat=True))
        self.conv7 = ConvRelu(decoder_filters[-1] * 2, decoder_filters[-2])
        self.conv7_2 = nn.Sequential(ConvRelu(decoder_filters[-2]+encoder_filters[-3] + 2 + 27 + 2, decoder_filters[-2]), SCSEModule(decoder_filters[-2], reduction=16, concat=True))
        self.conv8 = ConvRelu(decoder_filters[-2] * 2, decoder_filters[-3])
        self.conv8_2 = nn.Sequential(ConvRelu(decoder_filters[-3]+encoder_filters[-4] + 2 + 27 + 2, decoder_filters[-3]), SCSEModule(decoder_filters[-3], reduction=16, concat=True))
        self.conv9 = ConvRelu(decoder_filters[-3] * 2, decoder_filters[-4])
        self.conv9_2 = nn.Sequential(ConvRelu(decoder_filters[-4]+encoder_filters[-5] + 2 + 27 + 2, decoder_filters[-4]), SCSEModule(decoder_filters[-4], reduction=16, concat=True))
        self.conv10 = ConvRelu(decoder_filters[-4] * 2 + 2 + 2 + 27, decoder_filters[-5]) # + 27

        self.res = nn.Conv2d(decoder_filters[-5] + 2 + 2, 4, 1, stride=1, padding=0) #+ decoder_filters[-1] + decoder_filters[-2] + decoder_filters[-3]  + 27 

        self.off_nadir = nn.Sequential(nn.Linear(encoder_filters[-1], 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
        
        self._initialize_weights()

        encoder = se_resnext101_32x4d(pretrained=pretrained)

        conv1_new = nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        _w = encoder.layer0.conv1.state_dict()
        _w['weight'] = torch.cat([0.8 * _w['weight'], 0.1 * _w['weight'], 0.1 * _w['weight']], 1)
        conv1_new.load_state_dict(_w)
        self.conv1 = nn.Sequential(conv1_new, encoder.layer0.bn1, encoder.layer0.relu1)
        self.conv2 = nn.Sequential(encoder.pool, encoder.layer1)
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4


    def forward(self, x, y, cat_inp, coord_inp):
        batch_size, C, H, W = x.shape

        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)

        x1 = F.adaptive_avg_pool2d(enc5, output_size=1).view(batch_size, -1)
        x1 = F.dropout(x1, p=0.05, training=self.training)
        x1 = self.off_nadir(x1)

        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4,
                F.upsample(y.view(batch_size, -1, 1, 1,), scale_factor=H//16, mode='nearest'),
                F.upsample(x1.view(batch_size, -1, 1, 1,), scale_factor=H//16, mode='nearest'),
                F.upsample(cat_inp.view(batch_size, -1, 1, 1,), scale_factor=H//16, mode='nearest'),
                F.upsample(coord_inp.view(batch_size, -1, 1, 1,), scale_factor=H//16, mode='nearest')
                ], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3,
                F.upsample(y.view(batch_size, -1, 1, 1,), scale_factor=H//8, mode='nearest'),
                F.upsample(x1.view(batch_size, -1, 1, 1,), scale_factor=H//8, mode='nearest'),
                F.upsample(cat_inp.view(batch_size, -1, 1, 1,), scale_factor=H//8, mode='nearest'),
                F.upsample(coord_inp.view(batch_size, -1, 1, 1,), scale_factor=H//8, mode='nearest')
                ], 1))

        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2,
                F.upsample(y.view(batch_size, -1, 1, 1,), scale_factor=H//4, mode='nearest'),
                F.upsample(x1.view(batch_size, -1, 1, 1,), scale_factor=H//4, mode='nearest'),
                F.upsample(cat_inp.view(batch_size, -1, 1, 1,), scale_factor=H//4, mode='nearest'),
                F.upsample(coord_inp.view(batch_size, -1, 1, 1,), scale_factor=H//4, mode='nearest')
                ], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, 
                enc1,
                F.upsample(y.view(batch_size, -1, 1, 1,), scale_factor=H//2, mode='nearest'),
                F.upsample(x1.view(batch_size, -1, 1, 1,), scale_factor=H//2, mode='nearest'),
                F.upsample(cat_inp.view(batch_size, -1, 1, 1,), scale_factor=H//2, mode='nearest'),
                F.upsample(coord_inp.view(batch_size, -1, 1, 1,), scale_factor=H//2, mode='nearest')
                ], 1))

        dec10 = self.conv10(torch.cat([F.interpolate(dec9, scale_factor=2), 
                F.upsample(y.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest'),
                F.upsample(x1.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest'),
                F.upsample(cat_inp.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest'),
                F.upsample(coord_inp.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest')
                ], 1))

        dec10 = torch.cat([dec10,
                F.upsample(y.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest'),
                F.upsample(x1.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest'),
                # F.upsample(cat_inp.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest'),
                F.upsample(coord_inp.view(batch_size, -1, 1, 1,), scale_factor=H, mode='nearest')
                ], 1)
        return self.res(dec10), x1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
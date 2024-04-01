
import torch.nn as nn
import torch


class ResnetBranch(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(ResnetBranch, self).__init__()

        ngf=64
        norm_layer=nn.BatchNorm2d
        padding_type='reflect'
        n_blocks=9
        use_bias = False
        use_dropout = True

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        self.down1 = nn.Sequential(*model)


        mult = 1
        model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        self.down2 = nn.Sequential(*model)


        mult = 2
        model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        self.down3 = nn.Sequential(*model)

        self.guideDecoder = GuideDecoder(n_blocks,output_nc=3)

        mult = 4
        self.block1 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        
        self.block2 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.block3 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.block4 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.block5 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.block6 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.block7 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.block8 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.block9 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        
        self.conv1 = nn.Conv2d(ngf * mult * 2, ngf * mult, 3, padding=1,bias=use_bias)
        self.conv2 = nn.Conv2d(ngf * mult * 2, ngf * mult, 3, padding=1,bias=use_bias)
        self.conv3 = nn.Conv2d(ngf * mult * 2, ngf * mult, 3, padding=1,bias=use_bias)
        self.conv4 = nn.Conv2d(ngf * mult * 2, ngf * mult, 3, padding=1,bias=use_bias)
        self.conv5 = nn.Conv2d(ngf * mult * 2, ngf * mult, 3, padding=1,bias=use_bias)
        self.conv6 = nn.Conv2d(ngf * mult * 2, ngf * mult, 3, padding=1,bias=use_bias)
        self.conv7 = nn.Conv2d(ngf * mult * 2, ngf * mult, 3, padding=1,bias=use_bias)
        self.conv8 = nn.Conv2d(ngf * mult * 2, ngf * mult, 3, padding=1,bias=use_bias)
        self.conv9 = nn.Conv2d(ngf * mult * 2, ngf * mult, 3, padding=1,bias=use_bias)
       
        ## up
        mult = 4
        model = [nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ngf * mult, int(ngf * mult / 2), 3, padding=1,bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)]
        self.up1 = nn.Sequential(*model)

        mult = 2
        model = [nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ngf * mult, int(ngf * mult / 2), 3, padding=1,bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)]
        self.up2 = nn.Sequential(*model)

        model = [nn.ReflectionPad2d(3),nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),nn.Tanh()]
        self.out = nn.Sequential(*model)

    def forward(self,input):
        e1 = self.down1(input)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        guide_pred, guide_feats = self.guideDecoder([e2,e3])
        
        r1 = self.block1(e3)
        r1 = torch.cat([r1,guide_feats[0]],1)
        r1 = self.conv1(r1)

        r2 = self.block2(r1)
        r2 = torch.cat([r2,guide_feats[1]],1)
        r2 = self.conv2(r2)

        r3 = self.block3(r2)
        r3 = torch.cat([r3,guide_feats[2]],1)
        r3 = self.conv3(r3)

        r4 = self.block4(r3)
        r4 = torch.cat([r4,guide_feats[3]],1)
        r4 = self.conv4(r4)

        r5 = self.block5(r4)
        r5 = torch.cat([r5,guide_feats[4]],1)
        r5 = self.conv5(r5)

        r6 = self.block6(r5)
        r6 = torch.cat([r6,guide_feats[5]],1)
        r6 = self.conv6(r6)

        r7 = self.block7(r6)
        r7 = torch.cat([r7,guide_feats[6]],1)
        r7 = self.conv7(r7)
        
        r8 = self.block8(r7)
        r8 = torch.cat([r8,guide_feats[7]],1)
        r8 = self.conv8(r8)

        r9 = self.block9(r8)
        r9 = torch.cat([r9,guide_feats[8]],1)
        r9 = self.conv9(r9)

        d1 = r9 + e3
        d2 = self.up1(d1)
        d2 = d2 + e2
        d3 = self.up2(d2)
        stress_pred = self.out(d3)
        
        return stress_pred, guide_pred
        # torch.Size([1, 2, 256, 256]) torch.Size([1, 64, 256, 256]) torch.Size([1, 128, 128, 128]) torch.Size([1, 256, 64, 64]) torch.Size([1, 256, 64, 64])

class GuideDecoder(nn.Module):
    def __init__(self,n_blocks,output_nc):
        super(GuideDecoder,self).__init__()
        self.n_blocks = n_blocks
        
        ngf=64
        norm_layer=nn.BatchNorm2d
        padding_type='reflect'
        n_blocks=9
        use_bias = False
        use_dropout = True

        mult = 4
        self.block1 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.block2 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.block3 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.block4 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.block5 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.block6 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.block7 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.block8 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.block9 = ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)

        ## up
        mult = 4
        model = [nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ngf * mult, int(ngf * mult / 2), 3, padding=1,bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)]
        self.up1 = nn.Sequential(*model)

        mult = 2
        model = [nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ngf * mult, int(ngf * mult / 2), 3, padding=1,bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)]
        self.up2 = nn.Sequential(*model)

        model = [nn.ReflectionPad2d(3),nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),nn.Tanh()]
        self.out = nn.Sequential(*model)

    def forward(self,features):
        e2,e3 = features
        
        r1 = self.block1(e3)
        r2 = self.block2(r1)
        r3 = self.block3(r2)
        r4 = self.block4(r3)
        r5 = self.block5(r4)
        r6 = self.block6(r5)
        r7 = self.block7(r6)
        r8 = self.block8(r7)
        r9 = self.block9(r8)

        d1 = r9 + e3
        d2 = self.up1(d1)
        d2 = d2 + e2
        d3 = self.up2(d2)

        output = self.out(d3)

        res_feats = [r1,r2,r3,r4,r5,r6,r7,r8,r9]

        return output, res_feats




class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

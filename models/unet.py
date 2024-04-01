import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import AttentionModule

class UnetSingle(nn.Module):
    def __init__(self, input_nc,output_nc):
        super(UnetSingle, self).__init__()

        self.encoder = UnetEncoder(input_nc)
        self.decoder = UnetDecoder(output_nc)

    def forward(self,input):
        features = self.encoder.forward(input)
        output  = self.decoder(features)

        return output




class UnetBranch_coordinates_input(nn.Module):
    def __init__(self, input_nc, output_nc, netG):
        super(UnetBranch_coordinates_input, self).__init__()

        self.encoder = UnetEncoder(input_nc)
        if "normal" in netG:
            self.depthDecoder = SumDecoder_coordinates_input(output_nc=3)
        else:
            self.depthDecoder = SumDecoder_coordinates_input(output_nc=1)

        self.stressDecoder = ConcatSumDecoder2_coordinates_input(output_nc)

    def forward(self, input, coordinates):
        features = self.encoder.forward(input)
        features[6]=torch.cat((features[6], coordinates.float()),1)
        depth_features = self.depthDecoder(features)
        depth_pred = depth_features[-1]
        stress_pred = self.stressDecoder(features, depth_features)

        return stress_pred, depth_pred

class UnetBranch(nn.Module):
    def __init__(self, input_nc,output_nc,netG):
        super(UnetBranch, self).__init__()

        self.encoder = UnetEncoder(input_nc)
        if "normal" in netG:
            self.depthDecoder = SumDecoder(output_nc=3)
        else:
            self.depthDecoder = SumDecoder(output_nc=1)
        
        
        self.stressDecoder = ConcatSumDecoder2(output_nc)

    def forward(self,input):
        features = self.encoder.forward(input)
        depth_features  = self.depthDecoder(features)
        depth_pred = depth_features[-1]
        stress_pred = self.stressDecoder(features,depth_features)

        return stress_pred, depth_pred

class ConcatSumDecoder2(nn.Module):
    def __init__(self, output_nc):
        super().__init__()
        d = 64
        use_bias = False
        
        self.deconv1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),nn.Conv2d(d * 8, d * 8, 3, padding=1,bias=use_bias))
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),nn.Conv2d(d * 8 * 2, d * 8, 3, padding=1,bias=use_bias))
        self.deconv2_bn = nn.BatchNorm2d(d * 8)
        self.deconv3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),nn.Conv2d(d * 8 * 2, d * 8, 3, padding=1,bias=use_bias))
        self.deconv3_bn = nn.BatchNorm2d(d * 8)
        self.deconv4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),nn.Conv2d(d * 8 * 2, d * 4, 3, padding=1,bias=use_bias))
        self.deconv4_bn = nn.BatchNorm2d(d * 4) #256
        self.deconv5 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),nn.Conv2d(d * 4 * 2, d * 2, 3, padding=1,bias=use_bias))
        self.deconv5_bn = nn.BatchNorm2d(d * 2) # 128
        self.deconv6 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),nn.Conv2d(d * 2 * 2, d, 3, padding=1,bias=use_bias))
        self.deconv6_bn = nn.BatchNorm2d(d) # 64
        self.deconv7 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),nn.Conv2d(d * 2, output_nc, 3, padding=1,bias=use_bias))

        # self.deconv7_bn = nn.BatchNorm2d(d) 
        # self.conv_final = nn.Conv2d(d, output_nc, 3, padding=1,bias=use_bias)

    def forward(self,sk_features,depth_features):
        e1,e2,e3,e4,e5,e6,e7 = sk_features # sum features
        c1,c2,c3,c4,c5,c6,c7 = depth_features # concat features

        d1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(e7))), 0.5, training=True)

        d1 = d1 + e6
        d1 = torch.cat([d1, c1], 1)
        
        d2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(d1))), 0.5, training=True)
        d2 = d2 + e5
        d2 = torch.cat([d2, c2], 1)
        
        d3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(d2))), 0.5, training=True)
        d3 = d3 + e4
        d3 = torch.cat([d3, c3], 1)
        
        d4 = self.deconv4_bn(self.deconv4(F.relu(d3)))
        d4 = d4 + e3
        d4 = torch.cat([d4, c4], 1)

        d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))
        d5 = d5 + e2
        d5 = torch.cat([d5, c5], 1)

        d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
        d6 = d6 + e1
        d6 = torch.cat([d6, c6], 1)

        d7 = self.deconv7(F.relu(d6))
        output = torch.tanh(d7)

        # d7 = self.deconv7_bn(self.deconv7(F.relu(d6)))
        # final = self.conv_final(F.relu(d7))

        # output = torch.tanh(final)

        return output


class ConcatSumDecoder2_coordinates_input(nn.Module):
    def __init__(self, output_nc):
        super().__init__()
        d = 64
        use_bias = False

        self.deconv1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                     nn.Conv2d(d * 8+1, d * 8, 3, padding=1, bias=use_bias))
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                     nn.Conv2d(d * 8 * 2, d * 8, 3, padding=1, bias=use_bias))
        self.deconv2_bn = nn.BatchNorm2d(d * 8)
        self.deconv3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                     nn.Conv2d(d * 8 * 2, d * 8, 3, padding=1, bias=use_bias))
        self.deconv3_bn = nn.BatchNorm2d(d * 8)
        self.deconv4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                     nn.Conv2d(d * 8 * 2, d * 4, 3, padding=1, bias=use_bias))
        self.deconv4_bn = nn.BatchNorm2d(d * 4)  # 256
        self.deconv5 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                     nn.Conv2d(d * 4 * 2, d * 2, 3, padding=1, bias=use_bias))
        self.deconv5_bn = nn.BatchNorm2d(d * 2)  # 128
        self.deconv6 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                     nn.Conv2d(d * 2 * 2, d, 3, padding=1, bias=use_bias))
        self.deconv6_bn = nn.BatchNorm2d(d)  # 64
        self.deconv7 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                     nn.Conv2d(d * 2, output_nc, 3, padding=1, bias=use_bias))

        # self.deconv7_bn = nn.BatchNorm2d(d)
        # self.conv_final = nn.Conv2d(d, output_nc, 3, padding=1,bias=use_bias)

    def forward(self, sk_features, depth_features):
        e1, e2, e3, e4, e5, e6, e7 = sk_features  # sum features
        c1, c2, c3, c4, c5, c6, c7 = depth_features  # concat features

        d1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(e7))), 0.5, training=True)

        d1 = d1 + e6
        d1 = torch.cat([d1, c1], 1)

        d2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(d1))), 0.5, training=True)
        d2 = d2 + e5
        d2 = torch.cat([d2, c2], 1)

        d3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(d2))), 0.5, training=True)
        d3 = d3 + e4
        d3 = torch.cat([d3, c3], 1)

        d4 = self.deconv4_bn(self.deconv4(F.relu(d3)))
        d4 = d4 + e3
        d4 = torch.cat([d4, c4], 1)

        d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))
        d5 = d5 + e2
        d5 = torch.cat([d5, c5], 1)

        d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
        d6 = d6 + e1
        d6 = torch.cat([d6, c6], 1)

        d7 = self.deconv7(F.relu(d6))
        output = torch.tanh(d7)

        # d7 = self.deconv7_bn(self.deconv7(F.relu(d6)))
        # final = self.conv_final(F.relu(d7))

        # output = torch.tanh(final)

        return output


class UnetBranchAt(nn.Module):
    def __init__(self, input_nc,output_nc):
        super(UnetBranchAt, self).__init__()

        self.encoder = UnetEncoder(input_nc)
        
        self.depthDecoder = SumDecoder(output_nc=1)
        self.stressDecoder = ConcatSumAtDecoder(output_nc)

    def forward(self,input):
        features = self.encoder.forward(input)
        depth_features  = self.depthDecoder(features)
        depth_pred = depth_features[-1]
        stress_pred = self.stressDecoder(features,depth_features)

        return stress_pred, depth_pred
     

class UnetEncoder(nn.Module):
    def __init__(self, input_nc):
        super().__init__()
        d = 64
        use_bias = False

        # Unet encoder
        self.conv1 = nn.Conv2d(input_nc, d, 4, 2, 1,bias=use_bias)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1,bias=use_bias)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1,bias=use_bias)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1,bias=use_bias)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, d * 8, 4, 2, 1,bias=use_bias)
        self.conv5_bn = nn.BatchNorm2d(d * 8)
        self.conv6 = nn.Conv2d(d * 8, d * 8, 4, 2, 1,bias=use_bias)
        self.conv6_bn = nn.BatchNorm2d(d * 8)
        self.conv7 = nn.Conv2d(d * 8, d * 8, 4, 2, 1,bias=use_bias)
        self.conv7_bn = nn.BatchNorm2d(d * 8)

    def forward(self, input):
        e1 = self.conv1(input)
        e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))
        e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))
        e4 = self.conv4_bn(self.conv4(F.leaky_relu(e3, 0.2)))
        e5 = self.conv5_bn(self.conv5(F.leaky_relu(e4, 0.2)))
        e6 = self.conv6_bn(self.conv6(F.leaky_relu(e5, 0.2)))
        e7 = self.conv7_bn(self.conv7(F.leaky_relu(e6, 0.2)))

        return [e1,e2,e3,e4,e5,e6,e7]

class UnetDecoder(nn.Module):
    def __init__(self, output_nc):
        super().__init__()
        d = 64
        use_bias = False

        self.deconv1 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1,bias=use_bias)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1,bias=use_bias)
        self.deconv2_bn = nn.BatchNorm2d(d * 8)
        self.deconv3 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1,bias=use_bias)
        self.deconv3_bn = nn.BatchNorm2d(d * 8)
        self.deconv4 = nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1,bias=use_bias)
        self.deconv4_bn = nn.BatchNorm2d(d * 4) #256
        self.deconv5 = nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1,bias=use_bias)
        self.deconv5_bn = nn.BatchNorm2d(d * 2) # 128
        self.deconv6 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1,bias=use_bias)
        self.deconv6_bn = nn.BatchNorm2d(d) # 64
        self.deconv7 = nn.ConvTranspose2d(d * 2, output_nc, 4, 2, 1,bias=use_bias) # output nc
        # self.deconv7_bn = nn.BatchNorm2d(d) 

        # self.conv_final = nn.Conv2d(d, output_nc, 3, padding=1,bias=use_bias)

    def forward(self,sk_features):
        e1,e2,e3,e4,e5,e6,e7 = sk_features

        d1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(e7))), 0.5, training=True)
        d1 = torch.cat([d1, e6], 1)
        
        d2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(d1))), 0.5, training=True)
        d2 = torch.cat([d2, e5], 1)
        
        d3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(d2))), 0.5, training=True)
        d3 = torch.cat([d3, e4], 1)
        
        
        d4 = self.deconv4_bn(self.deconv4(F.relu(d3)))
        d4 = torch.cat([d4, e3], 1)
        

        d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))
        d5 = torch.cat([d5, e2], 1)

        d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
        d6 = torch.cat([d6, e1], 1)

        d7 = self.deconv7(F.relu(d6))

        output = torch.tanh(d7)
        return output

class ConcatSumAtDecoder(nn.Module):
    def __init__(self, output_nc):
        super().__init__()
        d = 64
        use_bias = False

        self.deconv1 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1,bias=use_bias)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1,bias=use_bias)
        self.deconv2_bn = nn.BatchNorm2d(d * 8)
        self.deconv3 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1,bias=use_bias)
        self.deconv3_bn = nn.BatchNorm2d(d * 8)
        self.deconv4 = nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1,bias=use_bias)
        self.deconv4_bn = nn.BatchNorm2d(d * 4) #256
        self.deconv5 = nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1,bias=use_bias)
        self.deconv5_bn = nn.BatchNorm2d(d * 2) # 128
        self.deconv6 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1,bias=use_bias)
        self.deconv6_bn = nn.BatchNorm2d(d) # 64
        self.deconv7 = nn.ConvTranspose2d(d * 2, output_nc, 4, 2, 1,bias=use_bias) # output nc

        self.at1 = AttentionModule(1024)
        self.at2 = AttentionModule(1024)
        self.at3 = AttentionModule(1024)

    def forward(self,sk_features,depth_features):
        e1,e2,e3,e4,e5,e6,e7 = sk_features # sum features
        c1,c2,c3,c4,c5,c6,c7 = depth_features # concat features

        d1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(e7))), 0.5, training=True)
        d1 = d1 + e6
        d1 = torch.cat([d1, c1], 1)
        at_d1 = self.at1(d1)

        
        d2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(at_d1))), 0.5, training=True)
        d2 = d2 + e5
        d2 = torch.cat([d2, c2], 1)
        at_d2 = self.at2(d2)
        
        d3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(at_d2))), 0.5, training=True)
        d3 = d3 + e4
        d3 = torch.cat([d3, c3], 1)
        at_d3 = self.at3(d3)
        
        d4 = self.deconv4_bn(self.deconv4(F.relu(at_d3)))
        d4 = d4 + e3
        d4 = torch.cat([d4, c4], 1)

        d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))
        d5 = d5 + e2
        d5 = torch.cat([d5, c5], 1)

        d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
        d6 = d6 + e1
        d6 = torch.cat([d6, c6], 1)

        d7 = self.deconv7(F.relu(d6))
        output = torch.tanh(d7)

        return output

class ConcatSumDecoder(nn.Module):
    def __init__(self, output_nc):
        super().__init__()
        d = 64
        use_bias = False

        self.deconv1 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1,bias=use_bias)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1,bias=use_bias)
        self.deconv2_bn = nn.BatchNorm2d(d * 8)
        self.deconv3 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1,bias=use_bias)
        self.deconv3_bn = nn.BatchNorm2d(d * 8)
        self.deconv4 = nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1,bias=use_bias)
        self.deconv4_bn = nn.BatchNorm2d(d * 4) #256
        self.deconv5 = nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1,bias=use_bias)
        self.deconv5_bn = nn.BatchNorm2d(d * 2) # 128
        self.deconv6 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1,bias=use_bias)
        self.deconv6_bn = nn.BatchNorm2d(d) # 64
        self.deconv7 = nn.ConvTranspose2d(d * 2, output_nc, 4, 2, 1,bias=use_bias) # output nc
        # self.deconv7_bn = nn.BatchNorm2d(d) 

        # self.conv_final = nn.Conv2d(d, output_nc, 3, padding=1,bias=use_bias)

    def forward(self,sk_features,depth_features):
        e1,e2,e3,e4,e5,e6,e7 = sk_features # sum features
        c1,c2,c3,c4,c5,c6,c7 = depth_features # concat features

        d1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(e7))), 0.5, training=True)
        d1 = d1 + e6
        d1 = torch.cat([d1, c1], 1)
        
        d2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(d1))), 0.5, training=True)
        d2 = d2 + e5
        d2 = torch.cat([d2, c2], 1)
        
        d3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(d2))), 0.5, training=True)
        d3 = d3 + e4
        d3 = torch.cat([d3, c3], 1)
        
        d4 = self.deconv4_bn(self.deconv4(F.relu(d3)))
        d4 = d4 + e3
        d4 = torch.cat([d4, c4], 1)

        d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))
        d5 = d5 + e2
        d5 = torch.cat([d5, c5], 1)

        d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
        d6 = d6 + e1
        d6 = torch.cat([d6, c6], 1)

        d7 = self.deconv7(F.relu(d6))
        output = torch.tanh(d7)
        return output

class SumDecoder(nn.Module):
    def __init__(self, output_nc):
        super().__init__()
        d = 64
        use_bias = False

        self.deconv1 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1,bias=use_bias)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1,bias=use_bias)
        self.deconv2_bn = nn.BatchNorm2d(d * 8)
        self.deconv3 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1,bias=use_bias)
        self.deconv3_bn = nn.BatchNorm2d(d * 8)
        self.deconv4 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1,bias=use_bias)
        self.deconv4_bn = nn.BatchNorm2d(d * 4) #256
        self.deconv5 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1,bias=use_bias)
        self.deconv5_bn = nn.BatchNorm2d(d * 2) # 128
        self.deconv6 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1,bias=use_bias)
        self.deconv6_bn = nn.BatchNorm2d(d) # 64
        self.deconv7 = nn.ConvTranspose2d(d, output_nc, 4, 2, 1,bias=use_bias) # output nc
        # self.deconv7_bn = nn.BatchNorm2d(d) 

        # self.conv_final = nn.Conv2d(d, output_nc, 3, padding=1,bias=use_bias)

    def forward(self,sum_features):
        e1,e2,e3,e4,e5,e6,e7 = sum_features

        d1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(e7))), 0.5, training=True)
        d1 = d1 + e6
        
        d2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(d1))), 0.5, training=True)
        d2 = d2 + e5
        
        d3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(d2))), 0.5, training=True)
        d3 = d3 + e4
        
        d4 = self.deconv4_bn(self.deconv4(F.relu(d3)))
        d4 = d4 + e3

        d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))
        d5 = d5 + e2

        d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
        d6 = d6 + e1
        
        d7 = self.deconv7(F.relu(d6))

        output = torch.tanh(d7)

        return [d1,d2,d3,d4,d5,d6,output]


class SumDecoder_coordinates_input(nn.Module):
    def __init__(self, output_nc):
        super().__init__()
        d = 64
        use_bias = False

        self.deconv1 = nn.ConvTranspose2d(d * 8 + 1, d * 8, 4, 2, 1, bias=use_bias)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1, bias=use_bias)
        self.deconv2_bn = nn.BatchNorm2d(d * 8)
        self.deconv3 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1, bias=use_bias)
        self.deconv3_bn = nn.BatchNorm2d(d * 8)
        self.deconv4 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1, bias=use_bias)
        self.deconv4_bn = nn.BatchNorm2d(d * 4)  # 256
        self.deconv5 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1, bias=use_bias)
        self.deconv5_bn = nn.BatchNorm2d(d * 2)  # 128
        self.deconv6 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1, bias=use_bias)
        self.deconv6_bn = nn.BatchNorm2d(d)  # 64
        self.deconv7 = nn.ConvTranspose2d(d, output_nc, 4, 2, 1, bias=use_bias)  # output nc
        # self.deconv7_bn = nn.BatchNorm2d(d)

        # self.conv_final = nn.Conv2d(d, output_nc, 3, padding=1,bias=use_bias)

    def forward(self, sum_features):
        e1, e2, e3, e4, e5, e6, e7 = sum_features

        d1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(e7))), 0.5, training=True)
        d1 = d1 + e6

        d2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(d1))), 0.5, training=True)
        d2 = d2 + e5

        d3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(d2))), 0.5, training=True)
        d3 = d3 + e4

        d4 = self.deconv4_bn(self.deconv4(F.relu(d3)))
        d4 = d4 + e3

        d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))
        d5 = d5 + e2

        d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
        d6 = d6 + e1

        d7 = self.deconv7(F.relu(d6))

        output = torch.tanh(d7)

        return [d1, d2, d3, d4, d5, d6, output]
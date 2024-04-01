import torch
from models.base_model import BaseModel
from models import networks
import matplotlib.pyplot as plt
import numpy as np

class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    # @staticmethod
    # def modify_commandline_options(parser, is_train=True):
    #     """Add new dataset-specific options, and rewrite default values for existing options.

    #     Parameters:
    #         parser          -- original option parser
    #         is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

    #     Returns:
    #         the modified parser.

    #     For pix2pix, we do not use image buffer
    #     The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
    #     By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
    #     """
    #     # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
    #     parser.set_defaults(norm='batch', netG='branch_depth', dataset_mode='aligned')
    #     print("is train")
    #     exit()
    #     if is_train:
            
    #         parser.set_defaults(pool_size=0, gan_mode='vanilla')
    #         parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
    #         parser.add_argument('--lambda_depth', type=float, default=100.0, help='weight for depth loss')

    #     return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.opt=opt

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'G_depth']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'real_depth', 'fake_depth']
        if 'mask' in opt.model_name:
            self.loss_names.append('G_M')
            self.visual_names+=['real_mask','fake_mask']
        if opt.D_normal:
            self.loss_names.append('G_GAN_normal')
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
            if opt.D_normal:
                self.model_names.append('D_normal')
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)


        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            if 'mask' in opt.model_name:
                self.netD = networks.define_D(opt.input_nc + opt.output_nc - 1, opt.ndf, opt.netD,opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
                if opt.D_normal:
                    self.netD_normal = networks.define_D(opt.input_nc + opt.output_nc - 1, opt.ndf, opt.netD,opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            else:
                self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
                if opt.D_normal:
                    self.netD_normal = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            if opt.model_name.endswith('L2'):
                self.criterionL1 = torch.nn.MSELoss()
            else:
                self.criterionL1 = torch.nn.L1Loss()
            # self.perception_loss = lpips.LPIPS(net='vgg16').to(self.device)  ## here we added a perception loss


            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            if opt.D_normal:
                self.optimizer_D_normal = torch.optim.Adam(self.netD_normal.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D_normal)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        # AtoB = self.opt.direction == 'AtoB'
        self.sk_real=input['sk'].to(self.device) #sketch
        self.fm_real=input['force'].to(self.device) # force node mask
        self.real_A= torch.cat((self.sk_real,self.fm_real),1)
        self.real_B=input['stress'].to(self.device) # stress_map
        self.real_depth = input['guidance'].to(self.device) # depth_map


        if 'mask' in self.opt.model_name:
            self.real_mask = input['mask'].to(self.device)

        self.image_paths = input['img_path']

        if 'weighted' in self.opt.model_name:
            self.weight = input['weight'].to(self.device) + 1

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B, self.fake_depth = self.netG(self.real_A)  # G(A)

        self.fake_B = self.fake_B.clamp(0,1)
        self.fake_depth = self.fake_depth.clamp(0,1)

        if 'mask' in self.opt.model_name:
            
            self.fake_mask = self.fake_B[:,3:4]
            self.fake_B = self.fake_B[:,:3]

            # test1=self.fake_B*self.fake_mask
            # test2=torch.ones_like(self.fake_B) * (1-self.fake_mask)

            self.fake_B = self.fake_B * self.fake_mask + torch.ones_like(self.fake_B) * (1-self.fake_mask)
            self.fake_depth = self.fake_depth * self.fake_mask + torch.ones_like(self.fake_depth)*(1-self.fake_mask)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())

        if self.opt.netD !='multi_scale':
            pred_fake=[pred_fake]

        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

        if self.opt.D_normal:
            fake_A_normal = torch.cat((self.real_A,self.fake_depth),1)
            pred_fake_normal = self.netD_normal(fake_A_normal.detach())
            self.loss_D_normal_fake = self.criterionGAN(pred_fake_normal, False)
            real_A_normal = torch.cat((self.real_A,self.real_depth),1)
            pred_real_normal = self.netD_normal(real_A_normal)
            self.loss_D_normal_real = self.criterionGAN(pred_real_normal, True)
            self.loss_D_normal = (self.loss_D_normal_fake + self.loss_D_normal_real) * 0.5
            self.loss_D_normal.backward()



    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        if 'weighted' in self.opt.model_name:
            self.loss_G_L1 = self.criterionL1(self.fake_B*self.weight, self.real_B*self.weight) * self.opt.lambda_L1
            self.loss_G_depth = self.criterionL1(self.fake_depth*self.weight, self.real_depth*self.weight) * self.opt.lambda_depth

        else:
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
            self.loss_G_depth = self.criterionL1(self.fake_depth, self.real_depth) * self.opt.lambda_depth

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_depth

        if self.opt.D_normal:
            fake_A_normal = torch.cat((self.real_A,self.fake_depth),1)
            pred_fake_normal = self.netD_normal(fake_A_normal)
            self.loss_G_GAN_normal = self.criterionGAN(pred_fake_normal, True)
            self.loss_G += self.loss_G_GAN_normal

        if 'mask' in self.opt.model_name:
            
            thres_min = torch.nn.Threshold(0.01,0)
            thres_max = torch.nn.Threshold(-0.01,-1)
            self.fake_mask = thres_min(self.fake_mask)
            self.fake_mask = -thres_max(-self.fake_mask)
            # fake_B = self.fake_B.detach().cpu().numpy().squeeze()
            # fake_B = np.transpose(fake_B, (1, 2, 0))
            # fake_depth = self.fake_depth.detach().cpu().numpy().squeeze()
            # mask = self.fake_mask.detach().cpu().numpy().squeeze()
            # plt.subplot(131)
            # plt.imshow(mask)
            # plt.subplot(132)
            # plt.imshow(fake_B)
            # plt.subplot(133)
            # plt.imshow(fake_depth)
            # plt.show()
            # exit()
            self.loss_G_M = self.criterionL1(self.fake_mask, self.real_mask) * self.opt.lambda_mask
            self.loss_G += self.loss_G_M
        self.loss_G.backward()
        

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        if self.opt.D_normal:  
            self.set_requires_grad(self.netD_normal, True)
            self.optimizer_D_normal.zero_grad()

        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights

        if self.opt.D_normal:
            self.optimizer_D_normal.step()
            self.set_requires_grad(self.netD_normal,False)
              
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G

        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights





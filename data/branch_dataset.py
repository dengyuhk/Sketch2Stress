# coding=utf-8
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import os,cv2
from data.base_dataset import BaseDataset, get_params, get_transform
import argparse,random
from torchvision import transforms
from scipy.ndimage.morphology import distance_transform_edt

def augShift(mask,all_imgs):
    r, c = np.where(mask!=0)
    top, bottom = np.min(r), np.max(r)
    left, right = np.min(c), np.max(c)

    r_shift = random.randint(-top,255-bottom)
    c_shift = random.randint(-left,255-right)

    mat_shift = np.float32([[1,0,c_shift], [0,1,r_shift]])

    aug_imgs = []

    for i,img in enumerate(all_imgs):
        if i in [2,4]:
            aug_img = cv2.warpAffine(img, mat_shift, (256, 256),borderValue=(255,255,255))
        else:
            aug_img = cv2.warpAffine(img, mat_shift, (256, 256))
        aug_imgs.append(aug_img)

    return aug_imgs


class Branch_Dataset(Dataset):
    def __init__(self, opt, phase='train', transform=None,*args, **kwargs): ## root_path, phase
        super(Branch_Dataset, self).__init__(*args, **kwargs)
        self.root = opt.dataroot+'/'+opt.cate_name
        self.data_path= os.path.join(opt.dataroot.split('/train')[0], opt.cate_name, '256x256')
        self.phase = phase
        self.transform = transform
        
        self.use_DM = opt.use_DM#False#opt.use_DM
        self.use_aug =opt.use_aug# True #opt.use_aug
        self.useMask = False
        self.isNormal = False
        self.useWeighted = True if 'weighted' in opt.model_name else False

        if 'normal' in opt.netG:
            self.isNormal = True
        if 'mask' in opt.model_name:
            self.useMask = True

        self.train_val_test_paths= os.path.join(self.root,self.phase+'.txt')
        with open(self.train_val_test_paths, 'r') as fh:
            self.imgs_all = fh.readlines()
            print()

        self.force_nodes_all = [os.path.join(self.data_path,el.rstrip('\n')) for el in self.imgs_all if el.rstrip('\n').split(']')[-1] == '.png']
        if opt.max_data_size != 0:
            self.force_nodes_all = self.force_nodes_all[:opt.max_data_size]

        self.trans=transforms.Compose([
                transforms.ToTensor(),
            ])

    def __getitem__(self, index):

        force_mask_path = self.force_nodes_all[index]
        skecth_path=force_mask_path.split('-[')[0]+'.png'
        stress_path=force_mask_path.split('.png')[0]+'-stress.png'
        
        if self.isNormal:
            guidance_path=force_mask_path.split('sketch-[')[0]+'normal.png'
        else:
            guidance_path=force_mask_path.split('sketch-[')[0]+'depth.png'


        fm = np.array(Image.open(force_mask_path))
        sk = np.array(Image.open(skecth_path))
        st = np.array(Image.open(stress_path))



        if self.isNormal:
            guidance = np.array(Image.open(guidance_path))
            gray = cv2.cvtColor(guidance, cv2.COLOR_RGB2GRAY)
            mask = ((1-(gray==0))*255).astype('uint8')
            guidance[gray==0]=255


        else:
            guidance = cv2.imread(guidance_path,0)
            mask = ((1-(guidance==255))*255).astype('uint8')
            guidance[guidance==255]=0

        if self.use_aug:
            fm,sk,st,mask,guidance = augShift(mask,[fm,sk,st,mask,guidance])

        if self.use_DM:
            fm = distance_transform_edt(255-fm)
            fm = np.clip(fm, 0, 255).astype('uint8')

            # sk = distance_transform_edt(255-sk)
            # sk = np.clip(sk, 0, 255).astype('uint8')

            # plt.subplot(131)
            # plt.imshow(dist_fm)
            # plt.subplot(132)
            # plt.imshow(fm)
            # plt.subplot(133)
            # plt.imshow(dist_fm==0)
            # plt.show()

        if self.useWeighted:
            # plt.figure(1)
            # plt.subplot(221)
            # plt.imshow(fm)

            thres = 50
            dist = distance_transform_edt(255-fm)
            # plt.subplot(222)
            # plt.imshow(dist)

            dist = np.clip(dist, 0, thres)
            weight = ((thres-dist)/thres*255).astype('uint8')
            # plt.subplot(223)
            # plt.imshow(weight)
            # plt.show()
            
            weight = self.trans(weight[:,:,np.newaxis])


        fm = self.trans(fm[:,:,np.newaxis])
        sk = self.trans(sk[:,:,np.newaxis])
        st = self.trans(st)
        guidance = self.trans(guidance)
        mask = self.trans(mask[:,:,np.newaxis])

        # print(fm.shape,sk.shape)
        # exit()
        # plt.figure(1,figsize=(10,8))
        # plt.subplot(141)
        # plt.axis('off')
        # plt.imshow(sk.squeeze())
        # plt.subplot(142)
        # plt.axis('off')
        # plt.imshow(mask.squeeze())
        # plt.subplot(143)
        # plt.axis('off')
        # plt.imshow(np.transpose(st, (1, 2, 0)).squeeze())
        # plt.subplot(144)
        # plt.axis('off')
        # plt.imshow(np.transpose(guidance, (1, 2, 0)).squeeze())
        # plt.imshow(guidance.squeeze())
        # plt.show()
        inputs = {'sk':sk, 'force':fm, 'stress':st, 'guidance': guidance, 'img_path': force_mask_path}

        if self.useMask:
            inputs['mask'] = mask

        if self.useWeighted:
            inputs['weight'] = weight
        
        return inputs

    def __len__(self):
        num = len(self.force_nodes_all)
        return num


class Branch_Dataset_coordinates_input(Dataset):
    def __init__(self, opt, phase='train', transform=None, *args, **kwargs):  ## root_path, phase
        super(Branch_Dataset_coordinates_input, self).__init__(*args, **kwargs)
        self.root = opt.dataroot + '/' + opt.cate_name
        self.data_path = os.path.join(opt.dataroot.split('/train')[0], opt.cate_name, '256x256')
        self.phase = phase
        self.transform = transform

        self.use_DM = opt.use_DM  # False#opt.use_DM
        self.use_aug = opt.use_aug  # True #opt.use_aug
        self.useMask = False
        self.isNormal = False
        self.useWeighted = True if 'weighted' in opt.model_name else False

        if 'normal' in opt.netG:
            self.isNormal = True
        if 'mask' in opt.model_name:
            self.useMask = True

        self.train_val_test_paths = os.path.join(self.root, self.phase + '.txt')
        with open(self.train_val_test_paths, 'r') as fh:
            self.imgs_all = fh.readlines()

        self.force_nodes_all = [os.path.join(self.data_path, el.rstrip('\n')) for el in self.imgs_all if
                                el.rstrip('\n').split(']')[-1] == '.png']
        if opt.max_data_size != 0:
            self.force_nodes_all = self.force_nodes_all[:opt.max_data_size]

        self.trans = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):

        force_mask_path = self.force_nodes_all[index]
        skecth_path = force_mask_path.split('-[')[0] + '.png'
        stress_path = force_mask_path.split('.png')[0] + '-stress.png'

        if self.isNormal:
            guidance_path = force_mask_path.split('sketch-[')[0] + 'normal.png'
        else:
            guidance_path = force_mask_path.split('sketch-[')[0] + 'depth.png'

        fm = np.array(Image.open(force_mask_path))
        sk = np.array(Image.open(skecth_path))
        st = np.array(Image.open(stress_path))

        if self.isNormal:
            guidance = np.array(Image.open(guidance_path))
            gray = cv2.cvtColor(guidance, cv2.COLOR_RGB2GRAY)
            mask = ((1 - (gray == 0)) * 255).astype('uint8')
            guidance[gray == 0] = 255


        else:
            guidance = cv2.imread(guidance_path, 0)
            mask = ((1 - (guidance == 255)) * 255).astype('uint8')
            guidance[guidance == 255] = 0

        if self.use_aug:
            fm, sk, st, mask, guidance = augShift(mask, [fm, sk, st, mask, guidance])

        coordinates=np.where(fm==255)
        coordinates_center=np.array([[np.float(int((coordinates[0][0]+coordinates[0][-1])/2)),np.float(int((coordinates[1][0]+coordinates[1][-1])/2))]])
        coordinates_center_cat=np.concatenate((coordinates_center,coordinates_center),axis=0) #([coordinates_center,np.newaxis])

        # plt.imshow(fm)
        # plt.show()

        if self.use_DM:
            fm = distance_transform_edt(255 - fm)
            fm = np.clip(fm, 0, 255).astype('uint8')

            # sk = distance_transform_edt(255-sk)
            # sk = np.clip(sk, 0, 255).astype('uint8')

            # plt.subplot(131)
            # plt.imshow(dist_fm)
            # plt.subplot(132)
            # plt.imshow(fm)
            # plt.subplot(133)
            # plt.imshow(dist_fm==0)
            # plt.show()

        if self.useWeighted:
            # plt.figure(1)
            # plt.subplot(221)
            # plt.imshow(fm)

            thres = 50
            dist = distance_transform_edt(255 - fm)
            # plt.subplot(222)
            # plt.imshow(dist)

            dist = np.clip(dist, 0, thres)
            weight = ((thres - dist) / thres * 255).astype('uint8')
            # plt.subplot(223)
            # plt.imshow(weight)
            # plt.show()

            weight = self.trans(weight[:, :, np.newaxis])

        fm = self.trans(fm[:, :, np.newaxis])
        sk = self.trans(sk[:, :, np.newaxis])
        coordinates_center_cat=self.trans(coordinates_center_cat[:, :, np.newaxis])
        st = self.trans(st)
        guidance = self.trans(guidance)
        mask = self.trans(mask[:, :, np.newaxis])

        # print(fm.shape,sk.shape)
        # exit()
        # plt.figure(1,figsize=(10,8))
        # plt.subplot(141)
        # plt.axis('off')
        # plt.imshow(sk.squeeze())
        # plt.subplot(142)
        # plt.axis('off')
        # plt.imshow(mask.squeeze())
        # plt.subplot(143)
        # plt.axis('off')
        # plt.imshow(np.transpose(st, (1, 2, 0)).squeeze())
        # plt.subplot(144)
        # plt.axis('off')
        # plt.imshow(np.transpose(guidance, (1, 2, 0)).squeeze())
        # plt.imshow(guidance.squeeze())
        # plt.show()
        inputs = {'sk': sk, 'force': fm, 'stress': st, 'guidance': guidance, 'img_path': force_mask_path,'coordinates':coordinates_center_cat}

        if self.useMask:
            inputs['mask'] = mask

        if self.useWeighted:
            inputs['weight'] = weight

        return inputs

    def __len__(self):
        num = len(self.force_nodes_all)
        return num

if __name__ == '__main__':
    ##here is the test
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='',
                        help='path to images')
    parser.add_argument('--phase', type=str, default='train',
                        help='train or test')
    opt = parser.parse_args()

    data_path_temp = '/home/wanchao/new_drive/sk2fabrication/data/sketch_out/train_val_test'
    opt.dataroot = data_path_temp

    ##assingned data
    opt.cate_name = 'shapeAirplane'
    opt.model_name=  'branch_normal_mask_D_normal'#'branch_normal_mask_D_normal_weighted'
    opt.netG='branch_normal_coordinates_input'#'branch_normal'
    opt.max_data_size=1

    train_dataset =Branch_Dataset_coordinates_input(opt) #Branch_Dataset(opt)
    for i in range(20):
        sk, fm, st = train_dataset[i]
        sk, fm, st = np.transpose(sk, (1, 2, 0)), np.transpose(fm, (1, 2, 0)), np.transpose(st, (1, 2, 0))

        plt.figure(1, figsize=(10, 8))
        plt.subplot(131)
        plt.axis('off')
        plt.imshow(sk.squeeze())
        plt.subplot(132)
        plt.axis('off')
        plt.imshow(fm.squeeze())
        plt.subplot(133)
        plt.axis('off')
        plt.imshow(st.squeeze())
        plt.show()

    print()
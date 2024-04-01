# coding=utf-8
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import os
from data.base_dataset import BaseDataset, get_params, get_transform
import argparse
from torchvision import transforms

class dataset_sketch_force_mask_stress(Dataset):
    def __init__(self, opt, phase='train', transform=None,*args, **kwargs): ## root_path, phase
        super(dataset_sketch_force_mask_stress, self).__init__(*args, **kwargs)
        self.root = opt.dataroot+'/'+opt.cate_name
        self.data_path= os.path.join(opt.dataroot.split('/train')[0], opt.cate_name, '256x256')
        self.phase = phase
        self.transform = transform


        self.train_val_test_paths= os.path.join(self.root,self.phase+'.txt')
        with open(self.train_val_test_paths, 'r') as fh:
            self.imgs_all = fh.readlines()

        self.force_nodes_all = [os.path.join(self.data_path,el.rstrip('\n')) for el in self.imgs_all if el.rstrip('\n').split(']')[-1] == '.png']
        if opt.max_data_size != 0:
            self.force_nodes_all = self.force_nodes_all[:opt.max_data_size]


    def __getitem__(self, index):

        force_mask_path = self.force_nodes_all[index]
        skecth_path=force_mask_path.split('-[')[0]+'.png'
        stress_path=force_mask_path.split('.png')[0]+'-stress.png'


        fm = Image.open(force_mask_path)
        sk = Image.open(skecth_path)
        st = Image.open(stress_path)

        if self.transform is None:
            self.trans=transforms.Compose([
                transforms.ToTensor(),
            ])
            fm = self.trans(fm)
            sk = self.trans(sk)
            st = self.trans(st)

        else:
            # apply the same transform to both A and B
            transform_params = get_params(self.transform,fm.size)
            fm_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
            sk_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
            st_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

            fm = fm_transform
            sk = sk_transform
            st = st_transform

        # st = np.transpose(st, (1, 2, 0))

        # plt.figure(1,figsize=(10,8))
        # plt.subplot(131)
        # plt.axis('off')
        # plt.imshow(sk.squeeze())
        # plt.subplot(132)
        # plt.axis('off')
        # plt.imshow(fm.squeeze())
        # plt.subplot(133)
        # plt.axis('off')
        # plt.imshow(st.squeeze())
        # plt.show()

        return (sk, fm, st, force_mask_path)

    def __len__(self):
        num = len(self.force_nodes_all)
        return num


if __name__ == '__main__':
    parser= argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='',
                        help='path to images')
    parser.add_argument('--phase', type=str, default='train',
                        help='train or test')
    opt=parser.parse_args()

    data_path_temp= '../data/sketch_out/train_val_test'
    opt.dataroot=data_path_temp
    opt.cate_name='psbAirplane'

    train_dataset=dataset_sketch_force_mask_stress(opt)
    for i in range(20):
        sk,fm,st = train_dataset[i]
        sk,fm,st = np.transpose(sk, (1, 2, 0)),np.transpose(fm, (1, 2, 0)), np.transpose(st, (1, 2, 0))

        plt.figure(1,figsize=(10,8))
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



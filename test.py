##created by deng
from models.pix2pix_branch import Pix2PixModel
import time
from data.branch_dataset import Branch_Dataset
import torch
from options.testOptions import TestOptions
import os
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np
from utils import html
from utils.visualizer import save_images

def tensor2numpy(a):
    return (a.cpu().numpy()*255).astype("uint8")
    

###todo:
if __name__ == '__main__':

    ##dataset
    opt = TestOptions().parse()   # get training options

    test_dataset = Branch_Dataset(opt,phase='test')
    dataset_size = len(test_dataset)  # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    ## dataloader
    test_loader= torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size,
                                               shuffle=False,
                                               num_workers=4)
    ## model
    model= Pix2PixModel(opt)
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    ##=====================================================================================
    ## test
    ##=====================================================================================
    res_dir = os.path.join('./results', opt.cate_name, opt.model_name,opt.epoch)
    webpage = html.HTML(res_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.model_name, opt.phase, opt.epoch))
    os.makedirs(res_dir,exist_ok=True)
    if opt.eval:
        model.eval()
    with torch.no_grad():
        tensor_to_pil = transforms.ToPILImage()
        im_name=0
        for i, test_data in enumerate(test_loader):  # inner loop within one epoch
            # time1 = time.time()
            model.set_input(test_data)         # unpack data from dataset and apply preprocessing
            model.test()

            ## visulize generated results

            model.compute_visuals()
            current_results=model.get_current_visuals()
            # time2=time.time()

            # print('data amount:{}, time:{}'.format(len(test_data['sk']),time2-time1))
            img_paths = model.get_image_paths()

            for j in range(len(current_results['real_A'])):
                ## sketch
                sk_force = current_results['real_A'][j,:2]

                ## generated stress map
                stress_fake = current_results['fake_B'][j]

                ## gt stress map
                stress_gt = current_results['real_B'][j]

                depth_syn = current_results['fake_depth'][j]
                depth_gt = current_results['real_depth'][j]

                visuals = {"Sketch+Force":sk_force,"Stress":stress_fake,"Stress_GT":stress_gt,"Depth_syn":depth_syn,"Depth_GT":depth_gt}
                if 'mask' in opt.model_name:
                    visuals['Mask_syn'] = current_results['fake_mask'][j]
                    visuals['Mask_GT'] = current_results['real_mask'][j]


                # print(sk.shape,stress_gt.shape)
                # # input = np.ones_like(stress_gt)*255
                # # input[sk==255] = 0
                # plt.imshow(stress_fake[1])
                # plt.show()
                # ## display the syntheiszed
                # plt.figure(1, figsize=(12, 10))
                # plt.subplot(221)
                # plt.title('sketch')
                # plt.axis('off')
                # plt.imshow(sk)
                # plt.subplot(222)
                # plt.title('force_nodes')
                # plt.axis('off')
                # plt.imshow(sk_force_node)
                # plt.subplot(223)
                # plt.title('generated_stress_map')
                # plt.axis('off')
                # plt.imshow(stress_fake)
                # plt.subplot(224)
                # plt.title('gt_stress_map')
                # plt.axis('off')
                # plt.imshow(stress_gt)
                # plt.show()

                ##save to disk
                # sk_name = '%s_%s.png' % (im_name,'sk')
                # sk_save_path = os.path.join(res_dir, sk_name)
                # sk.save(sk_save_path)
                #
                # fn_name = '%s_%s.png' % (im_name,'fn')
                # fn_save_path = os.path.join(res_dir, fn_name)
                # sk_force_node.save(fn_save_path)

                # stf_name = '%s_%s.png' % (im_name,'stf' )
                # stf_save_path = os.path.join(res_dir, stf_name)
                # stress_fake.save(stf_save_path)

                # stgt_name = '%s_%s.png' % (im_name,'stgt')
                # stgt_save_path = os.path.join(res_dir, stgt_name)
                # stress_gt.save(stgt_save_path)

                im_name=im_name+1
                print('[{}/{}]..........................'.format(im_name,dataset_size))
                save_images(webpage, visuals, img_paths[j], opt,aspect_ratio=1, width=256)
    webpage.save()  # save the HTML











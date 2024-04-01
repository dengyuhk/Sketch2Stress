##created by chufeng for joint training
import time
import sys
sys.path.append('models')
sys.path.append('dataset')
from models.pix2pix_branch import Pix2PixModel
from data.branch_dataset import Branch_Dataset
import torch
from options.trainOptions import TrainOptions
from tensorboardX import SummaryWriter
from utils import util
import os
import torchvision
from utils.visualizer import Visualizer

def get_results(opt, current_results):
    sk_force = current_results['real_A'][0,:2]
    stress_fake = current_results['fake_B'][0]

    stress_gt = current_results['real_B'][0]

    depth_syn = current_results['fake_depth'][0]
    depth_gt = current_results['real_depth'][0]

    visuals = {"Sketch+Force":sk_force,"Stress":stress_fake,"Stress_GT":stress_gt,"Depth_syn":depth_syn,"Depth_GT":depth_gt}
    if 'mask' in opt.model_name:
        visuals['Mask_syn'] = current_results['fake_mask'][0]
        visuals['Mask_GT'] = current_results['real_mask'][0]
    return visuals

if __name__ == '__main__':

    ##dataset
    ## python train_branch.py --batch_size 16 --gpu_ids 0 --model_name 'branch_normal_mask_D_normal' --lr_policy 'step' --n_epochs 2 --n_epochs_decay 2 --lr_decay_iters 20 --gan_mode 'l1' --netD 'multi_scale' --num_D 3 --netG 'branch_normal' --output_nc 3 --save_epoch_freq 1 --save_image_freq 1 --D_normal --cate_name 'shapeChairnew' --use_aug
    opt = TrainOptions().parse()   # get training options

    #for debuggging
    # opt.cate_name='shapeAirplane'
    # opt.model_name='branch_normal_mask_D_normal_weighted'
    # opt.batch_size=16
    # opt.lr_policy='step'
    # opt.n_epochs=2
    # opt.n_epochs_decay=2
    # opt.lr_decay_iters=20
    # opt.gan_mode='l1'
    # opt.netD='multi_scale'
    # opt.num_D=3
    # opt.netG='branch_normal'
    # opt.output_nc=4
    # opt.save_epoch_freq=1
    # opt.save_image_freq=1
    # opt.D_normal=True
    # opt.use_aug=True


    train_dataset = Branch_Dataset(opt,phase='train')
    test_dataset = Branch_Dataset(opt,phase='test')
    dataset_size = len(train_dataset)  # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    ## dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
                                               shuffle=True ,
                                               num_workers=4)
    test_loader= torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size,
                                               shuffle=False,
                                               num_workers=4)
    ## model
    model= Pix2PixModel(opt)
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    ## visualizer
    visualizer = Visualizer(opt)
    tensorboard_log_dir=os.path.join(opt.checkpoints_dir, opt.cate_name, opt.model_name,"logs")
    if not os.path.exists(tensorboard_log_dir):util.mkdir(tensorboard_log_dir)
    writer = SummaryWriter(logdir=tensorboard_log_dir)


    ##
    save_loss =1e20
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch

        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        loss_sum_G=[]
        loss_sum_G_l1 = []
        loss_sum_G_depth = []
        loss_sum_D = []
        loss_sum_M = []

        ##=====================================================================================
        ## training
        ##=====================================================================================

        for i, data in enumerate(train_loader):  # inner loop within one epoch

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            ## visulize generated results
            model.compute_visuals()
            current_results=model.get_current_visuals()

            losses= model.get_current_losses()
            loss_G_L1 =losses['G_L1']
            loss_G_depth = losses['G_depth']

            loss_G = losses['G_GAN'] + losses['G_L1'] + loss_G_depth

            if 'mask' in opt.model_name:
                loss_G_M = losses['G_M']
                loss_G += loss_G_M
                loss_sum_M.append(loss_G_M)

            loss_D=  (losses['D_real'] + losses['D_fake'])*0.5

            if i % opt.train_display_interval == 0:
                if 'mask' in opt.model_name:
                    print('Train Epoch: {} iters:{} Loss: G:{:.6f},      G_L1:{:.6f},      G_depth:{:.6f},       D:{:.6f}, G_M:{:.6f}'.format(epoch,i,loss_G,loss_G_L1,loss_G_depth,loss_D,loss_G_M))
                else:
                    print('Train Epoch: {} iters:{} Loss: G:{:.6f},      G_L1:{:.6f},      G_depth:{:.6f},       D:{:.6f}'.format(epoch, i, loss_G,loss_G_L1,loss_G_depth,loss_D))

            loss_sum_G_l1.append(loss_G_L1)
            loss_sum_G_depth.append(loss_G_depth)
            loss_sum_G.append(loss_G)
            loss_sum_D.append(loss_D)

        if epoch % opt.save_image_freq == 0:
            visuals = get_results(opt, current_results)
            visualizer.save_current_results(visuals, epoch)

        avg_l1 = sum(loss_sum_G_l1) / len(loss_sum_G_l1)
        avg_depth = sum(loss_sum_G_depth) / len(loss_sum_G_depth)
        avg_G = sum(loss_sum_G) / len(loss_sum_G)
        avg_D = sum(loss_sum_D) / len(loss_sum_D)
        if 'mask' in opt.model_name:
            avg_M = sum(loss_sum_M) / len(loss_sum_M)
            writer.add_scalar('train_G_M_loss', avg_M, epoch)

        writer.add_scalar('train_G_L1_loss', avg_l1, epoch)
        writer.add_scalar('train_G_depth', avg_depth, epoch)
        writer.add_scalar('train_G_loss', avg_G, epoch)
        writer.add_scalar('train_D_loss', avg_D, epoch)
        print('Train Stage: {} \tAvg Loss: G:{:.6f},    G_L1:{:.6f},    G_depth:{:.6f},    D:{:.6f}'.format(epoch, avg_G, avg_l1, avg_depth, avg_D))

        ##=====================================================================================
        ## test
        ##=====================================================================================
        with torch.no_grad():
            test_loss_sum_G = []
            test_loss_sum_G_l1 = []
            test_loss_sum_G_depth = []
            test_loss_sum_D = []
            for i, test_data in enumerate(test_loader):  # inner loop within one epoch

                model.set_input(test_data)         # unpack data from dataset and apply preprocessing
                model.forward()
                # model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

                ## visulize generated results
                model.compute_visuals()
                current_results=model.get_current_visuals()

                losses= model.get_current_losses()
                loss_G_L1 =losses['G_L1']
                loss_G_depth = losses['G_depth']
                loss_G = losses['G_GAN'] + losses['G_L1'] + loss_G
                loss_D =  (losses['D_real'] + losses['D_fake'])*0.5

                test_loss_sum_G_l1.append(loss_G_L1)
                test_loss_sum_G.append(loss_G)
                test_loss_sum_G_depth.append(loss_G_depth)
                test_loss_sum_D.append(loss_D)

            test_avg_l1 = sum(test_loss_sum_G_l1) / len(test_loss_sum_G_l1)
            test_avg_G = sum(test_loss_sum_G) / len(test_loss_sum_G)
            test_avg_depth = sum(test_loss_sum_G_depth) / len(test_loss_sum_G_depth)
            test_avg_D = sum(test_loss_sum_D) / len(test_loss_sum_D)

            writer.add_scalar('test_G_L1_loss', test_avg_l1, epoch)
            writer.add_scalar('test_G_depth_loss', test_avg_depth, epoch)
            writer.add_scalar('test_G_loss', test_avg_G, epoch)
            writer.add_scalar('test_D_loss', test_avg_D, epoch)
            print('Test Stage: {} \tAvg Loss: G:{:.6f},     G_L1:{:.6f},    G_depth:{:.6f},    D:{:.6f}'.format(epoch, test_avg_G, test_avg_l1, test_avg_depth, test_avg_D))
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs+ opt.n_epochs_decay, time.time() - epoch_start_time))

        if epoch % opt.save_image_freq == 0:
            visuals = get_results(opt, current_results)
            visualizer.save_current_results(visuals, "T%d"%epoch)
        if epoch == (opt.n_epochs + opt.n_epochs_decay) or epoch % opt.save_epoch_freq == 0:
            model.save_networks(epoch)

        
        

    writer.close()
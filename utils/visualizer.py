
import numpy as np
import os,cv2
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE
import matplotlib.pyplot as plt

def save_images(webpage, visuals, image_path, opt, aspect_ratio=1.0, width=256):

    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path)
    name = os.path.splitext(short_path)[0]
    
    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = getImage(label, im_data, opt)

        image_name = '%s_%s.png' % (name, label)
    
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    
    webpage.add_images(ims, txts, links, width=width)

def getImage(label, im_data, opt):
    if label == "Sketch+Force":
        force = util.tensor2im(im_data[1:2])

        if opt.use_DM:
            force = (force == 0)*255
        sk = util.tensor2im(im_data[:1])
        

        im = 255-sk
        im[force[:,:,0]==255]=[255,0,0]
        
        
    else:
        im = util.tensor2im(im_data)
    return im



class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        self.opt = opt  # cache the option

        self.name = opt.model_name

        self.saved = False
        self.save_epochs = [] 

        self.web_dir = os.path.join(opt.checkpoints_dir, opt.cate_name,opt.model_name)

        self.img_dir = os.path.join(self.web_dir, 'images')
        print('create web directory %s...' % self.web_dir)
        util.mkdirs([self.web_dir, self.img_dir])

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def save_current_results(self, visuals, epoch):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """

        self.saved = True
        epoch = str(epoch)
        # save images to the disk
        for label, image in visuals.items():
            image_numpy = getImage(label,image, self.opt)
            
            img_path = os.path.join(self.img_dir, 'epoch_%s_%s.png' % (epoch, label))
            util.save_image(image_numpy, img_path)

        self.save_epochs.append(epoch)

        # update website
        webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=0)
        
        for n_i in range(len(self.save_epochs)-1,-1,-1):
            n = self.save_epochs[n_i]
            webpage.add_header('epoch [%s]' % n)
            
            ims, txts, links = [], [], []

            for label, image_numpy in visuals.items():
                img_path = 'epoch_%s_%s.png' % (n, label)
                ims.append(img_path)
                txts.append(label)
                links.append(img_path)
            webpage.add_images(ims, txts, links, width=256)
        webpage.save()



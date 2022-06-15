import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from scipy.io import loadmat

import networks
import tta_loss as loss
import tta_util as util
from torch_sobel import Sobel
from tta_model.get_model import get_model
from tta_model.network_swinir import define_model


class TTASR_Base:

    def __init__(self, conf):
        # Fix random seed
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True # slightly reduces throughput

        # Acquire configuration
        self.conf = conf

        # Define the networks
        # 1. Define and Load the pretrained swinir
        self.G_UP = get_model(conf)
        self.D_DN = networks.Discriminator_DN().cuda()
        # 2. Define the down sample network
        self.G_DN = networks.Generator_DN().cuda()

        # Losses
        self.criterion_gan = loss.GANLoss().cuda()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_interp = torch.nn.L1Loss()
        self.regularization = loss.DownsamplerRegularization(conf.scale_factor_downsampler, self.G_DN.G_kernel_size)

        # Initialize networks weights
        self.D_DN.apply(networks.weights_init_D_DN)

        # Optimizers
        self.optimizer_G_UP = torch.optim.Adam(self.G_UP.parameters(), lr=conf.lr_G_UP, betas=(conf.beta1, 0.999))
        self.optimizer_D_DN = torch.optim.Adam(self.D_DN.parameters(), lr=conf.lr_D_DN, betas=(conf.beta1, 0.999))
        self.optimizer_G_DN = torch.optim.Adam(self.G_DN.parameters(), lr=conf.lr_G_DN, betas=(conf.beta1, 0.999))


        # TODO: below need to rewrite
        # Read input image
        self.read_image(self.conf)


        # debug variables
        self.debug_steps = []
        self.UP_psnrs = [] if self.gt_img is not None else None
        self.DN_psnrs = [] if self.gt_kernel is not None else None

        if self.conf.debug:
            self.loss_GANs = []
            self.loss_cycle_forwards = []
            self.loss_cycle_backwards = []
            self.loss_interps = []
            self.loss_Discriminators = []

        self.iter = 0

        self.train_G_DN_switch = False
        self.train_G_UP_switch = False
        self.train_D_DN_switch = True
        
        self.reshap_train_data = False

    def set_input(self, data):
        '''
        get image for training.
        '''
        self.real_HR = data['HR'].cuda()
        self.real_LR = data['LR'].cuda()
        
        if self.reshap_train_data:
            self.real_HR = self.real_HR.reshape([self.real_HR.size(0)*self.real_HR.size(1), self.real_HR.size(2), self.real_HR.size(3), self.real_HR.size(4)])
            self.real_LR = self.real_LR.reshape([self.real_LR.size(0)*self.real_LR.size(1), self.real_LR.size(2), self.real_LR.size(3), self.real_LR.size(4)])

    def read_image(self, conf):
        '''
        Read image for testing.
        '''
        
        if conf.input_image_path:
            self.in_img = util.read_image(conf.input_image_path)
            self.in_img_t= util.im2tensor(self.in_img).cuda()
            b_x = self.in_img_t.shape[2] % conf.scale_factor
            b_y = self.in_img_t.shape[3] % conf.scale_factor
            self.in_img_cropped_t = self.in_img_t[..., b_x:, b_y:]
        
        self.gt_img = util.read_image(conf.gt_path) if conf.gt_path is not None else None
        self.gt_kernel = loadmat(conf.kernel_path)['Kernel'] if conf.kernel_path is not None else None
        self.UP_psnrs = [] if self.gt_img is not None else None
        self.DN_psnrs = [] if self.gt_kernel is not None else None

    def eval(self, iteration, save_sr=False):
        self.quick_eval()
        if self.conf.debug:
            self.plot()

        if save_sr:
            plt.imsave(os.path.join(self.conf.visual_dir, f"upsampled_img_{self.conf.abs_img_name}_{iteration+1}.png"), self.upsampled_img)
            plt.imsave(os.path.join(self.conf.visual_dir, f"downsampled_img_{self.conf.abs_img_name}_{iteration+1}.png"), self.downsampled_img)

        if self.gt_img is not None:
            print('Upsampler PSNR = ', self.UP_psnrs[-1])
        if self.gt_kernel is not None:
            print("Downsampler PSNR = ", self.DN_psnrs[-1])
        print('*' * 60 + '\nOutput is saved in \'%s\' folder\n' % self.conf.visual_dir)
        plt.close('all')


    def quick_eval(self):
        # Evaluate trained upsampler and downsampler on input data
        with torch.no_grad():
            downsampled_img_t = self.G_DN(self.in_img_cropped_t)

            self.G_UP.eval()
            # NOTE: this code comes from SwinIR
            # pad input image to be a multiple of window_size
            if self.conf.source_model == "swinir":
                window_size = 8
                _, _, h_old, w_old = self.in_img_t.size()
                in_img_t = self.in_img_t.clone()
                h_pad = (h_old // window_size + 1) * window_size - h_old
                w_pad = (w_old // window_size + 1) * window_size - w_old
                in_img_t = torch.cat([in_img_t, torch.flip(in_img_t, [2])], 2)[:, :, :h_old + h_pad, :]
                in_img_t = torch.cat([in_img_t, torch.flip(in_img_t, [3])], 3)[:, :, :, :w_old + w_pad]
                
                upsampled_img_t = self.G_UP(in_img_t)
                upsampled_img_t = upsampled_img_t[..., :h_old * self.conf.scale_factor, :w_old * self.conf.scale_factor]

            else:
                in_img_t = self.in_img_t
                upsampled_img_t = self.G_UP(in_img_t)


        self.downsampled_img = util.tensor2im(downsampled_img_t)
        self.upsampled_img = util.tensor2im(upsampled_img_t)
        
        if self.gt_kernel is not None:
            self.DN_psnrs += [util.cal_y_psnr(self.downsampled_img, self.gt_downsampled_img, border=self.conf.scale_factor)]
        if self.gt_img is not None:
            # import ipdb; ipdb.set_trace()
            _, _, h_old, w_old = self.in_img_t.size()
            self.UP_psnrs += [util.cal_y_psnr(self.upsampled_img, self.gt_img[:h_old * self.conf.scale_factor, :w_old * self.conf.scale_factor, ...], border=self.conf.scale_factor)]
        self.debug_steps += [self.iter]



    
    def save_model(self, iteration):

        torch.save(self.G_UP, os.path.join(self.conf.model_save_dir, f"ckpt_GUP_{iteration+1}.pth"))
        torch.save(self.G_UP, os.path.join(self.conf.model_save_dir, f"ckpt_GDN_{iteration+1}.pth"))








class DualPath(TTASR_Base):
    '''
    In this mehod, we implement our most basic idea in which we train the G_DN
    and G_UP
    '''
    def __init__(
        self,
        conf
    ):
        super().__init__(conf)


    def train(self, data):
        self.set_input(data)

        loss = {}

        # Train G_DN
        self.G_UP.eval()
        # Turn off gradient calculation for G_UP
        util.set_requires_grad([self.G_UP], False)
        # # Turn on gradient calculation for G_DN
        util.set_requires_grad([self.G_DN], True)
        util.set_requires_grad([self.D_DN], False)
        loss_train_G_DN = self.train_G_DN()

        # Train G_UP
        self.G_UP.train()
        # # Turn on gradient calculation for G_UP
        util.set_requires_grad([self.G_UP], True)
        # # Turn off gradient calculation for G_DN
        # util.set_requires_grad([self.G_DN], True)
        util.set_requires_grad([self.G_DN], False)
        # # Turn off gradient calculation for D_DN
        loss_train_G_UP = self.train_G_UP()

        # Train D_DN
        # Turn on gradient calculation for discriminator
        util.set_requires_grad([self.D_DN], True)
        loss_train_D_DN = self.train_D_DN()

        loss.update(loss_train_G_DN)
        loss.update(loss_train_G_UP)
        loss.update(loss_train_D_DN)



        if self.conf.debug:
            if self.iter % self.conf.eval_iters == 0:
                self.quick_eval()
            if self.iter % self.conf.plot_iters == 0:
                self.plot()
        self.iter = self.iter + 1

        return loss

    def train_G_DN(self):
        loss_cycle_forward = 0
        loss_cycle_backward = 0
        total_loss = 0
        loss_GAN = 0
        self.loss_regularization = 0

        if self.train_G_DN_switch:

            # Reset gradient valus
            # self.optimizer_G_UP.zero_grad()
            self.optimizer_G_DN.zero_grad()

            # Forward path
            self.fake_HR = self.G_UP(self.real_LR)
            self.rec_LR = self.G_DN(self.fake_HR)
            # Backward path
            self.fake_LR = self.G_DN(self.real_HR)
            self.rec_HR = self.G_UP(self.fake_LR)

            # Losses
            loss_GAN = self.criterion_gan(self.D_DN(self.fake_LR), True)
            loss_cycle_forward = self.criterion_cycle(self.rec_LR, util.shave_a2b(self.real_LR, self.rec_LR)) #* self.conf.lambda_cycle
            loss_cycle_backward = self.criterion_cycle(self.rec_HR, util.shave_a2b(self.real_HR, self.rec_HR)) #* self.conf.lambda_cycle

            total_loss = loss_cycle_forward + loss_cycle_backward + loss_GAN + self.loss_regularization
            # total_loss = loss_cycle_forward + loss_cycle_backward + self.loss_regularization

            total_loss.backward()

            # self.optimizer_G_UP.step()
            self.optimizer_G_DN.step()

        return {
            "train_G_DN/loss_cycle_forward": loss_cycle_forward, 
            "train_G_DN/loss_cycle_backward":loss_cycle_backward,
            "train_G_DN/total_loss":total_loss,
            "train_G_DN/loss_GAN": loss_GAN,
            "train_G_DN/loss_regularization": self.loss_regularization,
            }

    def train_G_UP(self):
        loss_cycle_forward = 0
        loss_cycle_backward = 0
        total_loss = 0
        loss_interp = 0
        
        if self.train_G_UP_switch:

            # Rese gradient valus
            self.optimizer_G_UP.zero_grad()

            # Forward path
            self.fake_HR = self.G_UP(self.real_LR)
            self.rec_LR = self.G_DN(self.fake_HR)
            # Backward path
            self.fake_LR = self.G_DN(self.real_HR)
            self.rec_HR = self.G_UP(self.fake_LR)

            # Losses
            loss_cycle_forward = self.criterion_cycle(self.rec_LR, util.shave_a2b(self.real_LR, self.rec_LR)) #* self.conf.lambda_cycle
            loss_cycle_backward = self.criterion_cycle(self.rec_HR, util.shave_a2b(self.real_HR, self.rec_HR)) #* self.conf.lambda_cycle
            # self.fake_LR = self.G_DN(self.real_LR)
            # self.rec_HR = self.G_UP(self.fake_LR)
            # loss_cycle_backward = self.criterion_cycle(self.rec_HR, util.shave_a2b(self.real_LR, self.rec_HR)) * self.conf.lambda_cycle

            total_loss = loss_cycle_forward + loss_cycle_backward + loss_interp

            total_loss.backward()

            # Update weights
            self.optimizer_G_UP.step()

        return {
            "train_G_UP/loss_cycle_forward": loss_cycle_forward, 
            "train_G_UP/loss_cycle_backward":loss_cycle_backward,
            "train_G_UP/total_loss":total_loss,
            "train_G_UP/loss_interp":loss_interp,
            }

    def train_D_DN(self):
        # Rese gradient valus
        self.loss_Discriminator = 0
        
        if self.train_D_DN_switch:
            self.optimizer_D_DN.zero_grad()
            
            # Fake
            pred_fake = self.D_DN(self.fake_LR.detach())
            loss_D_fake = self.criterion_gan(pred_fake, False)
            # Real
            pred_real = self.D_DN(util.shave_a2b(self.real_LR, self.fake_LR))
            loss_D_real = self.criterion_gan(pred_real, True)
            # Combined loss and calculate gradients
            self.loss_Discriminator = (loss_D_real + loss_D_fake) * 0.5
            self.loss_Discriminator.backward()

            # Update weights
            self.optimizer_D_DN.step()

        return {
            "train_D_DN/loss_Discriminator": self.loss_Discriminator
        }




class BicubicGDN(TTASR_Base):
    '''
    In this mehod, we implement our most basic idea in which we train the G_DN
    and G_UP
    '''
    def __init__(
        self,
        conf
    ):
        super().__init__(conf)


    def train(self, data):
        self.set_input(data)

        loss = {}

        # Train G_UP
        self.G_UP.train()
        # # Turn on gradient calculation for G_UP
        util.set_requires_grad([self.G_UP], True)

        loss_train_G_UP = self.train_G_UP()
        loss.update(loss_train_G_UP)

        return loss


    def train_G_UP(self):
        loss_cycle_forward = 0
        loss_cycle_backward = 0
        total_loss = 0
        
        if self.train_G_UP_switch:

            # Rese gradient valus
            self.optimizer_G_UP.zero_grad()

            # Forward path
            # self.fake_HR = self.G_UP(self.real_LR)
            # self.rec_LR = torch.nn.functional.interpolate(input=self.fake_HR, scale_factor=0.5, mode='bicubic')
            # self.rec_LR = self.G_DN(self.fake_HR)
            # Backward path
            # self.fake_LR = self.G_DN(self.real_HR)
            self.fake_LR = torch.nn.functional.interpolate(input=self.real_HR, scale_factor=0.5, mode='bicubic')
            self.rec_HR = self.G_UP(self.fake_LR)

            # Losses
            # loss_cycle_forward = self.criterion_cycle(self.rec_LR, util.shave_a2b(self.real_LR, self.rec_LR)) #* self.conf.lambda_cycle
            loss_cycle_backward = self.criterion_cycle(self.rec_HR, util.shave_a2b(self.real_HR, self.rec_HR)) #* self.conf.lambda_cycle

            total_loss = loss_cycle_forward + loss_cycle_backward

            total_loss.backward()

            # Update weights
            self.optimizer_G_UP.step()

        return {
            "train_G_UP/loss_cycle_forward": loss_cycle_forward, 
            "train_G_UP/loss_cycle_backward":loss_cycle_backward,
            "train_G_UP/total_loss":total_loss,
            }





class BackwardPathDDN(TTASR_Base):
    '''
    In this mehod, we implement our most basic idea in which we train the G_DN
    and G_UP
    '''
    def __init__(
        self,
        conf
    ):
        super().__init__(conf)


    def train(self, data):
        self.set_input(data)

        loss = {}

        # Train G_DN
        self.G_UP.eval()
        # Turn off gradient calculation for G_UP
        util.set_requires_grad([self.G_UP], False)
        # # Turn on gradient calculation for G_DN
        util.set_requires_grad([self.G_DN], True)
        util.set_requires_grad([self.D_DN], False)
        loss_train_G_DN = self.train_G_DN()

        # Train G_UP
        self.G_UP.train()
        # # Turn on gradient calculation for G_UP
        util.set_requires_grad([self.G_UP], True)
        # # Turn off gradient calculation for G_DN
        # util.set_requires_grad([self.G_DN], True)
        util.set_requires_grad([self.G_DN], False)
        # # Turn off gradient calculation for D_DN
        loss_train_G_UP = self.train_G_UP()

        # Train D_DN
        # Turn on gradient calculation for discriminator
        util.set_requires_grad([self.D_DN], True)
        loss_train_D_DN = self.train_D_DN()

        loss.update(loss_train_G_DN)
        loss.update(loss_train_G_UP)
        loss.update(loss_train_D_DN)



        if self.conf.debug:
            if self.iter % self.conf.eval_iters == 0:
                self.quick_eval()
            if self.iter % self.conf.plot_iters == 0:
                self.plot()
        self.iter = self.iter + 1

        return loss

    def train_G_DN(self):
        loss_cycle_forward = 0
        loss_cycle_backward = 0
        total_loss = 0
        loss_GAN = 0
        self.loss_regularization = 0

        if self.train_G_DN_switch:

            # Reset gradient valus
            # self.optimizer_G_UP.zero_grad()
            self.optimizer_G_DN.zero_grad()

            # Forward path
            # self.fake_HR = self.G_UP(self.real_LR)
            # self.rec_LR = self.G_DN(self.fake_HR)
            # Backward path
            self.fake_LR = self.G_DN(self.real_HR)
            self.rec_HR = self.G_UP(self.fake_LR)

            # Losses
            loss_GAN = self.criterion_gan(self.D_DN(self.fake_LR), True)
            # loss_cycle_forward = self.criterion_cycle(self.rec_LR, util.shave_a2b(self.real_LR, self.rec_LR)) #* self.conf.lambda_cycle
            loss_cycle_backward = self.criterion_cycle(self.rec_HR, util.shave_a2b(self.real_HR, self.rec_HR)) #* self.conf.lambda_cycle

            total_loss = loss_cycle_forward + loss_cycle_backward + loss_GAN + self.loss_regularization
            # total_loss = loss_cycle_forward + loss_cycle_backward + self.loss_regularization

            total_loss.backward()

            # self.optimizer_G_UP.step()
            self.optimizer_G_DN.step()

        return {
            "train_G_DN/loss_cycle_forward": loss_cycle_forward, 
            "train_G_DN/loss_cycle_backward":loss_cycle_backward,
            "train_G_DN/total_loss":total_loss,
            "train_G_DN/loss_GAN": loss_GAN,
            "train_G_DN/loss_regularization": self.loss_regularization,
            }

    def train_G_UP(self):
        loss_cycle_forward = 0
        loss_cycle_backward = 0
        total_loss = 0
        loss_interp = 0
        
        if self.train_G_UP_switch:

            # Rese gradient valus
            self.optimizer_G_UP.zero_grad()

            # Forward path
            # self.fake_HR = self.G_UP(self.real_LR)
            # self.rec_LR = self.G_DN(self.fake_HR)
            # Backward path
            self.fake_LR = self.G_DN(self.real_HR)
            self.rec_HR = self.G_UP(self.fake_LR)

            # Losses
            # loss_cycle_forward = self.criterion_cycle(self.rec_LR, util.shave_a2b(self.real_LR, self.rec_LR)) #* self.conf.lambda_cycle
            loss_cycle_backward = self.criterion_cycle(self.rec_HR, util.shave_a2b(self.real_HR, self.rec_HR)) #* self.conf.lambda_cycle
            # self.fake_LR = self.G_DN(self.real_LR)
            # self.rec_HR = self.G_UP(self.fake_LR)
            # loss_cycle_backward = self.criterion_cycle(self.rec_HR, util.shave_a2b(self.real_LR, self.rec_HR)) * self.conf.lambda_cycle

            total_loss = loss_cycle_forward + loss_cycle_backward + loss_interp

            total_loss.backward()

            # Update weights
            self.optimizer_G_UP.step()

        return {
            "train_G_UP/loss_cycle_forward": loss_cycle_forward, 
            "train_G_UP/loss_cycle_backward":loss_cycle_backward,
            "train_G_UP/total_loss":total_loss,
            "train_G_UP/loss_interp":loss_interp,
            }

    def train_D_DN(self):
        # Rese gradient valus
        self.loss_Discriminator = 0
        
        if self.train_D_DN_switch:
            self.optimizer_D_DN.zero_grad()
            
            # Fake
            pred_fake = self.D_DN(self.fake_LR.detach())
            loss_D_fake = self.criterion_gan(pred_fake, False)
            # Real
            pred_real = self.D_DN(util.shave_a2b(self.real_LR, self.fake_LR))
            loss_D_real = self.criterion_gan(pred_real, True)
            # Combined loss and calculate gradients
            self.loss_Discriminator = (loss_D_real + loss_D_fake) * 0.5
            self.loss_Discriminator.backward()

            # Update weights
            self.optimizer_D_DN.step()

        return {
            "train_D_DN/loss_Discriminator": self.loss_Discriminator
        }











class BackwardPath(TTASR_Base):
    '''
    In this mehod, we implement our most basic idea in which we train the G_DN
    and G_UP
    '''
    def __init__(
        self,
        conf
    ):
        super().__init__(conf)


    def train(self, data):
        self.set_input(data)

        loss = {}

        # Train G_DN
        self.G_UP.eval()
        # Turn off gradient calculation for G_UP
        util.set_requires_grad([self.G_UP], False)
        # # Turn on gradient calculation for G_DN
        util.set_requires_grad([self.G_DN], True)
        # util.set_requires_grad([self.D_DN], False)
        loss_train_G_DN = self.train_G_DN()

        # Train G_UP
        self.G_UP.train()
        # # Turn on gradient calculation for G_UP
        util.set_requires_grad([self.G_UP], True)
        # # Turn off gradient calculation for G_DN
        # util.set_requires_grad([self.G_DN], True)
        util.set_requires_grad([self.G_DN], False)
        # # Turn off gradient calculation for D_DN
        loss_train_G_UP = self.train_G_UP()

        # Train D_DN
        # Turn on gradient calculation for discriminator
        # util.set_requires_grad([self.D_DN], True)
        # loss_train_D_DN = self.train_D_DN()

        loss.update(loss_train_G_DN)
        loss.update(loss_train_G_UP)
        # loss.update(loss_train_D_DN)



        if self.conf.debug:
            if self.iter % self.conf.eval_iters == 0:
                self.quick_eval()
            if self.iter % self.conf.plot_iters == 0:
                self.plot()
        self.iter = self.iter + 1

        return loss

    def train_G_DN(self):
        loss_cycle_forward = 0
        loss_cycle_backward = 0
        total_loss = 0
        loss_GAN = 0
        self.loss_regularization = 0

        if self.train_G_DN_switch:

            # Reset gradient valus
            # self.optimizer_G_UP.zero_grad()
            self.optimizer_G_DN.zero_grad()

            # Forward path
            # self.fake_HR = self.G_UP(self.real_LR)
            # self.rec_LR = self.G_DN(self.fake_HR)
            # Backward path
            self.fake_LR = self.G_DN(self.real_HR)
            self.rec_HR = self.G_UP(self.fake_LR)

            # Losses
            # loss_GAN = self.criterion_gan(self.D_DN(self.fake_LR), True)
            # loss_cycle_forward = self.criterion_cycle(self.rec_LR, util.shave_a2b(self.real_LR, self.rec_LR)) #* self.conf.lambda_cycle
            loss_cycle_backward = self.criterion_cycle(self.rec_HR, util.shave_a2b(self.real_HR, self.rec_HR)) #* self.conf.lambda_cycle

            total_loss = loss_cycle_forward + loss_cycle_backward + loss_GAN + self.loss_regularization
            # total_loss = loss_cycle_forward + loss_cycle_backward + self.loss_regularization

            total_loss.backward()

            # self.optimizer_G_UP.step()
            self.optimizer_G_DN.step()

        return {
            "train_G_DN/loss_cycle_forward": loss_cycle_forward, 
            "train_G_DN/loss_cycle_backward":loss_cycle_backward,
            "train_G_DN/total_loss":total_loss,
            "train_G_DN/loss_GAN": loss_GAN,
            "train_G_DN/loss_regularization": self.loss_regularization,
            }

    def train_G_UP(self):
        loss_cycle_forward = 0
        loss_cycle_backward = 0
        total_loss = 0
        loss_interp = 0
        
        if self.train_G_UP_switch:

            # Rese gradient valus
            self.optimizer_G_UP.zero_grad()

            # Forward path
            # self.fake_HR = self.G_UP(self.real_LR)
            # self.rec_LR = self.G_DN(self.fake_HR)
            # Backward path
            self.fake_LR = self.G_DN(self.real_HR)
            self.rec_HR = self.G_UP(self.fake_LR)

            # Losses
            # loss_cycle_forward = self.criterion_cycle(self.rec_LR, util.shave_a2b(self.real_LR, self.rec_LR)) #* self.conf.lambda_cycle
            loss_cycle_backward = self.criterion_cycle(self.rec_HR, util.shave_a2b(self.real_HR, self.rec_HR)) #* self.conf.lambda_cycle
            # self.fake_LR = self.G_DN(self.real_LR)
            # self.rec_HR = self.G_UP(self.fake_LR)
            # loss_cycle_backward = self.criterion_cycle(self.rec_HR, util.shave_a2b(self.real_LR, self.rec_HR)) * self.conf.lambda_cycle

            total_loss = loss_cycle_forward + loss_cycle_backward + loss_interp

            total_loss.backward()

            # Update weights
            self.optimizer_G_UP.step()

        return {
            "train_G_UP/loss_cycle_forward": loss_cycle_forward, 
            "train_G_UP/loss_cycle_backward":loss_cycle_backward,
            "train_G_UP/total_loss":total_loss,
            "train_G_UP/loss_interp":loss_interp,
            }

    # def train_D_DN(self):
    #     # Rese gradient valus
    #     self.loss_Discriminator = 0
        
    #     if self.train_D_DN_switch:
    #         self.optimizer_D_DN.zero_grad()
            
    #         # Fake
    #         pred_fake = self.D_DN(self.fake_LR.detach())
    #         loss_D_fake = self.criterion_gan(pred_fake, False)
    #         # Real
    #         pred_real = self.D_DN(util.shave_a2b(self.real_LR, self.fake_LR))
    #         loss_D_real = self.criterion_gan(pred_real, True)
    #         # Combined loss and calculate gradients
    #         self.loss_Discriminator = (loss_D_real + loss_D_fake) * 0.5
    #         self.loss_Discriminator.backward()

    #         # Update weights
    #         self.optimizer_D_DN.step()

    #     return {
    #         "train_D_DN/loss_Discriminator": self.loss_Discriminator
    #     }





class BackwardPathDDNPlus(TTASR_Base):
    '''
    In this mehod, we implement our most basic idea in which we train the G_DN
    and G_UP
    '''
    def __init__(
        self,
        conf
    ):
        super().__init__(conf)


    def train(self, data):
        self.set_input(data)

        loss = {}

        # Train G_DN
        self.G_UP.eval()
        # Turn off gradient calculation for G_UP
        util.set_requires_grad([self.G_UP], False)
        # # Turn on gradient calculation for G_DN
        util.set_requires_grad([self.G_DN], True)
        util.set_requires_grad([self.D_DN], False)
        loss_train_G_DN = self.train_G_DN()

        # Train G_UP
        self.G_UP.train()
        # # Turn on gradient calculation for G_UP
        util.set_requires_grad([self.G_UP], True)
        # # Turn off gradient calculation for G_DN
        # util.set_requires_grad([self.G_DN], True)
        util.set_requires_grad([self.G_DN], False)
        # # Turn off gradient calculation for D_DN
        loss_train_G_UP = self.train_G_UP()

        # Train D_DN
        # Turn on gradient calculation for discriminator
        util.set_requires_grad([self.D_DN], True)
        loss_train_D_DN = self.train_D_DN()

        loss.update(loss_train_G_DN)
        loss.update(loss_train_G_UP)
        loss.update(loss_train_D_DN)



        if self.conf.debug:
            if self.iter % self.conf.eval_iters == 0:
                self.quick_eval()
            if self.iter % self.conf.plot_iters == 0:
                self.plot()
        self.iter = self.iter + 1

        return loss

    def train_G_DN(self):
        loss_cycle_forward = 0
        loss_cycle_backward = 0
        total_loss = 0
        loss_GAN = 0
        self.loss_regularization = 0

        if self.train_G_DN_switch:

            # Reset gradient valus
            # self.optimizer_G_UP.zero_grad()
            self.optimizer_G_DN.zero_grad()

            # Forward path
            # self.fake_HR = self.G_UP(self.real_LR)
            # self.rec_LR = self.G_DN(self.fake_HR)
            # Backward path
            self.fake_LR = self.G_DN(self.real_HR)
            self.rec_HR = self.G_UP(self.fake_LR)

            # Losses
            loss_GAN = self.criterion_gan(self.D_DN(self.fake_LR), True)
            # loss_cycle_forward = self.criterion_cycle(self.rec_LR, util.shave_a2b(self.real_LR, self.rec_LR)) #* self.conf.lambda_cycle
            loss_cycle_backward = self.criterion_cycle(self.rec_HR, util.shave_a2b(self.real_HR, self.rec_HR)) #* self.conf.lambda_cycle

            total_loss = loss_cycle_forward + loss_cycle_backward + loss_GAN + self.loss_regularization
            # total_loss = loss_cycle_forward + loss_cycle_backward + self.loss_regularization

            total_loss.backward()

            # self.optimizer_G_UP.step()
            self.optimizer_G_DN.step()

        return {
            "train_G_DN/loss_cycle_forward": loss_cycle_forward, 
            "train_G_DN/loss_cycle_backward":loss_cycle_backward,
            "train_G_DN/total_loss":total_loss,
            "train_G_DN/loss_GAN": loss_GAN,
            "train_G_DN/loss_regularization": self.loss_regularization,
            }

    def train_G_UP(self):
        loss_cycle_forward = 0
        loss_cycle_backward = 0
        total_loss = 0
        loss_interp = 0
        
        if self.train_G_UP_switch:

            # Rese gradient valus
            self.optimizer_G_UP.zero_grad()

            # Forward path
            self.fake_HR = self.G_UP(self.real_LR)
            self.rec_LR = self.G_DN(self.fake_HR)
            # Backward path
            self.fake_LR = self.G_DN(self.real_HR)
            self.rec_HR = self.G_UP(self.fake_LR)

            # Losses
            loss_cycle_forward = self.criterion_cycle(self.rec_LR, util.shave_a2b(self.real_LR, self.rec_LR)) #* self.conf.lambda_cycle
            loss_cycle_backward = self.criterion_cycle(self.rec_HR, util.shave_a2b(self.real_HR, self.rec_HR)) #* self.conf.lambda_cycle
            # self.fake_LR = self.G_DN(self.real_LR)
            # self.rec_HR = self.G_UP(self.fake_LR)
            # loss_cycle_backward = self.criterion_cycle(self.rec_HR, util.shave_a2b(self.real_LR, self.rec_HR)) * self.conf.lambda_cycle

            total_loss = loss_cycle_forward + loss_cycle_backward + loss_interp

            total_loss.backward()

            # Update weights
            self.optimizer_G_UP.step()

        return {
            "train_G_UP/loss_cycle_forward": loss_cycle_forward, 
            "train_G_UP/loss_cycle_backward":loss_cycle_backward,
            "train_G_UP/total_loss":total_loss,
            "train_G_UP/loss_interp":loss_interp,
            }

    def train_D_DN(self):
        # Rese gradient valus
        self.loss_Discriminator = 0
        
        if self.train_D_DN_switch:
            self.optimizer_D_DN.zero_grad()
            
            # Fake
            pred_fake = self.D_DN(self.fake_LR.detach())
            loss_D_fake = self.criterion_gan(pred_fake, False)
            # Real
            pred_real = self.D_DN(util.shave_a2b(self.real_LR, self.fake_LR))
            loss_D_real = self.criterion_gan(pred_real, True)
            # Combined loss and calculate gradients
            self.loss_Discriminator = (loss_D_real + loss_D_fake) * 0.5
            self.loss_Discriminator.backward()

            # Update weights
            self.optimizer_D_DN.step()

        return {
            "train_D_DN/loss_Discriminator": self.loss_Discriminator
        }

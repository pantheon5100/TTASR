import copy
import glob
import imp
import json
import os
import shutil
import time
from argparse import Namespace
from datetime import datetime

import ipdb
import numpy as np
import setGPU
import torch
import tqdm
import wandb

import tta_util as util
from tta_data import create_dataset
# from DualSR import DualSR
# from tta_sr import TTASR
# from tta_sr_one_cycle import TTASR
from tta_learner import Learner
from tta_options import options
from tta_sr import TTASR_Base


def train_and_eval(conf):
    from tta_sr import TTASR_Base as TTASR
    model = TTASR(conf)
    dataloader = create_dataset(conf)
    learner = Learner(model)

    print('*' * 60 + '\nTraining started ...')
    # for iteration, data in enumerate(tqdm.tqdm(dataloader)):
    best_res = {
        "iteration": 0,
        "PSNR": 0,
    }
    psnr_record = []
    for iteration, data in enumerate(tqdm.tqdm(dataloader)):
        # NOTE: iteration start from 0

        # Training state change
        if iteration+1 == 1:
            model.train_G_DN_switch = True
            model.train_G_UP_switch = False
            util.set_requires_grad([model.G_UP], False)
            # Turn on gradient calculation for G_DN
            util.set_requires_grad([model.G_DN], True)

        if (iteration+1) % model.conf.switch_iters == 0:
            model.train_G_UP_switch = not model.train_G_UP_switch
            model.train_G_DN_switch = not model.train_G_DN_switch
            util.set_requires_grad([model.G_UP], True)
            # Turn off gradient calculation for G_DN
            # util.set_requires_grad([self.G_DN], True)
            util.set_requires_grad([model.G_DN], False)

        loss = model.train(data)
        learner.update(iteration, model)

        if (iteration+1) % conf.model_save_iter == 0:
            model.save_model(iteration+1)

        if (iteration+1) % conf.eval_iters == 0 and model.train_G_UP_switch:
            model.eval(iteration)
            with open(os.path.join(conf.experimentdir, "psnr.txt"), "a") as f:
                f.write(
                    f"IMG_IDX: {conf.img_idx}. Iteration: {iteration}. {conf.abs_img_name}. PSNR: {model.UP_psnrs[-1]} \n")
            print(
                f"IMG_IDX: {conf.img_idx}. Iteration: {iteration}. {conf.abs_img_name}. PSNR: {model.UP_psnrs[-1]}")

            loss["eval/psnr"] = model.UP_psnrs[-1]

            psnr_record.append([iteration, model.UP_psnrs[-1]])

            if model.UP_psnrs[-1] > best_res["PSNR"]:
                best_res["PSNR"] = model.UP_psnrs[-1]
                best_res["iteration"] = iteration
            pass
        # else:
        #     loss["eval/psnr"] = 0

        if (iteration+1) % conf.eval_iters == 0:
            loss_log = {}
            for key, val in loss.items():
                key = f"{conf.abs_img_name}/{key}"
                loss_log[key] = val

            loss_log["iteration"] = iteration
            wandb.log(loss_log)

    torch.save(psnr_record, os.path.join(
        conf.model_save_dir, f"{conf.abs_img_name}_psnr.pt"))
    print("Best PSNR: {}, at iteration: {}".format(
        best_res["PSNR"], best_res["iteration"]))
    wandb.run.summary[f"best_psnr_{conf.abs_img_name}"] = best_res["PSNR"]

    model.eval(0)
    return model.UP_psnrs[-1]


def main():
    torch.set_num_threads(4)

    opt = options()

    #############################################################################################
    # code save
    #############################################################################################
    print("Start file saving...")
    time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # ipdb.set_trace()
    experimentdir = f"./log/{opt.conf.output_dir}/time_{time_stamp}"
    experimentdir += f"lr_GUP_{opt.conf.lr_G_UP}-lr_GDN_{opt.conf.lr_G_DN}input_size_{opt.conf.input_crop_size}-scale_factor_{opt.conf.scale_factor}"
    opt.conf.experimentdir = experimentdir
    code_saving_dir = os.path.join(experimentdir, "code")
    os.makedirs(code_saving_dir)

    shutil.copytree(f"./tta_model", os.path.join(code_saving_dir, 'tta_model'))
    shutil.copytree(f"./bash_files",
                    os.path.join(code_saving_dir, 'bash_files'))

    # search main dir .py file to save
    pathname = "*.py"
    files = glob.glob(pathname, recursive=True)
    for file in files:
        dest_fpath = os.path.join(code_saving_dir, os.path.basename(file))
        shutil.copy(file, dest_fpath)

    opt.conf.visual_dir = os.path.join(experimentdir, "visual")
    os.makedirs(opt.conf.visual_dir)
    opt.conf.model_save_dir = os.path.join(experimentdir, "ckpt")
    os.makedirs(opt.conf.model_save_dir)

    # save running argument
    with open(os.path.join(experimentdir, 'commandline_args.txt'), 'w') as f:
        json.dump(opt.conf.__dict__, f, indent=2)
    #############################################################################################
    #############################################################################################

    # init logger
    if not opt.conf.test_only:
        # os.environ["WANDB_MODE"] = "offline"
        # wandb logger
        wandb.init(
            project="TTA_SR",
            entity="kaistssl",
            name=opt.conf.output_dir,
            config=opt.conf,
            dir=opt.conf.experimentdir,
            save_code=True,
        )

    #############################################################################################
    # Testing
    #############################################################################################
    if opt.conf.test_only:
        from tta_sr import TTASR_Base as TTASR
        model = TTASR(opt.conf)
        all_psnr = []
        img_list = []
        for img_idx, img_name in enumerate(os.listdir(opt.conf.input_dir)):
            conf = opt.get_config(img_name)
            conf.img_idx = img_idx

            model.read_image(conf)
            model.eval(0)
            with open(os.path.join(conf.experimentdir, "psnr.txt"), "a") as f:
                f.write(
                    f"IMG: {img_idx}. Iteration: {0}. {conf.abs_img_name}. PSNR: {model.UP_psnrs[-1]}\n")
            all_psnr.append(model.UP_psnrs[-1])
            img_list.append(img_name)
        all_psnr = np.array(all_psnr)
        with open(os.path.join(conf.experimentdir, "final_psnr.txt"), "a") as f:
            f.write(f"Input directory: {opt.conf.input_dir}.\n")

            for img, psnr in zip(img_list, all_psnr):
                f.write(f"IMG: {img}, psnr: {psnr} .\n")

            f.write(f"Average PSNR: {np.mean(all_psnr)}.\n")
        print(f"Average PSNR for {opt.conf.input_dir}: {np.mean(all_psnr)}")

    #############################################################################################
    # Training
    #############################################################################################

    elif opt.conf.train_mode == "bicubic":

        from tta_sr import BicubicGDN
        train_image_specific_GUP(
            opt=opt,
            pretrained_gdn_state_dict=None,
            TTASR=BicubicGDN
        )

    elif opt.conf.train_mode == "backward_path":
        # import ipdb; ipdb.set_trace()
        from tta_sr import BackwardPath
        if opt.conf.pretrained_GDN == "":

            pretrained_gdn_state_dict = train_gdn(opt, TTASR=BackwardPath)

        else:
            print(f"Load pretrained GDN from {opt.conf.pretrained_GDN}.")
            pretrained_gdn_state_dict = torch.load(opt.conf.pretrained_GDN)

        # train image specific GUP
        train_image_specific_GUP(
            opt=opt,
            pretrained_gdn_state_dict=pretrained_gdn_state_dict,
            TTASR=BackwardPath,
        )

    elif opt.conf.train_mode == "backward_path_ddn":
        from tta_sr import BackwardPathDDN
        if opt.conf.pretrained_GDN == "":
            pretrained_gdn_state_dict = train_gdn(opt, TTASR=BackwardPathDDN)

        else:
            pretrained_gdn_state_dict = torch.load(opt.conf.pretrained_GDN)

        # train image specific GUP
        train_image_specific_GUP(
            opt=opt,
            pretrained_gdn_state_dict=pretrained_gdn_state_dict,
            TTASR=BackwardPathDDN,
        )

    elif opt.conf.train_mode == "backward_path_ddn_plus":
        from tta_sr import BackwardPathDDNPlus
        if opt.conf.pretrained_GDN == "":

            pretrained_gdn_state_dict = train_gdn(
                opt, TTASR=BackwardPathDDNPlus)

        else:
            pretrained_gdn_state_dict = torch.load(opt.conf.pretrained_GDN)

        # train image specific GUP
        train_image_specific_GUP(
            opt=opt,
            pretrained_gdn_state_dict=pretrained_gdn_state_dict,
            TTASR=BackwardPathDDNPlus,
        )

    elif opt.conf.train_mode == "single_image":

        # Run DualSR on all images in the input directory
        img_list = []
        all_psnr = []
        for img_name in os.listdir(opt.conf.input_dir):
            conf = opt.get_config(img_name)
            conf.img_idx = img_idx
            psnr = train_and_eval(conf)

            all_psnr.append(psnr)
            img_list.append(img_name)

        all_psnr = np.array(all_psnr)
        with open(os.path.join(conf.experimentdir, "final_psnr.txt"), "a") as f:
            f.write(f"Input directory: {opt.conf.input_dir}.\n")

            for img_idx, (img, psnr) in enumerate(zip(img_list, all_psnr)):
                f.write(f"{img_idx}. IMG: {img}, psnr: {psnr} .\n")

            f.write(f"Average PSNR: {np.mean(all_psnr)}.\n")
        print(f"Average PSNR for {opt.conf.input_dir}: {np.mean(all_psnr)}")

    elif opt.conf.train_mode == "image_agnostic_gdn":
        # image_agnostic_gdn
        # here we want to use all test image to train the gdn,
        # before we train the gup. There have two scenarios, (1)
        # when we train gup we do not update gdn and (2) we still
        # update gdn before or when we train gup
        # os.environ["WANDB_MODE"] = "offline"
        from tta_sr import TTASR_Base
        if opt.conf.pretrained_GDNtime.clock() == "":

            pretrained_gdn_state_dict = train_gdn(opt, TTASR=TTASR_Base)

        else:
            pretrained_gdn_state_dict = torch.load(opt.conf.pretrained_GDN)

        # train image specific GUP
        train_image_specific_GUP(
            opt=opt,
            pretrained_gdn_state_dict=pretrained_gdn_state_dict,
            TTASR=TTASR_Base,
        )

    else:

        raise NotImplementedError


def train_gdn(opt, TTASR: TTASR_Base):
    print('*' * 60 + '\nTraining GDN started ...')

    start = time.process_time()

    opt = copy.deepcopy(opt)
    opt.conf.num_iters = 3000
    opt.conf.switch_iters = 3000
    opt.conf.update_l_rate_freq_gdn = 750

    end = time.process_time()
    print(f"setting opt for gdn: {start-end} seconds.")

    start = time.process_time()

    model = TTASR(opt.conf)
    # use all image to train, every batch contains all image
    from tta_data import create_dataset_for_image_agnostic_gdn
    dataloader = create_dataset_for_image_agnostic_gdn(opt.conf)
    learner = Learner(model)

    # freeze GUP
    model.train_G_DN_switch = True
    model.train_G_UP_switch = False
    model.reshap_train_data = True
    model.G_UP.eval()

    util.set_requires_grad([model.G_UP], False)
    # Turn on gradient calculation for G_DN
    util.set_requires_grad([model.G_DN], True)

    end = time.process_time()
    print(f"setting model and dataloader for gdn: {start-end} seconds.")

    # train GDN
    # dataloader time: 0.0003034459998616512
    # optimization time: 1.9583428319999712
    # learner update time: 7.288000006155926e-06
    # iter total time: 1.9584988839999369
    # dataloader_time = time.process_time()
    # for iteration, data in enumerate(tqdm.tqdm(dataloader)):
    for iteration, data in enumerate(dataloader):

        # dataloader_time = time.process_time() - dataloader_time
        # print(f"dataloader time: {dataloader_time}")

        # iter_time = time.process_time()

        # train_time = time.process_time()
        loss = model.train(data)
        # train_time = time.process_time() - train_time
        # print(f"optimization time: {train_time}")

        if (iteration+1) % opt.conf.eval_iters == 0:
            # log_time = time.process_time()
            loss_log = {}
            for key, val in loss.items():
                key = f"train_GDN/{key}"
                loss_log[key] = val

            loss_log["train_GDN/iteration"] = iteration
            wandb.log(loss_log)
            # log_time = time.process_time() - log_time
            # print(f"log time: {log_time}")

        # learner_update_time = time.process_time()
        learner.update(iteration, model)
        # learner_update_time = time.process_time() - learner_update_time
        # print(f"learner update time: {learner_update_time}")

        # iter_time = time.process_time() - iter_time
        # print(f"iter total time: {iter_time}")
        # print("\n")
        # dataloader_time = time.process_time()

    # save the pretrained GDN
    torch.save(model.G_DN.state_dict(), os.path.join(
        opt.conf.model_save_dir, "pretrained_GDN.ckpt"))

    return model.G_DN.state_dict()


def train_image_specific_GUP(opt, pretrained_gdn_state_dict, TTASR):
    torch.set_num_threads(3)
    print('*' * 60 + '\nTraining image specific GUP started ...')

    best_res = {
        "iteration": 0,
        "PSNR": 0,
    }
    img_list = []
    all_psnr = []
    for img_idx, img_name in enumerate(os.listdir(opt.conf.input_dir)):
        conf = opt.get_config(img_name)
        conf.img_idx = img_idx

        model_img_specific = TTASR(conf)
        model_img_specific.reshap_train_data = False
        model_img_specific.train_D_DN_switch = False
        if pretrained_gdn_state_dict != None:
            model_img_specific.G_DN.load_state_dict(pretrained_gdn_state_dict)

        dataloader = create_dataset(conf)
        learner = Learner(model_img_specific)

        for iteration, data in enumerate(tqdm.tqdm(dataloader)):

            # Training state change
            if iteration == 0:
                model_img_specific.train_G_DN_switch = True
                model_img_specific.train_G_UP_switch = False
                util.set_requires_grad([model_img_specific.G_UP], False)
                # Turn on gradient calculation for G_DN
                util.set_requires_grad([model_img_specific.G_DN], True)
            if (iteration+1) == model_img_specific.conf.switch_iters:
                model_img_specific.train_G_UP_switch = not model_img_specific.train_G_UP_switch
                model_img_specific.train_G_DN_switch = not model_img_specific.train_G_DN_switch
                util.set_requires_grad([model_img_specific.G_UP], True)
                # Turn off gradient calculation for G_DN
                # util.set_requires_grad([self.G_DN], True)
                util.set_requires_grad([model_img_specific.G_DN], False)

            loss = model_img_specific.train(data)
            learner.update(iteration, model_img_specific)

            if (iteration+1) % conf.model_save_iter == 0:
                model_img_specific.save_model(iteration+1)

            if (iteration+1) % conf.eval_iters == 0 and model_img_specific.train_G_UP_switch:
                model_img_specific.eval(iteration)
                with open(os.path.join(conf.experimentdir, "psnr.txt"), "a") as f:
                    f.write(
                        f"IMG_IDX: {conf.img_idx}. iteration: {iteration}. {conf.abs_img_name}. PSNR: {model_img_specific.UP_psnrs[-1]} \n")
                # print(f"{iteration}. {conf.abs_img_name}. PSNR: {model_img_specific.UP_psnrs[-1]}")

                loss["eval/psnr"] = model_img_specific.UP_psnrs[-1]

                if model_img_specific.UP_psnrs[-1] > best_res["PSNR"]:
                    best_res["PSNR"] = model_img_specific.UP_psnrs[-1]
                    best_res["iteration"] = iteration
                pass

            if (iteration+1) % conf.eval_iters == 0:
                loss_log = {}
                for key, val in loss.items():
                    key = f"{conf.abs_img_name}/{key}"
                    loss_log[key] = val

                loss_log["iteration"] = iteration
                wandb.log(loss_log)

        # print("Best PSNR: {}, at iteration: {}".format(best_res["PSNR"], best_res["iteration"]))

        all_psnr.append(model_img_specific.UP_psnrs[-1])
        img_list.append(img_name)

    all_psnr = np.array(all_psnr)
    with open(os.path.join(conf.experimentdir, "final_psnr.txt"), "a") as f:
        f.write(f"Input directory: {opt.conf.input_dir}.\n")

        for img, psnr in zip(img_list, all_psnr):
            f.write(f"IMG: {img}, psnr: {psnr} .\n")

        f.write(f"Average PSNR: {np.mean(all_psnr)}.\n")
    print(f"Average PSNR for {opt.conf.input_dir}: {np.mean(all_psnr)}")


if __name__ == '__main__':
    main()

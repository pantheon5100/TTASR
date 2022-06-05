import os
import tqdm

from tta_options import options
from tta_data import create_dataset
# from DualSR import DualSR
from tta_sr import TTASR
from tta_learner import Learner

import torch

import shutil
import glob
import json
import os

from datetime import datetime
import wandb
import ipdb


def train_and_eval(conf):
    model = TTASR(conf)
    dataloader = create_dataset(conf)
    learner = Learner(model)

    if conf.test_only:
        model.eval(0)
        with open(os.path.join(conf.experimentdir, "psnr.txt"), "a") as f:
            f.write(f"{iteration}. {conf.abs_img_name}. PSNR: {model.UP_psnrs[-1]}\n")
        return

    # generate dataset first
    data_dir = f"generating_data/data_{conf.abs_img_name}_{conf.input_crop_size}_{conf.num_iters}.pth"
    os.makedirs(os.path.dirname(data_dir), exist_ok=True)

    data_collection = []
    if not os.path.exists(data_dir):
        for iteration, data in enumerate(tqdm.tqdm(dataloader)):
            data_collection.append(data)

        torch.save(data_collection, data_dir)
    else:
        data_collection = torch.load(data_dir)

    print('*' * 60 + '\nTraining started ...')
    # for iteration, data in enumerate(tqdm.tqdm(dataloader)):
    best_res = {
        "iteration": 0,
        "PSNR": 0,
    }
    for iteration, data in enumerate(tqdm.tqdm(data_collection)):
        if iteration == 0:
            model.train_G_DN_switch = True
            model.train_G_UP_switch = False
            

        loss = model.train(data)
        learner.update(iteration, model)

        loss["iteration"] = iteration


        if (iteration+1) % conf.model_save_iter == 0:
            model.save_model(iteration+1)

        if (iteration+1) % conf.eval_iters == 0 and model.train_G_UP_switch:
            model.eval(iteration)
            with open(os.path.join(conf.experimentdir, "psnr.txt"), "a") as f:
                f.write(f"{iteration}. {conf.abs_img_name}. PSNR: {model.UP_psnrs[-1]}\n")
            print(f"{iteration}. {conf.abs_img_name}. PSNR: {model.UP_psnrs[-1]}")
            
            loss["eval/psnr"] = model.UP_psnrs[-1]
            
            if model.UP_psnrs[-1] > best_res["PSNR"]:
                best_res["PSNR"] = model.UP_psnrs[-1]
                best_res["iteration"] = iteration
            pass
        else:
            loss["eval/psnr"] = 0


        if (iteration+1) % conf.eval_iters == 0:
            wandb.log(loss)

    print("Best PSNR: {}, at iteration: {}".format(best_res["PSNR"], best_res["iteration"]))
    wandb.run.summary[f"best_psnr_{conf.abs_img_name}"] = best_res["PSNR"]
    
    model.eval(0)


def main():
    opt = options()

    #############################################################################################
    #############################################################################################
    print("Start file saving...")
    time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # ipdb.set_trace()
    experimentdir = f"./log/{opt.conf.output_dir}/time_{time_stamp}"
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

    # wandb logger
    wandb.init(
        project="TTA_SR", 
        entity="kaistssl",
        name=opt.conf.output_dir,
        config=opt.conf,
        dir=opt.conf.experimentdir,
        save_code=True,

        )

    # Run DualSR on all images in the input directory
    for img_name in os.listdir(opt.conf.input_dir):
        conf = opt.get_config(img_name)
        train_and_eval(conf)


if __name__ == '__main__':
    main()

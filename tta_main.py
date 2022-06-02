import os
import tqdm

from tta_options import options
from tta_data import create_dataset
# from DualSR import DualSR
from tta_sr import TTASR
from tta_learner import Learner

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
        model.eval()
        
        return


    print('*' * 60 + '\nTraining started ...')
    for iteration, data in enumerate(tqdm.tqdm(dataloader)):
    # for iteration, data in enumerate(dataloader):

        loss = model.train(data)
        learner.update(iteration, model)

        loss["iteration"] = iteration

        if (iteration+1) % conf.eval_iters == 0:
            wandb.log(loss)

        if (iteration+1) % conf.model_save_iter == 0:
            model.save_model(iteration+1)

        if (iteration+1) % 100 == 0 and model.train_G_UP_switch:
            model.eval()
            with open(os.path.join(conf.experimentdir, "psnr.txt"), "a") as f:
                f.write(f"{iteration}. {conf.abs_img_name}. PSNR: {model.UP_psnrs[-1]}\n")
            print(f"{iteration}. {conf.abs_img_name}. PSNR: {model.UP_psnrs[-1]}")
            pass

    model.eval()


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
        dir=opt.conf.experimentdir
        )

    # Run DualSR on all images in the input directory
    for img_name in os.listdir(opt.conf.input_dir):
        conf = opt.get_config(img_name)
        train_and_eval(conf)


if __name__ == '__main__':
    main()

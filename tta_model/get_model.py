from argparse import Namespace

import torch

from tta_model.network_swinir import define_model
from tta_model.rcan import RCAN


def get_model(conf):
    if conf.source_model == "swinir":
        G_UP_model_conf = {
            "task": "classical_sr",
            "scale": conf.scale_factor,
            "model_type": f"classicalSR_s1_{conf.scale_factor}",
            # "training_patch_size": conf.input_crop_size,
            "training_patch_size": 48,
            "large_model": False
            }
        G_UP = define_model(**G_UP_model_conf).cuda()
        
        return G_UP

    elif conf.source_model == "rcan":
        rcan_config = Namespace()
        rcan_config.n_resgroups = 10
        rcan_config.n_resblocks = 20
        rcan_config.n_feats = 64
        rcan_config.scale = [2]
        rcan_config.data_train = "DIV2K"
        rcan_config.rgb_range = 255
        rcan_config.n_colors = 3
        rcan_config.res_scale = 1
        rcan_config.reduction = 16

        G_UP = RCAN(rcan_config)

        state_dict = torch.load("tta_pretrained/models_ECCV2018RCAN/RCAN_BIX2.pt")
        G_UP.load_state_dict(state_dict=state_dict)
        G_UP.cuda()

        return G_UP





        


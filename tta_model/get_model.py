from argparse import Namespace

from network_swinir import define_model
from rcan import RCAN


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
        
        
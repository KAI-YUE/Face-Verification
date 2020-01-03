import os
import cv2
import torch
import numpy as np
from PIL import Image

# My Libraries
from src.loadConfig import loadConfig

def model_save(model, epoch, logger=None, max_save=5):
    if not (logger is None):
        logger.info("saving the model in epoch {}".format(epoch))
    
    current_path = current_path = os.path.dirname(os.path.dirname(__file__))
    config = loadConfig(os.path.join(current_path, "config.json"))
    dst_dir = os.path.join(config.model_dir, "epoch{}.pth".format(epoch))

    torch.save({
        'epoch': epoch,
        'nn_state_dict': model.state_dict()
        }, dst_dir)

    
    # Check the model and only preserve the last max_save(default:5) models
    flist = os.listdir(config.model_dir)
    if (len(flist) > max_save):
        for f in flist:
            if (f == 'epoch' + str(epoch-max_save*config.save_interval) + '.pth'):
                os.remove(os.path.join(config.model_dir, f))
                break


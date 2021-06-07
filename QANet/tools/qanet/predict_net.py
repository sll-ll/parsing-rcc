import argparse
import os

import torch

import _init_paths  # pylint: disable=unused-import
from lib.utils.analyser import Analyser
from lib.utils.checkpointer import get_weights, load_weights
from lib.utils.comm import all_gather, is_main_process, synchronize
from lib.utils.logger import build_test_hooks
from lib.utils.misc import logging_rank, mkdir_p, setup_logging
from lib.utils.timer import Timer

from qanet.core.config import get_cfg, infer_cfg
from qanet.core.test import TestEngine
from qanet.datasets.dataset import build_dataset, make_test_data_loader
from qanet.datasets.evaluation import Evaluation
from qanet.modeling.model_builder import Generalized_CNN
from tifffile import tifffile
import numpy as np
import cv2
from PIL import Image


num_classes = 18

class_colors = [(0, 0, 0),     # 0 background
                (55, 125, 34), # 1 tou
                (177, 27, 124), # 2 zuobi
                (129, 127, 38), # 3 youbi
                (153, 126,127), # 4 zuotui
                (0, 12, 122), # 5 youtui
                (118, 20, 12), # 6 qugan
                (85, 125, 234),  # 1 tou
                (77, 233, 124),  # 2 zuobi
                (129, 23, 138),  # 3 youbi
                (153, 126, 56),  # 4 zuotui
                (111, 172, 122),  # 5 youtui
                (18, 220, 12),  # 6 qugan
                (53, 126, 217),  # 4 zuotui
                (45, 12, 12),  # 5 youtui
                (90, 120, 112),  # 6 qugan
                (45, 222, 212),  # 5 youtui
                (90, 20, 212),  # 6 qugan
                ]
def get_colored_segmentation_image(seg_arr, n_classes, colors=class_colors):
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]
    seg_img = np.zeros((output_height, output_width, 3))

    for cls in range(n_classes):
        seg_arr_c = seg_arr[:, :] == cls
        print('-class %s---count is %s'% (cls, np.sum(seg_arr == cls)))
        seg_img[:, :, 0] += ((seg_arr_c)*(colors[cls][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c)*(colors[cls][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c)*(colors[cls][2])).astype('uint8')

    return seg_img


def add_color(pre):
    seg_image = get_colored_segmentation_image(np.array(pre),
                                               num_classes,
                                               class_colors)
    return np.uint8(seg_image)



def main(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg = infer_cfg(cfg)
    cfg.freeze()
    # logging_rank(cfg)

    if not os.path.isdir(cfg.CKPT):
        mkdir_p(cfg.CKPT)
    setup_logging(cfg.CKPT)

    # Calculate Params & FLOPs & Activations
    n_params, conv_flops, model_flops, conv_activs, model_activs = 0, 0, 0, 0, 0
    if is_main_process() and cfg.MODEL_ANALYSE:
        model = Generalized_CNN(cfg)
        model.eval()
        analyser = Analyser(cfg, model, param_details=False)
        n_params = analyser.get_params()[1]
        conv_flops, model_flops = analyser.get_flops_activs(cfg.TEST.SCALE[0], cfg.TEST.SCALE[1], mode='flops')
        conv_activs, model_activs = analyser.get_flops_activs(cfg.TEST.SCALE[0], cfg.TEST.SCALE[1], mode='activations')
        del model

    synchronize()
    # Create model
    model = Generalized_CNN(cfg)
    logging_rank(model)

    # Load model
    test_weights = get_weights(cfg.CKPT, cfg.TEST.WEIGHTS)
    load_weights(model, test_weights)
    logging_rank(
        'Params: {} | FLOPs: {:.4f}M / Conv_FLOPs: {:.4f}M | '
        'ACTIVATIONs: {:.4f}M / Conv_ACTIVATIONs: {:.4f}M'.format(
            n_params, model_flops, conv_flops, model_activs, conv_activs
        )
    )

    model.eval()
    model.to(torch.device(cfg.DEVICE))

    from qanet.core.my_inference import Inference

    img_path = '/home/workspace/RegionSeg/Parsing-R-CNN/data/CIHP/val_img/7b92654a64e14e02a4cdc7780df4a5c9.tif'

    inference = Inference(cfg, model)
    image = tifffile.imread(img_path)
    image = np.expand_dims(image, axis=2)
    img = np.concatenate((image, image, image), axis=-1)

    ss = inference(img, [[20, 20, 260, 360]])
    # ss[4] (1, 384, 288)
    print(np.unique(ss[4]))
    # save seg
    color_mask = add_color(ss[4][0])

    img = img * 5
    img = img.astype(np.uint8)

    ss = cv2.addWeighted(img, 0.7, color_mask, 0.3, gamma=0)
    seg_output = './outputs/seg/2'
    if not os.path.exists(seg_output):
        os.makedirs(seg_output)
    name = img_path.split('/')[-1].split('.')[0]
    Image.fromarray(img).save(os.path.join(seg_output, '{}.png'.format(name)))
    Image.fromarray(ss).save(os.path.join(seg_output, '{}_seg.png'.format(name)))
    Image.fromarray(color_mask).save(os.path.join(seg_output, '{}_mask.png'.format(name)))



if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='QANet Model Training')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='/home/workspace/parsing-rcc/QANet/cfgs/CIHP/QANet/QANet_H-W48_512x384_1x.yaml', type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('opts', help='See qanet/core/config.py for all options',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    main(args)

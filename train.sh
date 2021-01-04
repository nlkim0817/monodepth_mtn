#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1 python main_monodepth_pytorch.py --model resnet18_md --l_type sl1 --model_output_directory model_output/res18
CUDA_VISIBLE_DEVICES=0,1 python main_monodepth_pytorch.py --model resnet18_md_v2 --l_type sl1 --model_output_directory model_output/res18_v2

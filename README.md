# surrogate_module_learning
 Code for surrogate module learning (SML)
 Create a new backward path for more accurate SNN gradients.
 
## Prerequisites
The Following Setup is tested and it is working:
 * Python>=3.5
 * Pytorch>=1.9.0
 * Cuda>=10.2

## Description
 * We use Dspike surrogate gradient to realize the backward of step function.
 * LIF model is build in LIFSpike in models/layer.py.
 * If you want to adjust the hyperparameters, number, position, and structure of the surrogate module, the relevant code is on lines 512-537 of Train_distribute_pallel.py.
 * If you have two GPUs, you can use the following code to run this demo on CIFAR100 with ResNet18 structure SNN with T=2, the default two surrogate modules are located after 3-rd and 6-th basicblocks.
 ```
 python -m torch.distributed.launch --nproc_per_node=2 --use_env Train_distribute_pallel.py \
    --batch-size 128 --cos_lr_T 300 --epochs 300 \
    --model ResNet_SB18 \
    --num_classes 100 --dataset cifar100 --T 2 \
    --sync-bn --optimizer adamw --lr 0.01 --weight-decay 0.02
 ```

## Pre-trained models
* All the pre-trained models we used are avilable [here](https://drive.google.com/drive/folders/1UaesWFejJKQ4PhAx2xK6Av5_b1U4vl9j?usp=share_link)

## Citation
Reference [paper](https://openreview.net/pdf?id=zRkz4duLKp).

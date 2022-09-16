# LKRM
This repo is the implementation of the paper ("Latent Knowledge Reasoning Incorporated for Multi-fitting Decoupling Detection on Electric Transmission Line"). The code is based on PyTorch and large part code is reference from [faster-rcnn](https://github.com/diamour/cascade-rcnn-fpn-faster_rcnn-pytorch1.0).

## Requirements
+ Python3.8
+ Python packages
  + PyTorch >= 1.0
  + Torchvision >= 0.9.0
  + opencv-python
  + scipy
  + matplotlib
  + numpy
  

## Demo
After successfully completing requirements, you can be ready to run the demo.

+ **Download** the cascade_fpn_1_12_2325.pth which finally use in the paper(LKRM) from [Weights](https://pan.baidu.com/s/16dILd0w_NenQlgVCnml89g) (extract code:idfv)

+ **Download**  the pretrained weights(pascal_voc_cascade.pth and resnet101_caffe.pth) from [Weights](https://pan.baidu.com/s/16dILd0w_NenQlgVCnml89g) (extract code:idfv)

+ **Put**  cascade_fpn_1_12_2325.pth into the: 
```sh
{repo_root}/models/res101/pascal_voc/0.0018_9_0.1_023010/
```
+ **Put**  pascal_voc_cascade.pth into the: 
```sh
{repo_root}/models/
```
+ **Put**  resnet101_caffe.pth into the: 
```sh
{repo_root}/data/pretrained_model/
```
+ **Using** this code to see the fitting detection results in demo images:
```sh
python cascade_test_net.py --cuda
```



## Citation

    @article{jjfaster2rcnn,
        Author = {Jianwei Yang and Jiasen Lu and Dhruv Batra and Devi Parikh},
        Title = {A Faster Pytorch Implementation of Faster R-CNN},
        Journal = {https://github.com/jwyang/faster-rcnn.pytorch},
        Year = {2017}
    }

    @inproceedings{renNIPS15fasterrcnn,
        Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
        Title = {Faster {R-CNN}: Towards Real-Time Object Detection
                 with Region Proposal Networks},
        Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
        Year = {2015}
    }

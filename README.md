# FairMOT
A simple baseline for one-shot multi-object tracking:
![](assets/pipeline.png)
> [**A Simple Baseline for Multi-Object Tracking**](http://arxiv.org/abs/2004.01888),            
> Yifu Zhang, Chunyu Wang, Xinggang Wang, Wenjun Zeng, Wenyu Liu,        
> *arXiv technical report ([arXiv 2004.01888](http://arxiv.org/abs/2004.01888))*
## Abstract
There has been remarkable progress on object detection and re-identification in recent years which are the core components for multi-object tracking. However, little attention has been focused on accomplishing the two tasks in a single network to improve the inference speed. The initial attempts along this path ended up with degraded results mainly because the re-identification branch is not appropriately learned. In this work, we study the essential reasons behind the failure, and accordingly present a simple baseline to addresses the problems. It remarkably outperforms the state-of-the-arts on the MOT challenge datasets at 30 FPS. We hope this baseline could inspire and help evaluate new ideas in this field.

## Tracking performance
### Results on MOT challenge test set
| Dataset    |  MOTA | IDF1 | IDS | MT | ML | FPS |
|--------------|-----------|--------|-------|----------|----------|--------|
|2DMOT15  | 59.0 | 62.2 |  582 | 45.6% | 11.5% | 30.5 |
|MOT16       | 68.7 | 70.4 | 953 | 39.5% | 19.0% | 25.9 |
|MOT17       | 67.5 | 69.8 | 2868 | 37.7% | 20.8% | 25.9 |
|MOT20       | 58.7 | 63.7 | 6013 | 66.3% | 8.5% | 13.2 |

 All of the results are obtained on the [MOT challenge](https://motchallenge.net) evaluation server under the “private detector” protocol. We rank first among all the trackers on 2DMOT15, MOT17 and the recently released (2020.02.29) MOT20. Note that our IDF1 score remarkably outperforms other one-shot MOT trackers by more than **10 points**. The tracking speed of the entire system can reach up to **30 FPS**.

### Video demos on MOT challenge test set
<img src="assets/MOT15.gif" width="400"/>   <img src="assets/MOT16.gif" width="400"/>
<img src="assets/MOT17.gif" width="400"/>   <img src="assets/MOT20.gif" width="400"/>


## Installation
* Clone this repo, and we'll call the directory that you cloned as ${FAIRMOT_ROOT}
* Install dependencies. We use python 3.7 and pytorch >= 1.2.0
```
conda create -n FairMOT
conda activate FairMOT
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
cd ${FAIRMOT_ROOT}
pip install -r requirements.txt
cd src/lib/models/networks/DCNv2_new sh make.sh
```
* We use [DCNv2](https://github.com/CharlesShang/DCNv2) in our backbone network and more details can be found in their repo. 



## Baseline model

Baseline FairMOT model can be downloaded from here: DLA-34: [[Google]](https://drive.google.com/file/d/1kUF1qEbr8YHqWpq6Viyy_k7dNisQ8NNl/view?usp=sharing) [[Baidu, code: 88yn]](https://pan.baidu.com/s/1YQGulGblw_hrfvwiO6MIvA).
After downloading, you should put the baseline model in the following structure:
```
${FAIRMOT_ROOT}
   └——————models
           └——————all_dla34.pth
           └——————all_hrnet_v2_w18.pth
           └——————...
```


## Tracking
```
cd src
python script.py -mp ../models/all_dla34.pth -vp path_to_video -od path_to_save_video
```

## Acknowledgement
A large part of the code is borrowed from [Zhongdao/Towards-Realtime-MOT](https://github.com/Zhongdao/Towards-Realtime-MOT) and [xingyizhou/CenterNet](https://github.com/xingyizhou/CenterNet). Thanks for their wonderful works.

## Citation

```
@article{zhang2020simple,
  title={A Simple Baseline for Multi-Object Tracking},
  author={Zhang, Yifu and Wang, Chunyu and Wang, Xinggang and Zeng, Wenjun and Liu, Wenyu},
  journal={arXiv preprint arXiv:2004.01888},
  year={2020}
}
```

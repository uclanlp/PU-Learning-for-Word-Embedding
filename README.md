
## [Learning Word Embeddings for Low-resource Languages by PU Learning](https://arxiv.org/abs/1805.03366) ##
Chao Jiang, [Hsiang-Fu Yu](http://www.cs.utexas.edu/~rofuyu/), [Cho-Jui Hsieh](http://www.stat.ucdavis.edu/~chohsieh/rf/), [Kai-Wei Chang](http://web.cs.ucla.edu/~kwchang/). NAACL 2018

**Please run the code through [demo.sh](https://github.com/uclanlp/MFEmbedding/blob/master/demo.sh)**

**For details, please refer to [this paper](https://arxiv.org/pdf/1805.03366.pdf)**


- ### Abstract

Language is increasingly being used to define rich visual recognition problems with supporting image collections sourced from the web. Structured prediction models are used in these tasks to take advantage of correlations between co-occurring labels and visual input but risk inadvertently encoding social biases found in web corpora. For example, in the following image, it is possible to predict  the *place* is the **kitchen**, because it is the common place for the *activity* **cooking**. However, in subfigure 4, the model predicts the agent as a woman even though it is a man, which is caused by the inappropriate correlations between the activity **cooking** and the **female** gender.

| ![bias](img/bias_teaser.png)             |
| ---------------------------------------- |
| *Structure prediction can help the model to build the correlations between different parts. However it will also cause some bias problem.* |

In our work, we study data and models associated with multilabel object classification (MLC) and visual semantic role labeling (vSRL). We find that (a) datasets for these tasks contain significant gender bias and (b) models trained on these datasets further amplify existing bias. For example, the activity **cooking** is over 33% more likely to involve females than males in a training set, and a trained model further amplifies the disparity to 68% at test time. We propose to inject corpus-level constraints for calibrating existing structured prediction models and design an algorithm based on Lagrangian relaxation for collective inference. Our method results in almost no performance loss for the underlying recognition task but decreases the magnitude of bias amplification by 47.5% and 40.5% for multilabel classification and visual semantic role labeling, respectively.


- ### Source Code

We provide our calibration function in file "fairCRF_gender_ratio.ipynb". It is based on the Lagrangian Relaxation algorithm. You need to provide your own inference algorithm and also the algorithm you used to get the accuracy performance. The function also needs you to provide your own constraints. We give detailed description about the parameters in the [jupyter notebook](https://github.com/uclanlp/reducingbias/blob/master/src/fairCRF_gender_ratio.ipynb) and we also provide the running example for both vSRL and MLC tasks. 

> To run the vSRL task, you need to have [caffe](http://caffe.berkeleyvision.org/installation.html) installed in your machine.  If you just want to run with the sampled data, be sure to download the .prototxt files from the data/imSitu/ folder and put them to the folder ("crf\_path" in our case) in the same level where caffe is installed. All the other files are also provided under data/imSitu/. Remember to modify all the path in the config.ini file with absolute path.

- ### Data

We provide all the potential scores for MS-COCO dataset in data/COCO folder.  Also there is sampled potentials for imSitu dataset in data/imSitu folder. For complete imSitu potentials, download at [here](https://s3.amazonaws.com/MY89_Transfer/crf_only.tar).

- ### Reference
  Please cite

 ```
@inproceedings{JYHC18,
author    = {Chao Jiang and Hsiang-Fu Yu and Cho-Jui Hsieh and Kai-Wei Chang},
title     = {Learning Word Embeddings for Low-resource Languages by PU Learning}, 
booktitle = {NAACL}, 
year      = {2018},
              }
 ```

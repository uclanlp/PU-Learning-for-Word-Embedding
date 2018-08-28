
## [Learning Word Embeddings for Low-resource Languages by PU Learning](https://arxiv.org/abs/1805.03366) ##
Chao Jiang, [Hsiang-Fu Yu](http://www.cs.utexas.edu/~rofuyu/), [Cho-Jui Hsieh](http://www.stat.ucdavis.edu/~chohsieh/rf/), [Kai-Wei Chang](http://web.cs.ucla.edu/~kwchang/). NAACL 2018

**Please run the code through [demo.sh](https://github.com/uclanlp/MFEmbedding/blob/master/demo.sh)**

**For details, please refer to [this paper](https://arxiv.org/pdf/1805.03366.pdf)**


- ### Abstract

Word embedding is a key component in many downstream applications in processing natural languages. Existing approaches often assume the existence of a large collection of text for learning effective word embedding. However, such a corpus may not be available for some low-resource languages. In this paper, we study how to effectively learn a word embedding model on a corpus with only a few million tokens. In such a situation, the co-occurrence matrix is sparse as the co-occurrences of many word pairs are unobserved. In contrast to existing approaches often only sample a few unobserved word pairs as negative samples, we argue that the zero entries in the co-occurrence matrix also provide valuable information. We then design a Positive-Unlabeled Learning (PU-Learning) approach to factorize the co-occurrence matrix and validate the proposed approaches in four different languages.

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

- ### Acknowledgments

 * Thanks for [Omer Levy's](https://levyomer.wordpress.com/) hyperwords [code](https://bitbucket.org/omerlevy/hyperwords)!


## [Learning Word Embeddings for Low-resource Languages by PU Learning](https://arxiv.org/abs/1805.03366) ##
Chao Jiang, [Hsiang-Fu Yu](http://www.cs.utexas.edu/~rofuyu/), [Cho-Jui Hsieh](http://www.stat.ucdavis.edu/~chohsieh/rf/), [Kai-Wei Chang](http://web.cs.ucla.edu/~kwchang/). NAACL 2018

**Please run the code through [demo.sh](https://github.com/uclanlp/MFEmbedding/blob/master/demo.sh)**

**For details, please refer to [this paper](https://arxiv.org/pdf/1805.03366.pdf)**


- ### Abstract

Word embedding is a key component in many downstream applications in processing natural languages. Existing approaches often assume the existence of a large collection of text for learning effective word embedding. However, such a corpus may not be available for some low-resource languages. In this paper, we study how to effectively learn a word embedding model on a corpus with only a few million tokens. In such a situation, the co-occurrence matrix is sparse as the co-occurrences of many word pairs are unobserved. In contrast to existing approaches often only sample a few unobserved word pairs as negative samples, we argue that the zero entries in the co-occurrence matrix also provide valuable information. We then design a Positive-Unlabeled Learning (PU-Learning) approach to factorize the co-occurrence matrix and validate the proposed approaches in four different languages.

- ### Source Code

To reproduce result on text8 dataset, please run demo.sh file. It could automatically generate all results on the text8 dataset in our paper.

> To run the code, please use python 2.7.

- ### Data

We provide the testsets in four different languages: English, Czech, Danish and Dutch. Testsets in Czech, Danish and Dutch are translated into English by Google Translation API. They are in the testsets_in_different_languages folder.

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

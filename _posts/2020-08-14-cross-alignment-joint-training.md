---
layout: post
title: "Cross-Lingual Alignment vs. Joint Training: A Comparative Study and a Simple Unified Framework"
categories: [Paper Review]
featured-img: chains
tags: [Multilingual, NLP]
---

**Abstract:** Learning multilingual representations of text has proven a successful method for many cross-lingual transfer learning tasks. There are two main paradigms for learning such representations: (1) alignment, which maps different independently trained monolingual representations into a shared space, and (2) joint training, which directly learns unified multilingual representations using monolingual and cross-lingual objectives jointly. In this paper, we first conduct direct comparisons of representations learned using both of these methods across diverse cross-lingual tasks. Our empirical results reveal a set of pros and cons for both methods, and show that the relative performance of alignment versus joint training is task-dependent.

[Original Paper](https://arxiv.org/abs/1910.04708)

--

# Paper Summary

**Goal**  
1. evaluate and compare alignment versus joint training methods across three diverse tasks: BLI, cross-lingual NER, and unsupervised MT. 
2. propose a simple, novel, and highly generic framework that uses unsupervised joint training as initialization and alignment as refinement to combine both paradigms. 

**Cross-Lingual Representations**  
- Alignment Methods: independently train embeddings in different languages using monolingual corpora alone, and then learn a mapping to align them to a shared vector space using a projection matrix. Such a mapping can be trained in a supervised fashion using parallel resources such as bilingual lexicons or even in an unsupervised manner based on distribution matching. 
-	Limitations: (1) disjoint sets of embeddings & lack of cross-lingual constraints at fine-tuning stage (2) assumes isomorphism of monolingual embedding spaces 

- Joint Training Methods: optimize a monolingual objective predicting the context of a word in a monolingual corpus along with either a hard or soft cross-lingual constraint. Unsupervised joint learning: (1) Construct a joint vocabulary that is shared across two languages (2) Concatenate the training corpora and learn an embedding set corresponding to joint vocabulary 
-	Limitations: (1) assumes all shared words serve implicitly as anchors and thus need not be aligned to other words – oversharing (2) does not utilize explicit form of seed dictionary – less accurate alignments (esp for words not shared) 

**Alignment vs. Joint Training**  
- alignment methods significantly outperform the joint training approach by a large margin in all language pairs for both BLI and NER. However, the unsupervised joint training method is superior than its alignment counterpart on the unsupervised MT task 

**Proposed Framework:** first use unsupervised joint training as a coarse initialization and then apply alignment methods for refinement 
1.	Joint Initialization: unsupervised joint training 
2.	Vocab Reallocation: “unshare” overshared words, so their embeddings can be better aligned in the next step 
3. Alignment Refinement: utilize alignment method to refine alignments across the non-sharing embedding sets

![illustration](https://raw.githubusercontent.com/thespectrewithin/joint_align/master/illustration.png)

**Contextualized Representations:** while our vocab reallocation technique is no longer necessary as contextualized representations are dependent on context and thus dynamic, we can still apply alignment refinement on extracted contextualized features for further improvement.

<br>

# My Thoughts

- A good paper that overviews alignment and joint training. Easy to understand for beginners to the subject or those curious to find out more about such cross-lingual learning methods
- Kind of an old-school method to cross-lingual learning and I can't remember if they compared it to more recent methods of cross-lingual learning such as knowledge distillation. Would be interesting to compare the two.

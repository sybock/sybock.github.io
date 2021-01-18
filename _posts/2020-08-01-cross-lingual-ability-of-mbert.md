---
layout: post
title: "Cross-Lingual Ability of Multilingual BERT: An Empirical Study"
featured-img: Multilingual
categories: [Paper Review]
tags: [BERT]
---

**Abstract:** Recent work has exhibited the surprising cross-lingual abilities of multilingual BERT (M-BERT) -- surprising since it is trained without any cross-lingual objective and with no aligned data. In this work, we provide a comprehensive study of the contribution of different components in M-BERT to its cross-lingual ability. 

[Original Paper](https://arxiv.org/abs/1912.07840)

--

# Paper Summary

**Goal:** Analyze two-languages version of M-BERT (B-BERT for bilingual BERT) in three dimensions (1) linguistic properties and similarities (2) network architecture (3) input and learning objective


**What is M-BERT?**  
• Pre-trained the same way as BERT but using Wikipedia text from top 104 languages  
• M-BERT is contextual and training requires no supervision -- no alignment between languages  
• M-BERT produces representation that seems to generalize well cross languages for a variety of downstream tasks


## (1) Linguistic Properties 

**Method:**  
• Fake English corpus created -- different language than English but having the exact same properties except word surface forms

**Results:**  
• Word-piece Overlap: no significance  
• Word-ordering similarity: performance drops, but still above baseline (other components of structural similarity contributing)  
• Word-frequency similarity: no significance  
• Structural Similarity (morphology, word-ordering, word frequency): some relation but definition of structural similarity too unclear  

## (2) Network Architecture

**Method:**  
• Variation only in the parameter under examination, other parameters are kept constant

**Results**  
• Depth: more depth helps model extract good semantic and structural features on English and cross-lingually  
• Multi-head Attention: # of attention heads doesn't have a significant effect on cross-lingual ability  
• Total # of Parameters: not as significant as depth, but B-BERT requires certain minimum number of parameters to extract good semantic and structural features  

## (3) Input and Learning Objectives

**Results**  
• Next Sentence Prediction: NSP objective hurts cross-lingual performance  
• Language Identity Marker: adding language identity marker doesn't affect cross-lingual performance  
• Characters vs. Word-piece vs. Word: Word-piece and Word tokenized input performs better than character tokenized input bc word-pieces and words carry much more information  


**Further Work**  
• Extending study to M-BERT  
• Better understand Structural Similarity as its significance is proven in cross-lingual ability


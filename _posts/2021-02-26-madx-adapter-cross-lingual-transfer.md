---
layout: post
title: "MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer"
categories: [Paper Review]
featured-img: science
tags: [NLP, Adapters]
---

**Abstract:** The main goal behind state-of-the-art pre-trained multilingual models such as multilingual BERT and XLM-R is enabling and boot- strapping NLP applications in low-resource languages through zero-shot or few-shot cross- lingual transfer. However, due to limited model capacity, their transfer performance is the weakest exactly on such low-resource languages and languages unseen during pretraining. We propose MAD-X, an adapter-based framework that enables high portability and parameter-efficient transfer to arbitrary tasks and languages by learning modular language and task representations. In addition, we introduce a novel invertible adapter architecture and a strong baseline method for adapting a pretrained multilingual model to a new lan- guage. 

[Original Paper](https://arxiv.org/abs/2005.00052)

**Authors:** Jonas Pfeiffer, Ivan Vuli´, Iryna Gurevych, Sebastian Ruder

--

# Paper Summary

**Goal**
1. Train language-specific adapter modules via masked language modelling (MLM) on unlabelled target language data
2. Train task-specific adapter modules via optimising a target task on labelled data in any source language
3. Propose invertible adapters, a new type of adapter that is well suited to performing MLM in another language to deal with mismatch between the shared multilingual vocabulary and target language vocabulary

**Contributions**
1. We propose MAD-X, a modular framework that mitigates the curse of multilinguality and adapts a multilingual model to arbitrary tasks and languages. Both code and adapter weights are integrated into the AdapterHub.ml repository (Pfeiffer et al., 2020b).
2. We propose invertible adapters, a new adapter variant for cross-lingual MLM
3. We demonstrate strong performance and robustness of MAD-X across diverse languages and tasks. MAD-X outperforms the baselines on seen and unseen high-resource and low-resource languages.
4.  We propose a simple and more effective base- line method for adapting a pretrained multilingual model to target languages
5. We shed light on the behaviour of current methods on languages that are unseen during multilingual pretraining

## Multilingual Model Adaptation for Cross-lingual Transfer

**Target Language Adaptation**
- fine-tune a pretrained multilingual model via MLM on unlabelled data of the target language prior to task-specific fine-tuning in the source language
- Disadvantage: this approach is that it no longer allows us to evaluate the same model on multiple target languages as it biases the model to a specific target language
- But it does not result in catastrophic forgetting of the multilingual knowledge already available in the pretrained model that enables the model to transfer to other languages.

**Adapters for Cross-Lingual Transfer**
![Figure 1](https://d3i71xaburhd42.cloudfront.net/7813ce9379fc76e83e3ece87ae2129dcdd25ca9c/3-Figure1-1.png)

- 3 types of adapters: language, task, and invertible adapters

![Formula](https://ibb.co/tBs9sNP)

- Language Adapters: simple down- and up-projection combined with residual connection
    - Down-projection *D* (hxd) where *h* is the hidden size of the Transformer model and *d* is the dimension of the adapter
    - ReLU activation and up-projection *U* (dxh) at every later *l*
    - Residual connection *r* is the output of the Transformer's feed-forward layer whereas *h* is the output of the subsequent layer normalization 

- Task Adapters: same architecture as lagnuage adapters but stacked on top of the language adapter and thus reciece the output of the language adapter *LA* as input, with residual *r* of the Transformer's feed-forward layer.
    - Output then passed to layer nomalization
    - Task adapters are the only parameters that are updated when training on a downstream task
    - Aim to capture knowledge that is task-specific but generalises across languages

![Figure 2](https://d3i71xaburhd42.cloudfront.net/7813ce9379fc76e83e3ece87ae2129dcdd25ca9c/4-Figure2-1.png)

- Invertible Adapters: mitigate  mismatch between multi-lingual and target language vocabulary
    -  Majority of the “parameter budget” of pre- trained multilingual models is spent on token embeddings of the shared multilingual vocabulary
    - Stacked on top of embedding layer while their respective inverses precede the output embedding layer
    - Invertibility allows us to leverage the same set of parameters for adapting both input and output representations.
    - The invertible adapter has a similar function to the language adapter, but aims to capture token-level language-specific transformations. As such, it is trained together with the language adapters using MLM on unlabelled data of a specific language

## Experiments

**Task:** NER, QA, Causal Commensese Reasoning (CCR)

**Languages:** Four categories: 1) high-resource languages and 2) low-resource languages covered by the pretrained SOTA multilingual models (i.e., by mBERT and XLM-R); as well as 3) low-resource languages and 4) truly low-resource languages not covered by the multilingual models.

**MAD-X: Experimental Setup**
- we learn language adapters, invertible adapters, and task adapters with dimensionalities of 384, 192 (384 for both directions), and 48, respectively.
- NER: five runs of fine-tuning on the WikiAnn training set of the source language— except for XLM-RBase MLM-TRG for which we conduct one run for efficiency purposes for every source language–target language combination
- QA: three runs of fine-tuning on the English SQuAD training set, evaluate on all XQuAD target languages, and report mean F1 and exact
match (EM) scores.
- CCR:  three runs of fine-tuning on the respective English train- ing set, evaluate on all XCOPA target languages, and report accuracy scores.

## Results and Discussion

**NER**
- MAD-X without language and invertible adapters performs on par with XLM-R for almost all languages present in the pretraining data
- However, looking at unseen languages, the performance of MAD-X that only uses task adapters deteriorates significantly compared to XLM-R.
- Adding language adapters to MAD-X improves its performance across the board, and their use- fulness is especially pronounced for low-resource languages.
- Even for high-resource languages, the addition of language-specific parameters yields substantial improvements.

**CCR**
-  Target language adaptation outperforms XLM-RBase while MAD-XBase achieves the best scores.
-  It shows gains in particular for the two unseen languages, Haitian Creole (ht) and Quechua (qu). Performance on the other languages is also generally competitive or better.

**QA**
- transferring from English to each target language
-  MAD-X achieves similar performance to the XLM-R baseline
- We note that all languages included in XQuAD can be considered high-resource
- MAD-X excels at transfer to unseen and low- resource languages, it achieves competitive perfor- mance even for high-resource languages and on more challenging tasks.
- These evaluations also hint at the modularity of the adapter-based MAD-X approach, which holds promise of quick adaptation to more tasks

## Further Analysis

- **Impact of Invertible Adapters:** Invertible adapters improve performance for many transfer pairs, and particularly when transferring to low- resource languages.
- **Sample Efficiency:**  due to the modularity of MAD-X, once trained, these adapters have an advantage of being directly reusable (i.e., “plug- and-playable”) across different tasks


# My Thoughts
- Modular training in machine learning is becoming more common, as it is seemingly more parameter-efficient and computationally fast. It's interesting to see this method adatped to cross-lingual transfer learning. I wonder whether it will pick up and become the industry norm. I do think it is an efficient way to use one model across a variety of languages and tasks. However, in my recent experiments, I've been having difficulty achieving the same level of accuracy as they do in this paper. Further experiments are obviously needed.
- Not sure that I completely understood the mechanics of the inverse adapter, but I semi-understood the need for it. I think I will have to re-read this section again and do extra research on NICE on why having this invertible module helps with parameter efficiency.
- A cool paper overall, but I do think one disadvantage of having all these modular bits is that they are difficult to keep track of and that you have to first train the source and target languages independently for it to work the way it did in this paper. Why can't XLM-R be stacked with just the target language adapter and stacked with the task adapter? Also will have to experiment with this.
- One note is that while the authors did a great job creating AdapterHub.ml repository with all the code and documentation, the current version is not compatible with the current Hugging Face interaface. The example code has to be edited here and there, which is kind of a minor inconvience but still. Also, I wish that they had more detailed documentation and example training codes on their github. 
- I had sort of put this adapter training experiment on hold because I wasn't sure how to carry on, but re-reading the paper and taking notes on it has sort of renewed my interest in the subject and has given me some direction as to how to conduct further experiments... Now for the hard part: actually writing the code.

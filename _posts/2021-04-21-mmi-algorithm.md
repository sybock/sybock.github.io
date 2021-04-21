---
layout: post
title: "Mutual Information and Diverse Decoding Improve Neural Machine Translation"
categories: [Algorithm]
featured-img: light
tags: [Translation, NLP]
mathjax: true
---

**Abstract**: Sequence-to-sequence neural translation models learn semantic and syntactic relations between sentence pairs by optimizing the likelihood of the target given the source, i.e., $p(y\|x)$, an objective that ignores other potentially useful sources of information. We introduce an alternative objective function for neural MT that maximizes the mutual information between the source and target sentences, modeling the bi-directional dependency of sources and targets. We implement the model with a simple re-ranking method, and also introduce a decoding algorithm that increases diversity in the N-best list produced by the first pass. Applied to the WMT German/English and French/English tasks, the proposed models offers a consistent performance boost on both standard LSTM and attention-based neural MT architectures.

[Original Paper](https://arxiv.org/abs/1601.00372)

**Authors:** Liu, Jurafsky 2016

--

# Paper Review

## Introduction

- Seq2Seq models for machine translation
    - Encoder-decoder network, in which a source sentence input $x$ is mapped (encoded) to a continuous vector representation from which a target output $y$ will be generated (decoded)
    - Optimized through maximizing the log-likelihood of observing the paired output $y$ given $x$:
        $Loss = -log p(y \| x)$
- Problem: standard Seq2Seq models capture unidirectional dependency from source to target i.e. $p(y \| x)$ but ignores $p(x \| y)$, the dependency from the target to the source, which has long been an important feautre in phrase-based translation (which also combine other features such as sentence length)
- Proposal: incorporate this bi-directional dependency and model the maximum mutual information (MMI) between source and target into SEQ2SEQ models.
- MMI based objective function is equivalent to linearly combining $p(x \| y)$ and $p(y \| x)$. With tuning weight $\lambda$, such a loss function can be written as:
    ![equation](https://i.ibb.co/XDW1YB5/Screen-Shot-2021-04-21-at-1-01-42-PM.png)

- But direct decoding from equation (2) is infeasible because computing $p(x \| y)$ cannot be done until the target has been computed.
- Solution: reranking approach to _approximate_ the mutual information between source and target in neural machine translation models. We separately trained two Seq2Seq models, one for $p(y \| x)$ and one for $p(x \| y)$. 
    - $p(y \| x)$ model is used to generate N-best lists from the source sentence $x$
    - $p(x \| y)$ is used to rerank the lists
    - diversity-promoting decoding model tailored to neural MT systems

## Background: Neural Machine Translation

**Definition**
Neural machine translation models map source $x = {x_1, x_2, ...x_N}$ to a continuous vector representation, from which target output $y = {y_1, y_2, ..., y_N}$ is to be generated.

**LSTM Models**
- A long-short term memory model (Hochreiter and Schmidhuber, 1997) associates each time step with an input gate, a memory gate and an output gate, denoted respectively as $i_t, f_t and o_t$.
- The LSTM defines a distribution over outputs $y$ and sequentially predicts tokens using a softmax function
- During decoding, the algorithm terminates when an $EOS$ token is predicted

**Attention Models**
- Attention models adopt a look-back strategy that links the current decoding stage with input time steps to represent which portions of the input are most responsible for the current decoding state 
- Let $H = {ˆh_1, ˆh_2, ..., ˆh_N}$ be the collection of
hidden vectors outputted from LSTMs during encoding. Each element in $H$ contains information about the input sequences, focusing on the parts surrounding each specific token.
- Attention models link the current-step decoding information, i.e., $h_t$ with each of the representations at decoding step $ˆh_{t'}$ using a weight variable $a_t$. 
- $a_t$ can be constructed from different scoring functions such as  _dot product_ or _concatenation_
- General strategy: dot product then average weights over all input time-steps

**Unknown Word Replacements**
-  From the attention models, we obtain word alignments from the training dataset, from which a bilingual dictionary is extracted. At test time, we first generate target sequences. Once a translation is generated, we link the generated UNK tokens back to positions in the source inputs, and replace each UNK token with the translation word of its correspondent source token using the pre-constructed dictionary.
- For SEQ2SEQ models, we first generate translations and replace UNK tokens within the translations using the pre-trained attention models to post-process the translations.

## Mutual Information via Reranking
Approximation approach:
1. Train $p(y\|x)$ and $p(x\|y)$ separately using vanilla SEQ2SEQ models or Attention models.
2. Generate N-best lists from $p(y\|x)$.
3. Rerank the N-best list by linearly adding $p(x\|y)$.

**Standard Beam Search for N-best lists**
- N-best lists are generated using a beam search decoder with beam size set to K = 200 from $p(y\|x)$ models. 
- We set the minimum length and maximum length to 0.75 and 1.5 times the length of sources.
- Beam size $N$ is set to 200
- To be specific, at each time step of decoding, we are presented with $K × K$ word candidates. We first add all hypotheses with an EOS token being generated at current time step to the N-best list. Next we preserve the top K unfinished hypotheses and move to next time step. 
- We therefore maintain batch size of 200 constant when some hypotheses are completed and taken down by adding in more unfinished hypotheses. This will lead the size of final N-best list for each input much larger than the beam size.

**Generating a Diverse N-best List**
- Unfortunately, the N-best lists outputted from standard beam search are a poor surrogate for the entire search space because most generated translations are similar, differing only by punctuation or minor morphological variations --> lack of diversity will decrease impact of our reranking process
- Change the way score for N-best list is computed
    - Rewrite the score by adding an additional part $\gamma k'$ where $k'$ denotes the ranking of the current hypothesis among its siblings
    - additional term $\gamma k'$ punishes bottom ranked hypotheses among siblings (hypotheses descended from the same parent)
    - model gives more credit to top hypotheses from each of different ancestors
    - For instance, even though the original score for _it is_ is lower than _he has_, the model favors the former as the latter is more severely punished by the intra-sibling ranking part $\gamma k'$

![fig1](https://d3i71xaburhd42.cloudfront.net/1cc0c322af508a8f7b6ea9705c9023c78bc7bc6f/5-Figure1-1.png)

**Reranking**
- The generated N-best list is then reranked by linearly combining $log p(y\|x)$ with $log p(x\|y)$.
- The score of the source given each generated translation can be immediately computed from the previously trained $p(x\|y)$.
- We also consider $log p(y)$, which denotes the average language model probability trained from monloingual data


## Experiments
- Train data: WMT'14 dataset containing 4.5 million pairs for English-German and German-English translation and 12 million pairs for English-French translation.
- Dev Data (Eng-De): newstest2013, eval in BLEU
- Dev Data (Eng-Fr): newstest2012, news-test-2013
- Eval: new-test-2014

**English-German Results**
- We reported progressive performances as we add in more features for reranking.
- Among all the features, reverse probability from mutual information (i.e., p(x|y)) yields the most significant performance boost, +1.4 and +1.1 for standard SEQ2SEQ models without and with unknown word replacement, +0.9 for attention models
- we observe consistent performance boost introduced by language model.
- We see the benefit from our diverse N-best list by comparing *mutual+diversity* models with *diversity* models. On top of the improvements from standard beam search due to reranking, the diversity models introduce additional gains of +0.4, +0.3 and +0.3, leading the total gains roughly up to +2.6, +2.6, +2.1 for different models.

![en-de](https://d3i71xaburhd42.cloudfront.net/1cc0c322af508a8f7b6ea9705c9023c78bc7bc6f/7-Table1-1.png)

**French-English Results**
- We again observe that applying mutual information yields better performance than the corresponding standard neural MT models.
- Relative to the English-German dataset, the English-French translation task shows a larger gap between our new model and vanilla models where reranking information is not considered


# My Thoughts
- DialoGPT 에 사용된 MMI가 뭔지 궁금해서 읽어본 논문
- Beam Search에 대해서도 잘 몰랐는데 자세히 설명되어 있어서 좋았다. Sister간에 hypotheses를 최소화 하면서 더 다양한 output을 generate하는 모델. 간단한 알고리즘! 구현 방법도 이렇게 간단한가... 코드를 보면서 더 공부를 해봐야될듯 
---
layout: post
title: "Margin-based Parallel Corpus Mining with Multilingual Sentence Embeddings"
categories: [Paper Review]
featured-img: miners
tags: [Multilingual, Seq2Seq]
mathjax: true
---

**Abstract**: Machine translation is highly sensitive to the size and quality of the training data, which has led to an increasing interest in collecting and filtering large parallel corpora. In this paper, we propose a new method for this task based on multilingual sentence embeddings. In contrast to previous approaches, which rely on nearest neighbor retrieval with a hard threshold over cosine similarity, our proposed method accounts for the scale inconsistencies of this measure, considering the margin between a given sentence pair and its closest candidates instead. Our experiments show large improvements over existing methods. We outperform the best published results on the BUCC mining task and the UN reconstruction task by more than 10 F1 and 30 precision points, respectively. Filtering the English-German ParaCrawl corpus with our approach, we obtain 31.2 BLEU points on newstest2014, an improvement of more than one point over the best official filtered version.

[Original Paper](https://www.aclweb.org/anthology/P19-1309)

**Authors:** Mikel Artetxe and Holger Schwenk 2019 (ACL)

--

# Paper Summary

## Introduction
- This paper is important because effective approaches to mine and filter parallel corpora are crucial to apply NMT in practical settings.
- Traditional approahces to parallel corpus mining: relied on heavily engineered systems
    - early approahces: based on metadata information from web crawls
    - recent methods: focus on textual content, relying on cross-lingual docuemnt retrieval or machine translation
    - more recent methods: use multilingual sentence embeddings. These methods use an NMT inspired encoder-decodre to train sentence embeddings on existing parallel data, which are then directly applied to retrieve and filter new parallel sentences using nearest neighbor retrieval over cosine similarity with a hard threshold
- This retrieval method (above) suffers from the scale of cosine similarity not being globally consistent
    - Table 1: some sentences without any correct translation have high scores, making them rank higher than other sentences with a correct ranslation
- This paper's contribution: tackles the aboec issue by considering the **margin** between the cosine of a given sentence pair and that of its respective *k* nearest neighbors.

![Table1](https://d3i71xaburhd42.cloudfront.net/30b09a853ab72e53078f1feefe6de5a847a2b169/2-Table1-1.png)

## Multilingual Sentence Embeddings

- Encoder-decoder architecture to learn mulitilingual sentence embeddings based on Schwenk (2018). 
    - Encoder: bidirectional LSTM, sentence embedding obtained by applying max-pooling operation over its output
    - Embeddings are fed into decoder to...
        1. initialize decoder hidden and cell state after linear transformation
        2. concatenated to input embeddings at every time step
    - Joing 40k BPE vocab learned on the concatenation of all training corpora
    - Encoder is fully language agnostic while decoder receives an output language ID embdding at every time step
- Training Parameters
    - 4 GPUs
    - batch size of 48,000 tokens
    - Adam with learning rate of 0.001 and droupout 0.1
    - Single layer for encoder and decoder w. hidden size of 512 and 2048 respectively, yielding 1024 dimensional sent embeddings
- After training, the decoder is discarded and the encoder is used to map a sentence to a fixed-length vector

## Scoring and filtering parallel sentences
- The multilingual encoder can be used to mine parallel sentences by taking the nearest neighbor of each source sentence in the target side according to cosine similarity, and filtering those below a fixed threshold.
- **Problem**: we argue that it suffers from the scale of cosine similarity not being globally consistent across different sentences. For instance, Table 1 shows an example where an incorrectly aligned sentence pair has a larger cosine similarity than a correctly aligned one, thus making it impossible to filter it through a fixed threshold. In that case, all four nearest neighbors have equally high values.
- **Solution**:  In contrast, for example B, there is a big gap between the nearest neighbor and its other candidates. As such, we argue that the margin between the similarity of a given candidate and that of its *k* nearest neighbors is a better indicator of the strength of the alignment.

**Margin-basd scoring**
- We consider the margin between the cosine of a given candidate and the average cosine of its *k* nearest neighbors in both directions as follows:
    $score(x,y) = margin (cos(x,y),$
    $\sum_{z ∈ NN_{k}(x)} \frac{cos(x,z)}{2k} + \sum_{z ∈ NN_{k}(y)} \frac{cos(y,z)}{2k} )$

- $NN_{k}(x)$ denotes $k$ nearest neighbors of $x$ in the other language excluding duplicates, and analogously for $NN_{k}(y)$

- Variants of general definition:
    1. Absolute $(margin(a,b) = a)$: ignoring the average, equivalent to cosine similarity, baseline
    2. Distance $(margin(a,b) = a - b): subtracting the average cosine similarity from that of the given candidate. 
    3. Ratio $(margin(a,b) = \frac{a}{b})$: ration between the candidate and the average cosing of its nearest neighbors in both directions

**Candidate generation and filtering**
1. Forward: Each source sentence is aligned with exactly one best scoring sentence. Some targe sentences may be aligned with multiple source sentences or with none.
2. Backward: Equivalent to the forward strategy but going in the opposite direction
3. Intersection: of forward and backward candidates, which discards sentences with inconsistent alignments.
4. Max. score: Combination of forward and backward candidates, where those with the highest score is selected.

## Experiments and results

**BUCC mining task**
- Building and Using Comparable Corpora (BUCC): evaluation for bitext mining; task is mining for parallel sentences between English and four foreign languages: German, French, Russian, Chinese

    ![BUCC](https://d3i71xaburhd42.cloudfront.net/30b09a853ab72e53078f1feefe6de5a847a2b169/4-Table2-1.png)

- Both of our bidirectional retrieval strategies achieve substantial improvements over this baseline while still relying on cosine similarity, with *intersection* giving the best results.
- Moreover, our proposed margin-based scoring brings large improvements when using either the distance or the ratio functions
- The best results are achieved by *ratio*, which outperforms distance by 0.3-0.5 points. 
- Interestingly, the retrieval strategy has a very small effect in both cases, suggesting that the proposed scoring is more robust than cosine.

**UN corpus reconstruction**
- Task: mimic experiment from Guo et al. (2018) where they align 11.3M sentences of the UN corpus
- Outperforms Guo by a large maring despite using only a fraction of the training data

**Filtering ParaCrawl for NMT**
- Filter English-German ParaCrawl corpus and evaluate NMT models trained on the
- NMT model: `fairseq`'s implementation of the big transformer model (Vaswani et al. 2018)
- Train for 100 epochs with same config as Ott et al., 2018, use newstest2013 and newstest2014 as dev sets
- Use beam size of 5 using ensemble of the last 10 epochs
- ParaCrawl 4.59 B sent corpus is reduced to 64.4 M sentences (deduplication, sentences with less than 3 more than 80 tokens are removed)
- Each sentence pair is scored with *ratio* function, processing the entire corpus in batches of 5 million sentences, and take the top scoring entries up to the desired size
- Outperforms all previous systems but Edunov et al. (2018) who uses a large in-domain monoloingual corpus through back-translation, making both works complementary
- Full system outperforms Ott et al. (2018) by nearly 2 points, despite using the same configuration and training data

![pc](https://d3i71xaburhd42.cloudfront.net/30b09a853ab72e53078f1feefe6de5a847a2b169/5-Table5-1.png)

## Conclusions and future work
- Contribution: In this paper, we propose a new method for parallel corpus mining based on multilingual sentence embeddings. We use a sequence-to-sequence architecture to train a multilingual sentence encoder on an initial parallel corpus, and a novel margin based scoring method that overcomes the scale inconsistencies of cosine similarity.
- The code of this work is freely available as part of the [LASER toolkit](https://github.com/facebookresearch/LASER), together with an additional single encoder which covers 93 languages.

# My Thoughts
- Multilingual model 학습을 하는데 데이터가 부족하다 ㅠㅠ 직접 마이닝을 하려고 하니 좀 막막하긴 한데 페이스북에서 코드를 공개했다고 하니 한 번 가서 써봐야겠다. 휴... 언제 하지...? ㅎㅎ... 
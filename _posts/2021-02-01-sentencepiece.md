---
layout: post
title: "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing"
categories: [Paper Review]
featured-img: Puzzle
tags: [NLP, Tokenizer]
---

**Abstract:** This paper describes SentencePiece, a language-independent subword tokenizer and detokenizer designed for Neural-based text processing, including Neural Machine Translation. It provides open-source C++ and Python implementations for subword units. While existing subword segmentation tools assume that the input is pre-tokenized into word sequences, SentencePiece can train subword models directly from raw sentences, which allows us to make a purely end-to-end and language independent system. We perform a validation experiment of NMT on English-Japanese machine translation, and find that it is possible to achieve comparable accuracy to direct subword training from raw sentences.

[Original Paper](https://www.aclweb.org/anthology/D18-2012/)

--

# Paper Summary

**Goal**: develop a simple, efficient, reproducible and language independent pre- and post- processor that can easily be integrated into Neural Network-based NLP systems, including NMT.

## System Overview
- Normalizer:  normalize semantically equivalent Unicode characters into canonical forms.
- Trainer: trains the subword segmentation model from the normalized corpus. We specify a type of subword model as the parameter of Trainer. 
- Encoder: internally executes Normalizer to nor- malize the input text and tokenizes it into a sub- word sequence with the subword model trained by Trainer. (tokenization)
- Decoder: converts the subword sequence into the normalized text. (detokenization)

## Library Design

![lossless_tokenization](https://d3i71xaburhd42.cloudfront.net/b5246fa284f86b544a7c31f050b3bd0defd053fd/2-Figure1-1.png)

- Lossless tokenization: implmenting the deocder as an inverse operation of Encoder
    - treat the input text just as a sequence of Unicode characters. Even whitespace is handled as a normal symbol. 
    - allows tokenizer to be used across languages with whitespace (English) and without whitespace(Chinese, Korean etc.) without having to manually code differences
- Efficient subword training and segmentation: given an input sentence (or word) of length N , SentencePiece adopts an O(N log(N )) algorithm in which the merged symbols are managed by a binary heap (priority queue). In addition, the training and segmentation complexities of unigram language models are linear to the size of input data.
- Vocab id management: final size of vocab specified before training --> applicable to other segmentation algorithms
- Custom character normalization: By default, SentencePiece normalizes the input text with the Unicode NFKC normalization. SentencePiece also supports custom normalization rules defined as a TSV file.
- Self-contained models: For perfect reproducibility, SentencePiece model is designed to be purely self-contained. The model file includes not only the vocabulary and segmentation parameters, but also the pre-compiled finite state transducer for character normalization.
- Library API for on-the-fly processing: SentencePiece not only provides a standalone command line tool for off-line preprocessing but supports a C++, Python and Tensorflow library API for on-the-fly processing, which can easily be integrated into existing NMT frameworks.

## Experiments
![Results](https://d3i71xaburhd42.cloudfront.net/b5246fa284f86b544a7c31f050b3bd0defd053fd/5-Table1-1.png)
- GNMT (Wu et al., 2016) as the implementation of the NMT system in our experiments. 
-  subword segmentations with SentencePiece consistently improve the BLEU scores compared to the word model.
- pre-tokenization is not always necessary to boost the BLEU scores.
- We can find larger improvements in BLEU when 1) SentencePiece is applied to Japanese, and 2) the target sentence is Japanese.

![Results2](https://d3i71xaburhd42.cloudfront.net/b5246fa284f86b544a7c31f050b3bd0defd053fd/6-Table2-1.png)
-  training and segmentation speed of both SentencePiece and subword-nmt is almost comparable on English data set regardless of the choice of pre-tokenization. 
- larger performance improvements when applying it to raw Japanese data (w/o pre-tok).
- SentencePiece is fast enough to be applied to raw data and the pre-tokenization is not always necessary. Consequently, SentencePiece helps to build a purely data-driven and language-independent system.    

# My Thoughts
- Tokenizer 논문을 처음 읽는 거라 모르는 부분이 조금 있어서 계속 찾아보면서 읽느라 오래 걸렸지만 재미있는 주제인거 같다. 특히 요즘 전처리를 하는데 계속 속도, 메모리 효율에 대해서 고민을 하는 중에 읽어서 이 모델이 얼마나 대단한건지 알 수 있다. 기존 subword-nmt와 시간 차이가 굉장히 많이 나서 놀랐다. 논문에 구현에 대한 부분은 자세히 나와있지 않기 때문에 깃헙을 들어가서 코드를 한 번 살펴봐야겠다는 생각을 했다. 
- 특히 이 논문의 핵심인 거 같은 lossless tokenization이 뭔지 잘 모르겠다. Decoder가 Encoder의 반대인 점은 모든 토크나이저에 해당되는 게 아닌가...? 이 부분도 깃헙 코드를 보거나 검색을 해서 더 알아보고 싶은 부분이다.
- Multilingual 모델의 흐름을 잘 읽어서 낸 논문 같다. 여러 언어를 하나의 모델로 해결할 수 있다는 건 굉장히 중요한 일이다. 모든 모델에 토크나이저만 갖다 바꾸면 학습을 쉽게 할 수 있는 장점이 있기 때문이다. 
- \s 를 _ 토큰으로 구현하는 건 되게 신박한 발상같다. 간단하면서 효율적이다. 논문을 읽을 수록 잘 나가는 모델?토크나이저?는 다 생각보다 간단한데 아무도 해보지 않은 것들이다. 약간 이렇게 한번 그냥 해보자!를 해봤는데 너무 잘 돼서 논문 쓴 느낌
---
layout: post
title: "Attention is All You Need"
categories: [Paper Review, NLP Model]
featured-img: transformers
tags: [NLP, Transformer, Attention]
mathjax: true
---

**Abstract:** The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

[Original Paper](https://arxiv.org/abs/1706.03762)

**Authors:** Vaswani et al. 2017

--

# Paper Summary

## Introduction
- **Issue**: RNN, LSTMs, GRUs are SoTA in NLP but the inherently sequential nature of these models precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples
- **Attention:** have become an integral part of sequence modeling and transduction models in various tasks(in conjunction with a reccurent network), allowing modeling of depenencies without regard to their distance in the input or output sequences
- **Transformer:** a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependences between input and output. The Transformer architecture allows for significantly more parllelization and can reach a new state of the art on various tasks.

## Background
- **Self Attention:** an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence
- The Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence- aligned RNNs or convolution.

## Model Architecture
- Previous Models: encoder-decoder structure where encoder maps input sequence of symbol representations $(x_1,..., x_n)$ to a sequence of continuous representaitons $z = (z_1, ..., z_n)$. Given $z$, the decoder then generates an output sequence $(y_1, ..., y_n)$ of symbols one element at a time. At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next.

- **Transformer:** follows overall encoder-decoder architecture with stacked self-attention and point-wise, fully connected layers for both the encoder and decoder

![transformer](https://camo.githubusercontent.com/8e489fab63c274c0dbbd3e882c0b9044f74392a1c0bda92393839796d44d621f/687474703a2f2f696d6775722e636f6d2f316b72463252362e706e67)


**Encoder:** 
    - N = 6 identical layers
    - 1 layer has two sub-layers (1) multi-head self attention, (2) feed-forward network
    - residual connection around each two sub-layers followed by layer normalization = $LayerNorm(x + Sublayer(x))$
    - $d_model = 512$
**Decoder:**
    - N = 6 identical layers
    - 1 layer has three sub-layers (1) masked encoder-decoder attention (2) multi-head self attention, (3) feed-forward network
    - residual connection around each two sub-layers followed by layer normalization 
    - masking ensures that predictions for position *i* can depende only on the known outputs at positions less than *i*.

**Attention:** mapping a query and a set of key-value pairs to an output, where query, keys, values, and output are all vectors
- **Scaled Dot Product Attention:** 
    - $Attention(Q,K,V) = softmax({\frac{QK^T}{\sqrt{d_k}}})V$
    - dot-product over additive attention bc dot-product is faster and more space-efficient in practice, implemented using highly optimized matrix multiplication code
    - scaled bc for large values of $d_k$ the dot products grow large in magnitude, pushing softmax to extremely small gradients -> scaling solves vanishing gradient
- **Multi-Head Attention:**  linearly project the queries, keys and values *h* times with different, learned linear projections to $d_k , d_k and d_v$ dimensions, respectively.
    - $MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O$
    - where $head_i = Attention(QW^Q, KW^K, VW^V)$
    - h = 8
    - d/h = 64

**Feed-Forward Networks**<br>
    - $FFN(x) = max(0, xW_1 + b_1)W_2 + b_2$<br>
    - two linear transformations with a ReLU activation in between
    - $d_{model} = 512$ and inner-layer has dimensionality $d_{ff} = 2048$

**Embeddings and Softmax**: used to convert decoder output to predicted next-token probabilities

**Positional Encoding:** 
    - Model has no recurrence/convolution -> must inject information about relative or absolute position of tokens in the sequence
    - Positional Encodings added to input embeddings at encoder and decoder stacks
    - Use sine and cosine functions of different frequencies to encode positions

## Why Self-Attention?
- Motivations for using self-attention
    1. total computational complexity per layer
    2. amount of computation that can be parallelized
    3. path length between long-range dependencies in the network
    4. models more interpretable

## Training
- Data: WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs
- Hardware: 8 NVIDIA P100 GPUs
- Schedule: base - 100,000 steps or 12 hours
- Optimizer: Adam optimizer with $B_1 = 0.9, B_2 = 0.98$
- Regularization:
    1. Residual Dropout: applied to output of each sub-layer (0.1)
    2. Label Smoothing: lanel smoothing of epsilon = 0.1

## Results

![Results](https://d3i71xaburhd42.cloudfront.net/204e3073870fae3d05bcbc2f6a8e263d9b72e776/8-Table2-1.png)


<br>

# My Thoughts
- NLP 분야의 고전?ㅋㅋ 이라고 부를 수 있는 이 페이퍼... 논자시 보기전에 간단히 한 번 더 리뷰하면 좋을 거 같아서 정리를 해보았다. 1년전에 처음 읽은 거 같은데 다행히 이제 페이퍼를 더 잘 이해할 수 있는 거 같아서 나름 뿌듯하다. 그리고 처음에는 뭘 이렇게 헷갈리게 써놨지... 이랬는데 다시 읽으니까 엄청 명확하게 잘 쓴 논문같다 ㅋㅋ 관점의 차이가 이렇게나 중요한 것인가. 다행히 모델을 손으로 대충 아웃라인만 그리라고 하면 이제 어느정도 논문을 참고하지 않아도 그릴 수 있을 것 같다.
- 전에 사용되었던 recurrent model 또는 convolution model을 떠나서 parallelizaiton을 가능하게한 Transformer. 이 논문이 나왔기에 현재 BERT, GPT 등 다양한 NLP 모델이 나올 수 있었다. Transformer의 Encoder 부분을 뜯어서 사용한 모델들은 Auto-encoding Model으로 분류하고, 대표적으로 BERT, RoBERTa, XLM 등이 있다. 반면 Transformer의 Decoder 부분을 뜯어서 사용한 모델들은 Auto-regressive Model으로 분류하고, 대표적으로 GPT, Reformer 모델이 있다. 
- 저자들은 Computational Complexity를 고려해서 self-attention을 모델에 적용하겠다고 했지만 사실 이 메커니즘이 OOM 에러의 주원인이 아닐까 싶다. 모든 단어에 대한 attention을 계산하기 때문에 굉장히 computationally 부담스럽다. 지금은 어떤 모델을 pre-train하고 싶으면 꼭 구글의 TPU를 사용해야되는 상황이다. 그렇기 때문에 요즘 진행되는 연구는 이 부담을 줄이는 게 대부분 목표가 된다.

조금 간단히 리뷰 했는데 나중에 시간이 나면 내가 직접 Transformer에서 BERT, RoBERTA, XLM-R, Reformer의 발전에 대해서 블로그 포스트를 써보고 싶다!
---
layout: post
title: "Adversarial NLI: A New Benchmark for Natural Language Understanding"
categories: [Paper Review]
tags: [Datasets, NLI, NLU]
---

**Abstract:** We introduce a new large-scale NLI benchmark dataset, collected via an iterative, adversarial human-and-model-in-the-loop procedure. We show that training models on this new dataset leads to state-of-the-art performance on a variety of popular NLI benchmarks, while posing a more difficult challenge with its new test set.

[Original Paper](https://arxiv.org/abs/1910.14599)

[Website:](https://www.adversarialnli.com/)

--

## Paper Summary


**Problems with existing NLU datasets:**  
• NLU benchmarks struggle to keep up with model improvement  
• Are current NLU models genuinely as good as their high performance on benchmarks suggests? SOTA models learn to exploit spurious statistical patterns in datasets instead of learning meaning in the flexible and generalizable way that humans do 


**ANLI: Human-And-Model-in-the-Loop Enabled Training (HAMLET):**   

**Method:**
1. Write Examples:  Given a context (also often called a “premise” in NLI), and a desired target label, we ask the human writer to provide a hypothesis that fools the model into misclassifying the label.  
2. Get model feedback  
3. Verify examples and make splits  
4. Retrain model for next round: Once data collection for the current round is finished, we construct a new training set from the collected data, with accompanying development and test sets, which are constructed solely from verified correct examples.   

• Repeat for 3 rounds (add training data from previous round to train new model)  
• Dataset is collected to be more difficult than previous datasets  
• Longer contexts should naturally lead to harder examples, and so we constructed ANLI contexts from longer, multi- sentence source material.  

**Results:**  
• RoBERTa achieves state-of-the-art performance... We obtain state of the art performance on both SNLI and MNLI with the RoBERTa model fine-tuned on our new data.   
• Training on more rounds improves robustness  
• Continuously augmenting training data does not downgrade performance.  
• Exclusive test subset difference is small.  
• Performs well on hard NLI test sets: SNLI-Hard and  NLI stress test  

**Analysis:**  
• Can infer what type of data each model has weaknesses in  

**Further Work:**  
• Future work could explore a detailed cost and time trade-off between adversarial and static collection.  

--

## My Thoughts
- 데이셋 구축 속도가 모델 성능 향상 속도를 못 따라 온다는 것은 사실 (특히 한국어 같이 연구가 활발하게 진행되지 않는 언어는 더욱 그럼... 영어만 데이터셋이 매년 새롭게 나오는듯)
- 하지만 이 논문에서 저자들이 제시하는 방법론은 너무 expensive하다는 아주 큰 단점이 있음. 모든 데이터를 이렇게 구축하는 건 현실적으로 불가능
- adversarial data를 수집하는 과정도 기계학습이 가능할까? 근데 어느정도 "어렵지만 참"인 레이블을 만드는 것은 인간의 intuition이 필요한 부분이기 때문에 어려울 수도
- 이러한 데이터셋을 한국어로도 구축해보면 재밌을 거 같다...! 최근에는 데이터 수집 알바는 국내에서도 많이 진행되고 있는 것으로 알고 있으니 국내 대기업에서 시도해볼만 한듯

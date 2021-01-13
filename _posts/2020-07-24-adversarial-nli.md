---
layout: post
title: Adversarial NLI: A New Benchmark for Natural Language Understanding
featured-img: sleek
categories: [Paper Review]
tags: [Datasets, NLI, NLU]
---

**Abstract:** We introduce a new large-scale NLI benchmark dataset, collected via an iterative, adversarial human-and-model-in-the-loop procedure. We show that training models on this new dataset leads to state-of-the-art performance on a variety of popular NLI benchmarks, while posing a more difficult challenge with its new test set.

**Original Paper:** https://arxiv.org/abs/1910.14599

**Website:** https://www.adversarialnli.com/

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
- 

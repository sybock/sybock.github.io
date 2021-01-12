---
layout: post
title: "Cross-Lingual Ability of Multilingual BERT: An Empirical Study"
featured-img: hello
categories: [Paper Review]
---

**Abstract:** Recent work has exhibited the surprising cross-lingual abilities of multilingual BERT (M-BERT) -- surprising since it is trained without any cross-lingual objective and with no aligned data. In this work, we provide a comprehensive study of the contribution of different components in M-BERT to its cross-lingual ability. 

**Original Paper:** https://arxiv.org/abs/1912.07840

--

## Paper Summary

**Goal:** Analyze two-languages version of M-BERT (B-BERT for bilingual BERT) in three dimensions (1) linguistic properties and similarities (2) network architecture (3) input and learning objective


**What is M-BERT?**
• Pre-trained the same way as BERT but using Wikipedia text from top 104 languages
• M-BERT is contextual and training requires no supervision -- no alignment between languages
• M-BERT produces representation that seems to generalize well cross languages for a variety of downstream tasks


**(1) Linguistic Properties**

Method:  
• Fake English corpus created -- different language than English but having the exact same properties except word surface forms

Results:  
• Word-piece Overlap: no significance  
• Word-ordering similarity: performance drops, but still above baseline (other components of structural similarity contributing)   
• Word-frequency similarity: no significance  
• Structural Similarity (morphology, word-ordering, word frequency): some relation but definition of structural similarity too unclear  
 
**(2) Network Architecture**

Method:  
• Variation only in the parameter under examination, other parameters are kept constant

Results  
• Depth: more depth helps model extract good semantic and structural features on English and cross-lingually  
• Multi-head Attention: # of attention heads doesn't have a significant effect on cross-lingual ability  
• Total # of Parameters: not as significant as depth, but B-BERT requires certain minimum number of parameters to extract good semantic and structural features  

**(3) Input and Learning Objectives**

Results  
• Next Sentence Prediction: NSP objective hurts cross-lingual performance  
• Language Identity Marker: adding language identity marker doesn't affect cross-lingual performance  
• Characters vs. Word-piece vs. Word: Word-piece and Word tokenized input performs better than character tokenized input bc word-pieces and words carry much more information  


Further Work  
• Extending study to M-BERT  
• Better understand Structural Similarity as its significance is proven in cross-lingual ability  

--

## My Thoughts

- Wordpiece overlap이 transfer learning에 크게 영향이 미치지 않는다는 것을 알게된 것에 의의를 가진다. 이 논문 전에는 구조적으로 다른 언어에 대한 태깅이나 임베딩이 없는 mBERT의 선방?을 대부분 word-piece overlap이 anchor 역할을 해서 가능하다고 생각을 했다. 하지만 크게 상관이 없었다는 것은 꽤 흥미로운 발견이다.
- 하지만 word-piece overlap의 영향을 실험하기 위해서 fake english를 만들었는데, 과연 이것이 타당한 방법인지?는 또 고민해봐야 될 것 같다
- 그리고 결국 왜 mBERT가 좋은 성능을 보였는지에 대한 답안을 제시하지 않았다 
- 효과적인 transfer learning에 필요한 언어학적인 요인에 대해서는 너무 두루뭉술하게 말했기 때문에 조금 아쉬웠지만 앞으로 이 분야의 연구에 가이드라인이 될 수 있을 것 같다



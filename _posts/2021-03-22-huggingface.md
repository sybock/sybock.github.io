---
layout: post
title: "HuggingFace 정복하기"
categories: [Guide]
featured-img: hugging
tags: [NLP, HuggingFace]
mathjax: true
---

최근 NLP 분야 공부 또는 연구하는 사람이라면 당연히 HuggingFace를 사용해봤거나 들어봤을 것이다. BERT부터 GPT까지 웬만한 NLP 분야 논문에 나온 모델을 하나의 API로 밑으로 구현해놓은 것이다. 대단한 사람들이다... 오늘은 간단하게 이 HuggingFace 사용방법에 대해서 써보려고 한다.

# Introduction to HuggingFace

사실 내가 생각했을 때 HuggingFace에서 알아야될 Class는 네 가지다:
1. AutoConfig
2. AutoTokenizer
3. AutoModel (AutoModelForSequenceClassification 등등)
4. TrainingArgs and Trainer

이 네 가지 클래스를 이해하고 사용할 수 있다면 아주 쉽게 fine-tuning, pre-training 다 가능하다. 물론 좀 더 customize를 하고 싶다면 모델을 뜯어 고치거나 trainer을 뜯어서 고쳐야겠지만 이 네 개의 클라스를 통해서 파라미터 조정하다. 그리고 사실 파라미터 조정, 학습 방법 수정을 통해서 더 좋은 성능을 가질 수도 있겠지만, 논문에서 제시한 파라미터가 주로 가장 적합한 경우가 제일 많다. 어쨋든 위에 있는 클래스를 소개하도록 하겠다.

## AutoConfig

AutoConfig는 말 그대로 모델의 기본적인 설정을 불러오는 클래스다. 사실 pre-trained 모델을 그대로 가지고 오고 싶으면 따로 Config를 설정해줄 필요가 없다. 

Config가 필요한 경우: (1) 기본 Pre-Trained Model의 특정 파라미터 조정 원할 때 (2) Custom Model 설정을 그대로 가지고 오고 싶을 때

(1)과 같은 경우라면 kwargs로 `AutoConfig.pretrained()`에 모델 이름과 함께 조정하고 싶은 값을 패싱해주면 되고 (2) 같은 경우라면 전에 학습한 체크포인트에서 config 를 path로 불러오면 된다. 

아래 예시로는 XLM-R 모델을 불러왔다. 

```python
from transformers import AutoConfig

# Set Keyword Args 
config_args = {'hidden_dropout_prob':0.2, 
              'num_labels':2}

# Or.. load from path
config_path = './model/checkpoint.ckpt.index'

# Set Model Name
model_name = 'xlm-roberta-base'

# Get Configuration
config = AutoConfig.from_pretrained(model_name, **config_args)
print(config)
>> XLMRobertaConfig {
            "architectures": [
                "XLMRobertaForMaskedLM"
            ],
            "attention_probs_dropout_prob": 0.1,
            "bos_token_id": 0,
            "eos_token_id": 2,
            "gradient_checkpointing": false,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.2,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-05,
            "max_position_embeddings": 514,
            "model_type": "xlm-roberta",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "output_past": true,
            "pad_token_id": 1,
            "position_embedding_type": "absolute",
            "transformers_version": "4.4.2",
            "type_vocab_size": 1,
            "use_cache": true,
            "vocab_size": 250002
    }
```

## AutoTokenizer

Tokenizer은 자연어를 실수값으로 변환해줄 때 필요한 것이다. 모델마다 사용하는 Tokenizer의 유형, 세부 사항이 다르기 때문에 모델에 따라서 맞는 Tokenizer을 불러와야 한다. 

Tokenizer을 직접 트레이닝한 게 아니라면 사실 설정값을 딱히 바꿀 필요가 없다. `AutoConfig`와 똑같이 모델 이름을 넣어주면 자동으로 그 Tokenizer을 불러오기 때문에 사용하기 굉장히 쉽다.

아래와 같이 아까 설정한 config를 Tokenizer에도 패싱을 해줄 수 있는데, 사실 우리는 아까 Tokenizer에 관련된 kwargs를 아예 바꾸지 않았기 때문에 바뀌는 게 없다. Tokenizer을 한 번 프린트 해보면 어떤 config 값을 설정할 수 있는지 알 수 있다. 만약에 바꿀 게 있다면 아마 `model_max_len`이거 하나일 것 같다.

```python
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, config=config)

print(tokenizer)
>> PreTrainedTokenizerFast(name_or_path='xlm-roberta-base', vocab_size=250002, model_max_len=512, is_fast=True, padding_side='right', special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'sep_token': '</s>', 'pad_token': '<pad>', 'cls_token': '<s>', 'mask_token': AddedToken("<mask>", rstrip=False, lstrip=True, single_word=False, normalized=False)})

print(tokenizer.tokenize('그 영화는 재밌었다'))
>> ['▁그', '▁영화', '는', '▁재', '밌', '었다']

print(tokenizer('그 영화는 재밌었다'))
>> {'input_ids': [0, 2625, 29549, 769, 11105, 249902, 23548, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}

print(tokenizer(['그 영화는 재밌었다','나는 별로였다']))
>> {'input_ids': [[0, 2625, 29549, 769, 11105, 249902, 23548, 2], [0, 37231, 6, 109777, 43680, 2]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]}

```

## AutoModel

모델 Config와 Tokenizer을 불러왔으니까 이제 가장 중요한 모델을 불러올 차례이다. 

다른 클래스와 똑같이 AutoClass이기 때문에 사용하기 굉장히 쉽다. 모델 이름과 미리 설정해놓은 AutoConfig object만 있으면 나만의 모델 생성하기 끝!

```python
# Loading model from configuration
model_config = AutoModel.from_config(config)

# Loading model pre-trained model without chaning configurations
model_pretrained = AutoModel.from_pretrained(model_name)
```

이렇게 Config를 사용할 수도 있고, 그냥 모델 이름만 패싱을 해주면 기본적인 Config로 설정된 모델을 생성할 수 있다. 

모델을 프린트해보면 그 차이를 볼 수 있다:

```python
print(model_config)
>>> XLMRobertaModel(
  (embeddings): RobertaEmbeddings(
    (word_embeddings): Embedding(250002, 768, padding_idx=1)
    (position_embeddings): Embedding(514, 768, padding_idx=1)
    (token_type_embeddings): Embedding(1, 768)
    (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (dropout): Dropout(p=0.2, inplace=False)
  )...

print(model_pretrained)
>>> XLMRobertaModel(
  (embeddings): RobertaEmbeddings(
    (word_embeddings): Embedding(250002, 768, padding_idx=1)
    (position_embeddings): Embedding(514, 768, padding_idx=1)
    (token_type_embeddings): Embedding(1, 768)
    (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )...
```

우리는 아까 config에서 dropout 값을 0.2로 설정했기 때문에 첫번째 Config로 생성한 모델은 Dropout 값이 기본 모델 값과 다른 것을 볼 수 있다.

모델을 프린트하면 Layer이 어떻게 구성되어있고, input dimension, output dimension 다 자세히 볼 수 있기 때문에 모델을 공부할 때도 도움이 많이 될것이다. 여기에는 일부만 출력해놓았다.

여기서 `AutoModel` 클래스는 가장 기본적이고 generic한 클라스임으로 encoder 레이어 스택 뒤에 풀링 레이어 하나가 있다. 근데 만약에 우리는 특정 Task에 fine-tuning을 하고 싶다면 그 task에 맞는 모델을 불러와야된다. HuggingFace에서 역시 이렇게 다양한 AutoModel을 제공하고 있다. 

```python
from transformers import AutoModelForPreTraining, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoModelForMultipleChoice, AutoModelForNextSentencePrediction, AutoModelForTokenClassification, AutoModelForQuestionAnswering, AutoModelForTableQuestionAnswering
```

가장 많이 사용되는 모델은 `AutoModelForSequenceClassification`이 아닐까 싶다. 이 특정 모델을 불러오면 `AutoModel`과 달리 encoder layer 스택 위에 Classification Layer이 하나 더 있는 것을 볼 수 있다:

```python
config = AutoConfig.from_pretrained(model_name, **config_args)
model_seq = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

print(model_seq)
>>>   ...
(classifier): RobertaClassificationHead(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (dropout): Dropout(p=0.2, inplace=False)
    (out_proj): Linear(in_features=768, out_features=2, bias=True)
  ) ...
```

이렇게 각 task specific `AutoModel`은 task에 따라서 마지막 레이어가 바뀌어있을 것이다. 

## TrainingArgs and Trainer

모델, 토크나이저를 불러오는 방법을 다 알았으면 이제 모델을 학습할 차례이다. 허깅페이스에서 제공하는 `Trainer`를 사용하면 정말 정말 너무 쉽게 모델을 학습할 수 있다. 모델을 처음부터 pre-train하는 과정에는 trainer가 부적절할 수 있지만 대부분의 fine-tuning task에는 사용하기 적합할 것이다.

먼저 아까 모델 config를 설정해준 것 처럼 `Trainer`의 argument를 설정해줘야 된다. 여기서 학습에 필요한 파라미터를 조절할 수 있고 이것 저것을 바꿔보며 실험해볼 수 있다. 

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1, 
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

print(training_args)
>>> TrainingArguments(output_dir=./results, overwrite_output_dir=False, do_train=False, do_eval=None, do_predict=False, evaluation_strategy=IntervalStrategy.NO, prediction_loss_only=False, per_device_train_batch_size=32, per_device_eval_batch_size=64, gradient_accumulation_steps=1, eval_accumulation_steps=None, learning_rate=5e-05, weight_decay=0.01, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-08, max_grad_norm=1.0, num_train_epochs=1, max_steps=-1, lr_scheduler_type=SchedulerType.LINEAR, warmup_ratio=0.0, warmup_steps=500, logging_dir=./logs, logging_strategy=IntervalStrategy.STEPS, logging_first_step=False, logging_steps=500, save_strategy=IntervalStrategy.STEPS, save_steps=500, save_total_limit=None, no_cuda=False, seed=42, fp16=False, fp16_opt_level=O1, fp16_backend=auto, fp16_full_eval=False, local_rank=-1, tpu_num_cores=None, tpu_metrics_debug=False, debug=False, dataloader_drop_last=False, eval_steps=500, dataloader_num_workers=0, past_index=-1, run_name=./results, disable_tqdm=False, remove_unused_columns=True, label_names=None, load_best_model_at_end=False, metric_for_best_model=None, greater_is_better=None, ignore_data_skip=False, sharded_ddp=[], deepspeed=None, label_smoothing_factor=0.0, adafactor=False, group_by_length=False, report_to=['tensorboard'], ddp_find_unused_parameters=None, dataloader_pin_memory=True, skip_memory_metrics=False, _n_gpu=1)
```

여기서도 한 번 출력을 해보면 바꿀 수 있는 파라미터를 쭉 볼 수 있다. 뭐가 많지만 사실 자주 바꾸는 건 내가 위에 설정해놓은 것들. 

이렇게 설정을 다 했으면 `Trainer`를 생성하면 된다:

```python
from transformers import Trainer

trainer = Trainer(
    model=model_sequence,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=test_data
)
```

여기서 `compute_metrics`는 task를 evaluate하고 싶은 기준을 함수로 써주면 된다. 기본 classification task 같은 경우에는 accuracy, f1 score, precision, recall로 주로 평가가 된다. 그러므로 `compute_metrics` 함수는 모델에서 `pred` 벡터를 받아서 우리가 원하는 eval 방법으로 모델을 평가하는 것. 

허깅페이스에서 기본으로 제공하는 `compute_metrics` 함수는 아래와 같다 (기본 classification task 사용에는 문제 없음):

```python
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
```

그럼 이제 `trainer`이라는 python object가 생겼는데, 학습하고 모델 평가는 어떻게 하는가?

```python
# 모델 학습
trainer.train()

# 모델 평가
trainer.evaluate()
```

`.train()` 메소드는 아까 `trainer`에 넣어준 `train_dataset`를 사용해서 모델 학습을 진행하고 `.evaluate()` 메소드는 같은 `trainer`의 `eval_dataset`를 사용해서 모델을 평가한다. 너무 간단해서 설명하기도 민망한 정도...

## Other

`Trainer`를 사용해서 학습하면 굉장히 간단하다는 장점이 있지만, 모델 학습에 대한 분석이 어렵다는 단점도 있다. 만약 모델에 대한 분석을 더 섬세하기 하고 싶다면 설정할 수 있는 몇 가지 방법이 있다.

### 1. Attention

Transformer모델의 핵심인 attention. 만약 모델의 attention이 각 레이어에 어디를 보고 있는지 알고 싶다면 모델을 불러올 때 `output_attentions=True`로 설정해주면 된다:
```python
model = AutoModelForSequenceClassification.from_pretrained(model_path_or_name, output_attentions=True)
```

그럼 ouput에 attention도 함께 출력된다.
```python
# 문장
sentence = "이 영화는 아주 재미있다."

# 문장을 vector로 변환
inputs = tokenizer.encode_plus(sentence, return_tensors='pt', add_special_tokens=True)
input_ids = inputs['input_ids'].to(model.device)

# vector을 모델에 넣어주고, attention만 가져오기
attention = model(input_ids)[-1]
```

attention을 출력했으면 다른 모듈을 사용해서 attention을 시각적으로 볼 수도 있다:

```python
from bertviz import head_view

tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
head_view(attention, tokens)
```

이런 그래픽이 나온다:
![attn](https://i.ibb.co/9nKGT54/Screen-Shot-2021-03-23-at-11-27-13-AM.png)

### 2. Predict

그 외에도 모델이 어떤 sequence를 잘못 예측했는지 분석하고 싶다면 `.predict()` 함수를 사용해서 모델의 preds를 직접 저장할 수 있다. 

```python
#Get Predictions as Array
preds = trainer.predict(test_data)

#Argmax Softmax Values
predictions = np.argmax(preds[0], axis=1)
labels = preds[1]

print(predictions)
>>>[1 1 0 ... 1 0 0]
print(labels)
>>>[1 0 0 ... 0 0 0]
```

이렇게 label, prediction이랑 데이터가 있으면 False Positive, False Negative 예시를 뽑아서 분석해볼 수 있다. 

# Fine-Tuning XLM-R on NSMC

HuggingFace를 사용해서 아주 쉽게 NSMC task를 fine-tuning 해보자!

```python
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset
import torch


# Model, Tokenizer
model_path_or_name = 'xlm-roberta-base'
config = AutoConfig.from_pretrained(model_path_or_name, num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained(model_path_or_name)
tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, config=config)

# Dataset
dataset = load_dataset('nsmc')
class nsmc_data(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        super().__init__()
        self.label = torch.tensor(dataset['label']).long()
        features = tokenizer([str(x) for x in dataset['document']], padding=True, truncation=True)
        self.input = torch.tensor(features['input_ids'])
        self.mask = torch.tensor(features['attention_mask'])
    
    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return {'input_ids': self.input[index], 'attention_mask': self.mask[index], 'label':self.label[index]}
train_data = nsmc_data(dataset['train'], tokenizer)
test_data = nsmc_data(dataset['test'], tokenizer)

# Evaluate Function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1, 
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=test_data
)

trainer.train()
>>> TrainOutput(global_step=4688, training_loss=0.34629279680219527, metrics={'train_runtime': 1437.0598, 'train_samples_per_second': 3.262, 'total_flos': 3.55341747708e+16, 'epoch': 1.0, 'init_mem_cpu_alloc_delta': 312794, 'init_mem_gpu_alloc_delta': 1112255488, 'init_mem_cpu_peaked_delta': 18306, 'init_mem_gpu_peaked_delta': 0, 'train_mem_cpu_alloc_delta': 832234, 'train_mem_gpu_alloc_delta': 3371369984, 'train_mem_cpu_peaked_delta': 10867081, 'train_mem_gpu_peaked_delta': 4122065920})

trainer.evaluate()
>>> {'eval_loss': 0.27084341645240784,
 'eval_accuracy': 0.88918,
 'eval_f1': 0.8901924258338123,
 'eval_precision': 0.8881683011705157,
 'eval_recall': 0.8922257974814285,
 'eval_runtime': 115.0763,
 'eval_samples_per_second': 434.494,
 'epoch': 1.0,
 'eval_mem_cpu_alloc_delta': 915530,
 'eval_mem_gpu_alloc_delta': 0,
 'eval_mem_cpu_peaked_delta': 2712591,
 'eval_mem_gpu_peaked_delta': 278054400}
```

GPU 하나로 epoch 1 돌렸을 때 xlm-r 모델은 f1 스코어 89% 정도 나온다. 학습양에 비해 굉장히 양호한 점수. 실제 학습 시간은 한 20분 밖에 소요되지 않는다. 

# 마지막으로...

사실 소개한 클래스 4개 외에도 `dataset` 클래스를 사용하면 좋은데 이건 나도 잘 몰라서 좀 더 연구해볼 예정이다. 이상하게 이 `dataset` 클래스를 사용하면 OOM 에러가 더 많이 나는 것 같다. 메모리 매니지먼트가 아직 잘 안되는 거 같기도 하고...

어쨋든 오늘은 허깅페이스를 쉽게 사용하는 방법에 대해서 알아봤고 한국어 task 중에서 제일 만만한 nsmc를 허깅페이스 이용해서 학습을 해보았다.

다음에는 허깅페이스 API를 사용해서 어떻게 더 customize할 수 있는지에 대해서 알아봐서 블로그 포스팅을 해야겠다.
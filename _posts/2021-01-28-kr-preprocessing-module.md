---
layout: post
title: "Korean NLP Preprocessing Moduel | 한국어 전처리 모듈 만들기"
categories: [Guides]
featured-img: calligraphy
tags: [NLP, Korean NLP]
---

전처리는 매번 느끼지만 정말 손이 많이 가는 작업이다. 하지만 굉장히 중요한 작업이다. 학습 데이터의 질에 따라 모델 성능도 천차만별이기 때문이다. 전처리가 어려운 이유 중 하나는 코퍼스마다 특성이 다르기 때문에 먼저 코퍼스에 전처리가 필요한 부분을 파악하고 전처리를 하는 것이 중요하다. 

전에는 여기 저기 만들어 놓은 regex 패턴이랑 함수랑 덕지덕지 붙여서 썼는데 그렇게 하게 되면 코드가 어떻게 쓴건지 생각도 잘 안나고 너무 지저분하다. 조금 더 체계적으로? 그리고 더 general하게 많은 코퍼스를 하나의 모듈을 이용해 전처리할 수 있도록 전처리 모듈을 개발해봤다. 

지난 일주일 동안 아주 천천히 조금씩 개발한 결과 오늘 완성해서 [깃헙](https://github.com/sybock/KR-Preprocess)에 올렸다. 그래서 별거 없지만 오늘은 모듈 개발 과정에 대해서 써보기로 했다. 

시작하기 전에 이미 누가 개발한 전처리 모듈이 있으면 거기에 덧붙여서 만드려고 찾아본 결과 [kor-text-preprocess](https://github.com/YongWookHa/kor-text-preprocess) 라는 모듈이 있었다. 여기서 더 얹혀서 할까 고민 했는데 내가 필요한 것 보다 더 다양한 용도가 포함되어 있고 결국 다 뜯어 고쳐야될 거 같아서 참고만 했다. 많이 참고 했다... 감사합니다.

# Overview

모듈은 이렇게 구성이 되어있다:
```
kr_preprocess/
        __init__.py
        document.py
        preprocessing.py
        utils.py
```

- `utils.py`에는 저번 블로그 포스트에서 설명한 `extract_text` 함수가 들어가 있다. 이 함수를 사용하고 싶으면  `from kr_preprocess import extract_text`하고 사용하면 된다. 아직까지는 main이나 argsparse를 사용할 수 있도록 연결하지 않았다. 
- `document.py`에는 기본적으로 하나의 문서를 읽고, 문장으로 나누고, 나눈 문장을 새로운 파일에 쓰는 기능을 가지고 있다. 그래서 문서를 그냥 문장별로 나누고 싶으면 여기에 있는 class만 사용하면 된다. 
- `preprocessing.py`에는 이 모듈의 꽃?인 전처리 함수들이 있다. 하나하나 부를 수 있지만 다 필요한 경우가 더 흔하기 때문에 `.apply()`를 사용하면 쉽게 전처리를 끝낼 수 있다.

## Preprocessing Class
전처리 클래스를 더 자세히 들여다 보자.
- `clean()`: 기본적인 전처리를 하는 함수. 영어/한글 및 필요한 기호 말고 모두 지워준다. 또 `\s`, `,`, `.` 연속으로 있는 경우를 지워준다. 이건 앞서 말한 kor-text-preprocess를 참고
- `remove_en_sent()`: 영어만 있는 문장을 지운다
- `remove_list()`: 전처리를 하다 보니까 목차나 긴 리스트 같은 문장이 꽤 있었다. 만약 task가 이런 데이터를 본다면 놔둬야겠지만 필요 없는 경우가 대부분일 거 같아서 이런 건 그냥 지워버렸다.
- `remove_parantheses()`: 괄호와 괄호안에 있는 모든 내용을 지워줌
- `remove_links()`: 링크나 Html 텍스트가 들어간 문장은 다 지워버림. 하나하나 찾아서 지우기엔 너무 어려워서 몽땅 삭제.
- `remove_unmatched_items()`: [ 나 ( 가 전처리 과정에서 하나 씩 남아 있는 경우가 있는 이걸 다 지워줌
- `remove_repeat()`: 한국어 코퍼스에서는 'ㅋㅋㅋ' 이런게 많은데 이걸 다 'ㅋㅋ'로 줄여줌

**Optional Functions**

특정 코퍼스에만 있는 Noise를 옵션으로 받아서 적용되는 함수. 뉴스나 웹크롤링 코퍼스가 제일 노이즈가 많다. 이런 코퍼스에 적용하기 좋은 함수들이지만 소설이나 잡지 코퍼스 등 다른 코퍼스에는 필요 없는 전처리 과정.

- `remove_datetime()`: 2021-01-28 이렇게 가끔 웹 크롤링 데이터에 남아 있는 경우가 있다. 이런 걸 제거해줌.
- `remove_news_brackets()`: 인터넷이나 뉴스 크롤링 한국어 코퍼스는 특정 뉴스 noise가 있다... 000기자= 이런거 라던지 사진제공= 뉴스= 이런거... 이런 거를 지워주는 함수.

필요한 것만 골라서 일일이 함수 불러서 사용할 수 있지만 `.apply()`로 하면 min_length에 따라 문장도 잘라주고 더 편리하게 사용할 수 있다!

# Usage

난 사실 CLI 코드를 많이 사용해보지도 않아서 좀 낯설지만... 코딩을 할수록 linux command의 편리함을 알게 되는 거 같아서 이번 모듈도 CLI로 작성해보았다. 그냥 조금 낯설 뿐이었지 파이썬에 있는 `argsparse` 모듈을 사용하니까 어려운 건 하나도 없었다 ㅎㅎ

그래서 이렇게 사용하면 된다:

```
# Get help with options
python3 main.py -h

# Example usage
python3 main.py -i /data/dataset.txt -o /data_cleaned/dataset.txt -m 10 --news True -dt True
```

-h 를 해보면 이렇게 나온다
```
usage: main.py [-h] -i INPUT [-o OUTPUT] [-m MIN_LENGTH] [--news NEWS]
               [-dt DATETIME]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input file path
  -o OUTPUT, --output OUTPUT
                        Output file path
  -m MIN_LENGTH, --min_length MIN_LENGTH
                        Minimum length of line
  --news NEWS           Text has news articles
  -dt DATETIME, --datetime DATETIME
                        Text has date time text to be erased
```

옵션으로 조정할 수 있는 게 많이 없는데... min_length와 뉴스 및 시간/날짜 전처리를 적용하는 게 옵션이다. 앞으로 더 많은 코퍼스를 전처리하면서 옵션으로 필요한 부분이 있으면 추가할 예정.

특정 코퍼스에 추가로 필요한 전처리 과정이 있다면 `main.py`에 추가하면된다. 

코드를 실제로 돌려보면 logging도 설정을 해놨기 때문에 어떤 파일을 읽고 있는지, 전처리 전/후에 문서 line 수, 전처리가 완료된 파일은 어디에 저장이 됐는지 나온다. 

# To-do

근데 사실 지금 모듈은 굉장히 불친절하다. docstring이 하나도 없다. ㅠㅠ  
곧 추가할 예정이고, 또 extract_text를 옵션으로 넣어서 한 줄에 사용할 수 있도록 코드를 수정해보려고 한다.  
또 고민해봐야될 부분은 코드 메모리/속도다. 전처리 하는 파일이 크면 30~40분씩 걸린다. 근데 이게 줄을 다 읽고 프로세싱을 해야되는 작업이어서 그렇게 밖에 안되는 거 같기도 한데, 조금 더 검색해보고 고민해보고 코드를 더 효율적으로 짤 수 있었음 좋겠다.

--

만들기 조금 귀찮았는데 막상 만들어놓으니까 여기저기 잘 사용하고 있다. 웬만한 한국어 코퍼스에는 많이 뭘 추가 안해도 굉장히 깨끗한 상태로 만들어 놓는다.

골라내기 어려운 부분은 그냥 문장을 지우는 걸로 만들었다. Quality > Quantity!!
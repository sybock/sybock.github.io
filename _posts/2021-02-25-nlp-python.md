---
layout: post
title: "자연어처리를 위한 Python 알쓸신잡"
categories: Guides
featured-img: soup
tags: [Python]
---

오늘은 자연어처리 작업을 할 때 알면 좋은 Python 기본 함수를 정리해보려고 한다. 실제로 내가 과제나 실험을 진행하면서 많이 사용하는 함수들 위주로 정리해보려고 한다. 그래서 여기서 "알쓸신잡"은 "알면 쓸모 있는 신기한 잡학사전"을 뜻한다... ㅎ;;

## 1. 영어/한글 구별

영어 알파벳을 구별할 때에는 `.isalpha()` 함수를 아래와 같이 사용하면 된다.

```Python
>>> 'Hello'.isalpha()
True
```

하지만 한국어에 같은 방법을 사용해도 True가 나온다.

```Python
>>> '안녕'.isalpha()
True
```

그럼 어떻게 해야 영어와 한글을 구별할까? 간단하게 `.encode()` 함수를 먼저 사용하고 `.isalpha()` 함수를 적용해주면 된다.

```Python
>>> '안녕'.encode()
b'\xec\x95\x88\xeb\x85\x95'

>>> '안녕'.encode().isalpha()
False
```

한국어 코퍼스에서 영어로 구성된 문장을 지우거나 구별하고 싶을 때 사용하면 되는 방법.

## 2. Splitting Corpora

자연어처리를 조금이라도 해본 사람이면 `.split()`을 분명히 사용해봤을 것이다. 가장 기본적으로는 whitespace를 기준으로 `string`을 split해주고, `()`안에 split을 기준으로 하고 싶은 기호를 넣으면, 그 기호대로 문장이 split된다.
```Python
>>> 'Hello. My name is Suyeon.'.split()
['Hello.', 'My', 'name', 'is', 'Suyeon.']

>>> 'Hello. My name is Suyeon.'.split('.')
['Hello', ' My name is Suyeon', '']
```
그런데 아래 `.`점 기준으로 split한 것을 보면 `.`은 결과 리스트에서 빠지게된다. 문장을 나누고 싶다면 `.`을 계속 보유하는 게 좋은 경우가 많다. 만약 split하고 싶은 기준 기호가 `newline`이라면 `.splitlines(True)` 함수를 사용하면된다.

```Python
>>> 'Hello.\n My name is Suyeon.\n'.splitlines(True)
['Hello.\n', ' My name is Suyeon.\n']
```

근데 다른 기호를 기준으로 split하고 싶다면...? 약간의 수작업을 할 수 밖에 없다.
```Python
>>> [n+'.' for n in 'Hello. My name is Suyeon.'.split('.') if n!='']
['Hello.', ' My name is Suyeon.']
```

## 3. .replace() 및 .sub()

전처리 과정에서는 불필요한 기호나 문장을 삭제 또는 대체해야되는 작업이 많다. 파이썬 기본 함수인 `.replace()`를 사용하거나 Regex의 `.sub()` 함수를 사용하면 된다. 용도가 살짝 다르다. `.replace()`같은 경우에는 기호 및 단어의 exact match 를 대체할 때 사용하면 되고 `.sub()`은 Regex 표현을 사용해서 여러 기호 및 단어를 묶어서 한번에 대체할 수 있다. 그러나 `.replace()`를 사용할 수 있을 때에는 이 함수를 사용하는 것이 더 좋다. 더 빠르고, Regex 오류로 대체하고 싶은 문장을 막상 놓칠수 있는 경우가 있기 때문이다. 

예를 들어서 얘기 하자면 문장이 whitespace로 구분되어있는 코퍼스에서 문장을 `newline`으로 구분짓고 싶다고 하자. ? 및 ! 은 그냥 `.replace()`로 바꿔주면 되지만 `.`같은 경우에는 사람 이름으로 Initial이 있을수도 있고 이런저런 제한 때문에 regex 표현을 정의하고 바꾸는 게 좋다. 아래와 같이 나는 코드를 짜서 사용했었다. 
```Python
# replace의 예시
>>> doc = 'David M. James went that way? But I saw him run in the other direction. '
>>> doc.replace('? ', '?\n')
'David M. James went that way?\nBut I saw him run in the other direction. '

# sub의 예시
>>> PERIOD = re.compile(r'(?<!([A-Z]))\. +')
>>> PERIOD.sub('.\n', doc)
'David M. James went that way? But I saw him run in the other direction.\n'

```

---

막상 써보니까... 딱히 쓸게 많이 없다. 앞으로 또 좋은 팁이 생각나면 추가하도록 하겠다. 

최근에 다른 공부를 좀 한다고 논문도 많이 못 읽고 블로그도 소홀히 했는데 앞으로 일주일에 최소 한 번, 그래도 두 번 정도 포스팅하는 것을 목표로 해야겠다... 
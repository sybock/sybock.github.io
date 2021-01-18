---
layout: post
title: "한국어 코퍼스 리스트 및 전처리 준비"
categories: [Guides]
featured-img: kr-corpus
tags: [NLP, Corpus, Korean NLP]
---

# 한국어 코퍼스

Pre-train 모델의 성능은 데이터의 양질에 완전히 의존한다. 한국어는 영어만큼 많은 코퍼스가 존재하지 않지만, 점점 늘어나는 추세이다. 아래는 내가 찾고 직접 다운 받아서 살펴본 코퍼스 리스트:


- [KAIST Corpus](http://semanticweb.kaist.ac.kr/home/index.php/KAIST_Corpus): Raw 코퍼스를 각 분야 별로 다운 받을 수 있고, Eng-Chinese-Japanese Parallel Corpus도 다운 받을 수 있음. DOWN을 누르고 메일 주소와 소속만 기재하면 바로 다운 받을 수 있는 링크가 전송된다. 
- [국립국어원 모두의 말뭉치](https://corpus.korean.go.kr/): 국립국어원에서 2020년도에 공개한 따끈따끈한 코퍼스. 소속과 목적을 간단하게 써서 신청하면 코퍼스를 다운 받을 수 있다. 코퍼스 목록은 구문 분석 말뭉치, 문법성 말뭉치, 문서 요약 말뭉치 등 주로 Task로 나눠져있다. 근데 이 task를 사용해서 모델 평가를 하는 건 아직 못 봤고 주로 학습에 사용하는 것 같다. 깨끗한 데이터이기 때문에 전처리가 비교적 쉬운 것이 가장 큰 장점인 거 같다. (최근 "이루다" 챗봇 논란으로 메신저 말뭉치는 다운 받을 수 없다...)
- [나무위키](https://github.com/lovit/namuwikitext): 나무위키의 덤프 데이터를 바탕을 제작한 wikitext 형식의 텍스트 파일입니다. 학습 및 평가를 위하여 위키페이지 별로 train (99%), dev (0.5%), test (0.5%) 로 나뉘어져있습니다.
- [세종 말뭉치](https://github.com/lovit/sejong_corpus): 쉽게 세종 말뭉치를 다운 받을 수 있는 깃헙으로 링크를 해두었다. 데이터 양도 많지 않고 깨끗하지 않기 때문에 사용할까 말까 고민되는 말뭉치...
- [HANTEC 2.0](http://kristalinfo.dynu.net/download/): 1998년부터 2003년까지 한국과학기술정보연구원과 충남대학교가 공동으로 개발한 정보검색시스템 평가를 위한 한글 테스트 컬렉션. 
- [OSCAR](https://oscar-corpus.com/): CommonCrawl 코퍼스를 언어별로 필터링해서 정리해 놓은 사이트. 한국어 데이터가 약 12GB 되는데 깨끗한 데이터는 아님

위에 나온 한국어 코퍼스 외에도 다양한 말뭉치가 존재한다.  


## Korpora
최근에 한국어 코퍼스 데이터를 모아서 쉽게 다운 받을 수 있도록 모듈 [Korpora: Korean Corpora Archives](https://github.com/ko-nlp/Korpora)이 생겼다. 일부 데이터는 이 모듈을 사용해서 다운로드 받았다. 사용하기도 편리하고 파이썬에서 바로 nltk 모듈과 같이 데이터셋을 불러올 수 있어서 편리하게 사용할 수 있다. 나는 terminal에서 다운 받아서 잘 되는지는 모르겠지만... 어쨋든 한국어 코퍼스 쉽게 다운 받고 싶다면 꼭 한 번 참고할만한 패키지! 앞으로 더 많은 데이터셋이 추가되었으면 좋겠다. 



<br>

# 전-전처리

본격적인 데이터 전처리를 하기 전에 전처리 준비를 해야된다.

무슨 말이냐면, 데이터가 아주 깨끗하게 한줄에 한 문장 씩 딱 나와있으면 좋겠지만 그런 경우는 드물고 encoding을 변경해야되는 경우도 있다. 그래서 전-전처리가 필요하다. 데이터를 전처리할 수 있도록 데이터를 가공해주는 것이다. 크게 보면 아래 세 가지 작업으로 나눌 수 있다:
1. 필요한 text만 파일에서 추출 
2. 한 파일에 모든 문장 저장 
3. 인코딩 변환 (필요하면) 

<br>

예를 보면서 설명해보자. 

`kaist-raw-corpus`는 압축을 풀면 아래와 같이 폴더별로 데이터가 정리되어 있다. 

```
academic/  generality/  law/  literature/  mass-media/  religion/  textbook/  utility/
```

 `literature` 폴더에 들어가보면 또 폴더별로 정리가 되어있다:

```
autobiography/  biography/  criticism/  diary/  essay/  juvenileAndfable/  novel/  poem/  theatre/
```

 `biography` 폴더를 보면, 그제서야 `.txt` 파일이 나온다: 

 ```
 kaistcorpus_written_raw_or_literature_biography_mh2-0048.txt  kaistcorpus_written_raw_or_literature_biography_mh2-0749.txt
kaistcorpus_written_raw_or_literature_biography_mh2-0264.txt  kaistcorpus_written_raw_or_literature_biography_mh2-0916.txt
kaistcorpus_written_raw_or_literature_biography_mh2-0316.txt  kaistcorpus_written_raw_or_literature_biography_mh2-0917.txt
kaistcorpus_written_raw_or_literature_biography_mh2-0320.txt  kaistcorpus_written_raw_or_literature_biography_mh2-1005.txt
kaistcorpus_written_raw_or_literature_biography_mh2-0340.txt  kaistcorpus_written_raw_or_literature_biography_mh2-1011.txt
kaistcorpus_written_raw_or_literature_biography_mh2-0392.txt  kaistcorpus_written_raw_or_literature_biography_mh2-1053.txt
kaistcorpus_written_raw_or_literature_biography_mh2-0479.txt  kaistcorpus_written_raw_or_literature_biography_mh2-1155.txt
kaistcorpus_written_raw_or_literature_biography_mh2-0488.txt  kaistcorpus_written_raw_or_literature_biography_mh2-1480.txt
kaistcorpus_written_raw_or_literature_biography_mh2-0511.txt  kaistcorpus_written_raw_tr_literature_biography_mh2-0122.txt
kaistcorpus_written_raw_or_literature_biography_mh2-0524.txt  kaistcorpus_written_raw_tr_literature_biography_mh2-0217.txt
kaistcorpus_written_raw_or_literature_biography_mh2-0736.txt  kaistcorpus_written_raw_tr_literature_biography_mh2-1489.txt
```

이 중에 아무거나 한 번 열어보면: 

```
<language> Korean </language>
<textcode> KSC-5601-1988 </textcode>
<process> raw </process>
<version> 2001(tr)</version>
<filename> kaistcorpus_written_raw_tr_literature_biography_mh2-1489.txt </filename>
<title> ¼Õ¹® ÆòÀü  </title>
<author> ½ÃÇÁ¸° Àú ; ¹ÎµÎ±â ¿ª</author>
<date> 1990 </date>
<publisher> Áö½Ä»ê¾÷»ç </publisher>
<kdc> 819</kdc>
<tdmsfiletext>
Á¦ 1Àå Á÷¾÷Àû Çõ¸í°¡ÀÇ Åº»ý
  ¼Õ¹®(áÝÙþ)Àº   Áß±¹    µ¿³²ºÎ   ÇØ¾ÈÀÇ    ±¤µ¿¼º(ÎÆÔÔàý)ÁÖ°­(ñÁË°)»ï°¢ÁÖ¿¡   À§Ä¡ÇÑ Çâ»êÇö(úÅß£úã)ÃëÇüÃÌ(ö¨úûõ½)¿¡¼­ 1866³â 11¿ù 12ÀÏ¿¡ ÅÂ¾î³µ´Ù. ÀÌ°÷Àº ±× »ý¾Ö¿Í ¾ß¸ÁÀÌ ¼­±¸¿Í  ³Ê¹«³ª  ¹ÐÁ¢ÇÏ°Ô   ¿¬°üµÇ°Ô  µÉ  ¿î¸íÀÌ¾ú´ø  »ç¶÷¿¡°Ô´Â   ÀûÀýÇÑ  Ãâ»ý±â¿´´Ù. ¼ºµµ(àýÔ´)±¤ÁÖ(ÎÃñ¶)·ÎºÎÅÍ   ÆÛÁ®³ª°¡´Â   ÀÎ±¸°¡    Á¶¹ÐÇÑ   »ï°¢ÁÖ´Â   Áß±¹°ú   ¼­±¸ ÇØ¾çÁ¦±¹(ð³ÏÐ)°£ÀÇ °¡Àå ¿À·¡µÈ  ¿¬°èÁ¡ÀÌ¾ú´Ù. 1517³â Æ÷¸£Åõ°¥ÀÎÀÌ µµÂøÇÑ  ÀÌÈÄ ±¤ÁÖ´Â ¼­±¸¿ÍÀÇ ±³¿ªÀ» À§ÇÑ  °¡Àå Áß¿äÇÑ Ç×±¸°¡ µÇ¾úÀ¸¸ç, 18¼¼±â  Áß¿±ºÎÅÍ ¾ÆÆíÀüÀï±îÁö °ÅÀÇ 1¹é ³â µ¿¾È ¿©ÀüÈ÷ Áß±¹ ÇØ¾ÈÀÇ À¯ÀÏÇÑ ÇÕ¹ýÀûÀÎ ÃâÀÔÁöÁ¡À¸·Î ³²¾Æ ÀÖ¾ú´Ù.
  ...
  ```

이렇게 나타난다.

이럴 땐 파이썬으로 끄적끄적 뭘 하려고 고생하는 것 보다 linux command를 사용하면 편하다.  
아래 command는 폴더에 있는 모든 파일을 `utf-8`로 인코딩을 해주고 concat해서 지정한 경로 (`path/directory`)에 저장을 해준다.

`iconv -c -f euc-kr -t utf-8 * > /path/directory/filename.txt
`

벌써 (2), (3) 끝!

--

그럼 파일의 형태는 이렇게 바뀌어있을 것이다:

```
<language> Korean </language>
<textcode> KSC-5601-1988 </textcode>
<process> raw </process>
<version> 2001(or)</version>
<filename> kaistcorpus_written_raw_or_literature_biography_mh2-0048.txt </filename>
<title> 부하린 : 혁명과 반혁명의 사이  </title>
<author> 김남국 </author>
<date> 1993 </date>
<publisher> 문학과지성사 </publisher>
<kdc> 340.99  </kdc>
<tdmsfiletext>
제 1 장
서론 : 부하린과 그의 시대

 19세기와 20세기에 걸친 러시아는 심각한 자기 모순과 고민 속에서 혁명을 행해 소용돌이치던 격정의 대륙이었다. 이 시기의 러시아는 또한 전
제 정치와 무정부주의, 민족주의와 보편주의, 개인주의와 집단주의, 잔학성과 인간성, 예속과 저항 등 극명하게 대조를 이루는 두 얼궁의 민족>적 특징이 시대마다 그 모습을 달리하며 나타나는 가운데 점증하는 급진적 혁명주의가 볼셰비즘을 중심으로 한 조직적인 결사에로 수렴되어가는
 시기이기도 했다.
```

보면 알 수 있겠지만 이 말뭉치에 있는 모든 파일은 `html`처럼 정리가 되어있고 사실 필요한 부분은 `<tdmsfiletext>`에 있는 문장들이다. 그 외 내용은 모델 pre-training에 별 도움이 안될 것이다.

이제는 파이썬 코드를 사용해서 `<tdmsfiletext>` 아래에 있는 문장들만 추출하자.

```python
import os
import re

# Define path to dataset
path = '/data/kaist_rawcorpus/'

# Get list of directories in dataset directory
for folder in os.listdir(path):

    # Get list of documents in each dataset directory
    # Assumes ONLY text files are in directory
    for docu in os.listdir(path+folder):

        # Read raw corpus doc
        f = open(path+folder+'/'+docu) 
        lines = f.readlines()

        # Write new corpus doc
        o = open('/data/kaist_rawcorpus/'+docu, 'w')
        for l in range(0, len(lines)):
            # Write lines from <tdmsfiletext> until </tdmsfiletext> found
            if re.search('<tdmsfiletext>', lines[l]):
                for k in range(l+1, len(lines)):
                    if lines[k].strip() == '</tdmsfiletext>':
                        break
                    else:
                        rawLine = re.sub('<[^<>]*>', '', lines[k])
                        o.write(rawLine+'\n')
        # Print progress
        print(f'file {docu} Done!')
```

사실 파이썬으로 인코딩도 바꿔줄 수 있고 한 파일로 모든 줄을 출력할 수 있을텐데, 인코딩 변환은 파이썬에서 더 까다롭고 잘 안되는 경우가 있다... (내 능력의 한계인가?) 리눅스로 하는 것이 더 마음이 편하고 오래 걸리지도 않는다. 파이썬을 통해서 모든 문서를 한 파일로 출력하는 건 `open(document, 'w')`를 그냥 바깥 loop 에 빼주면 된다. 필요에 따라 변형해서 사용하면 된다.

이렇게 간단히(?) 전-전처리가 끝나고 이제 정말 어려운 전처리를 해야될 차례다.

그건 다음 포스팅에...
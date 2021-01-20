---
layout: post
title: "Korean-English Parallel Corpora | OPUS 말뭉치 다운로드 및 사용방법"
categories: [Guides]
featured-img: flags
tags: [NLP, Corpus, Korean NLP]
---

# Parallel Corpus

To train multilingual NLP models for cross-lingual tasks, you need parallel corpora. Usually a parallel corpus consists of the same sentence translated in source and target languages, separated by a tab. It can come in various forms, one of the most commonly used parallel corpora being the Bible, a book that's obviously been translated into hundreds of languages and is not licensed. 

Parallel corpora also come in handy when attempting to train a model via transfer learning. Simply put, transfer learning is fine-tuning a single or multi-lingual model to a specific language, in the hopes that the fine-tuend model retains some of the information learned during general pre-training. 

<br>

# OPUS

A representative parallel corpora dataset is [OPUS](http://opus.nlpl.eu/), the Open Parallel corpUS. It's a project that attemps to align various open sourced resources on the web. The amount of data for each language varies but there is a good amount for most.


## Downloading the OPUS Dataset

Many of the datasets are very large in OPUS and there is the downfall that you have to download the corpus of both your desired source and target languages, after which an alignment file aligns them as your output.

Some days ago I downloaded the aligned en-ko files in the desired  `.tsv` form for wiki text and JW300 text without having to process the data again.

But I forgot what command/program I used to download them... and when I went back to download the OpenSubtitle and Tatoeba data today I had to do some of the processing myself. 

It wasn't that difficult just sort of... annoying. Just so I don't forget, I'll write about it here.

To download the corpus, I used [OpusTools](https://github.com/Helsinki-NLP/OpusTools/blob/master/opustools_pkg/README.md).

1. Install OpusTools
    ```
    pip install opustools
    ```
2. Use the `opus_read` script from Terminal to download raw files and align them. Refer to their official site for information on the language codes and find the directories [here](http://opus.nlpl.eu/opusapi/?corpora=True). English is `en` and Korean is `ko`
    ```
    opus_read -d DIRECTORY -s SOURCE_LANG -t TARGET_LANG -write FILENAME.txt
    ```

That's it for downloading the corpus. 

You'll get two raw zip files like this:  `DIRECTORY_latest_raw_LANGUAGE.zip`  
and an aligned `.txt` file with whatever filename you entered.


## Processing the OPUS Dataset

The aligned `.txt` files look like this:

**Tatoeba**
```
================================
(src)="1276">Let's try something.
(trg)="5350">뭔가 해보자!
================================
(src)="1277">I have to go to sleep.
(trg)="5351">자야 합니다.
================================
(src)="1280">Today is June 18th and it is Muiriel's birthday!
(trg)="5354">오늘은 6월 18일, Muiriel의 생일입니다!
================================
(src)="1282">Muiriel is 20 now.
(trg)="5356">Muiriel은 지금 20살입니다.
================================
(src)="1283">The password is "Muiriel".
(trg)="5357">비밀번호는 "Muiriel" 입니다.
================================
(src)="1284">I will be back soon.
(trg)="5358">곧 돌아올께요.
```

**OpenSubtitles**
```
================================

(trg)="3">Sub2smi by WAF-CAP 영화가 좋아
================================
(src)="1">Through the snow and sleet and hail, through the blizzard, through the gales, through the wind and through the rain, over mountain, over plain, through the blinding lightning flash, and the mighty thunder crash,
(trg)="4">폭설이 내리고 우박, 진눈깨비가 퍼부어도
(trg)="5">눈보라가 몰아쳐도
(trg)="6">강풍이 불고 비바람이 휘몰아쳐도
================================

(trg)="7">산 넘고 물을 건너
================================

(trg)="8">번개 쳐서 눈이 멀고
================================

(trg)="9">천둥이 때려 귀가 먹어도
================================
(src)="2">ever faithful, ever true, nothing stops him, he'll get through.
(trg)="10">우리의 한결같은 심부름꾼
(trg)="11">황새 아저씨 가는 길을 그 누가 막으랴!
================================
```

As you can tell, the OpenSubtitles file is a little messier...

- There are standalone target sentences that have no source or target sentences are broken up into different lines while the source sentence is on the same line
- Sometimes the line starts with `- `

To process these files into the desired `src_sent \t trg_sent` format, I used the function below. 

```python
def make_parallel_corpus(input_path, output_path):
    NOISE = re.compile(r"^\(.+>")
    with open(output_path, 'w') as o:
        with open(input_path, 'r') as f:
            trg = src = None
            for line in f:
                if line.startswith('(src)'): 
                    src = NOISE.sub('',line).strip('- ')
                elif line.startswith('(trg)') and src!= None: 
                    if trg == None: trg = NOISE.sub('',line).strip('- ')
                    else: trg = trg+' '+NOISE.sub('',line).strip('- ')
                if src != None and trg != None:
                    o.write(src.strip()+'\t'+trg.strip()+'\n')
                    trg = src = None
```

I only processed these two datasets, so I don't know what the other ones look like, but I'm sure this code can be adapted for other datasets in OPUS if necessary. It would've been a little bit more simplistic without all the noise in the subtitles dataset but I think it still looks okay.

I've been trying to minimize data storage in all of my codes, using the least possible number of variables without sacrificing readability. I think this code would've been easier to write by just saving all the source and target sentences to a list and then writing it into the output file later. But I'm begining to find that practicing using less variables actually makes the code cleaner and I'm sure it has implications in performance, although the datasets I processed were not that large. Would be fun later to time/measure the two different types of functions and compare.

<br>

# Other En-Ko Datasets

1. [Kaist Corpus](http://semanticweb.kaist.ac.kr/home/index.php/KAIST_Corpus) has a parallel dataset though it's not very large and you have to do some minor formatting.
2. [Korean Parallel Data](https://sites.google.com/site/koreanparalleldata/) also has cleaned, aligned data for the bible, news articles and more. Again, the data is not very large but it is already formatted in the standard parallel corpus format so it's easy to use.

<br>

# Concluding Remarks...

Fortunately, most parallel datasets seem to have aleady been pre-processed so you only need to align and format them and they're ready for training!

There seem to be some repetitive data in the OPUS files, with several translations of the same sentence. Will this count as noise? Or will it be better since the machine will be able to learn more diverse translations of the same sentence? Food for thought...
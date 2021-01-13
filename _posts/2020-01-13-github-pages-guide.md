---
layout: post
title: "Github Pages 블로그 만들기"
categories: Guides
featured-img: maxresdefault
tags: [Github Pages]
---

Github과 별로 친하지 않은 사람으로써 Github를 통해서 블로그를 만들기로 결심한 이후 조금 애를 썼다.

혹시 나중에 또 블로그를 만들 때 까먹을까봐 정말 간단하게 Github를 통해 블로그를 만드는 방법에 대해서 쓰기로 했다.

아직 배우는 입장이고 모르는 기능이 많이 숨겨져있을 거 같지만 그런건 천천히 알아가고 지금은 꼭 필요한 핵심 부분만 설명하고자 한다. 

공식 가이드인 <https://guides.github.com/features/pages/> 를 보고 만들었는데 사실 결론적으로 별로 도움이 되지는 않았기에 나만의 가이드를 작성해보기로...


## Setting Up Github Pages

1. 일단 블로그 테마를 무료 사이트에서 구경한다: 
    - [Jam Stack Themes](https://jamstackthemes.dev/ssg/jekyll/) 
    - [Github Themes](https://github.com/topics/jekyll-theme)

2. 마음에 드는 테마의 repo를 fork한다. Fork 버튼은 깃헙 웹사이트 맨위 우측에 있다
3. 프로필을 눌러 'Your Repositories'에서 방금 fork한 repo로 들어간다
4. Repo Settings에서 Repository name을 username.github.io 로 변경해준다 (username은 본인 github username)

이러면 끝! 몇 분 기다리면 username.github.io 링크로 들어갈 수 있는 블로그가 생성된다.


## About Github Pages 

이제 블로그는 생성되었다. 사실 모든 테마는 사용 방법이 `README.md`에 자세히 나와있는 거 같다. 거기에 있는 가이드라인을 참고하는 게 많은 도움이 되었다. 

--

모든 Pages는 공통적으로 가지고 있는 것들이 있다:  

- `index.md`: username.github.io 들어가면 보이는 homepage 개념. 내가 사용하는 테마 같은 경우에는 여기서 title만 바꿨다. 그러면 홈페이지의 타이틀 (크롬 탭에 뜨는 이름)을 바꿀 수 있었다.
- `_layouts`: 테마의 페이지별 HTML 코드가 들어있는 폴더. 많이 건드리지 않았지만 나는 살짝 수정해서 이미지를 지웠다. 앞으로 더 연구해서 블로그 레이아웃을 변형해볼 예정
- `_posts`: 블로그 포스트를 저장해야되는 폴더 -  앞으로 가장 많이 쓸 폴더
- `config.yml`: 여기서 이것저것 변경할 수 있음. 홈페이지에 뜨는 이름/글/이메일 주소 등. 테마마다 설정 범위는 다를 것이라고 생각함
- `filename.md`  파일을 메인 폴더에 넣으면 username.github.io/filename 이라는 링크가 생성되며 그 파일의 내용이 보인다.

--

Github Pages는 YAML을 사용하는데 각 `.md` 파일에 넣어줘야 테마의 레이아웃이 적용 가능하다. 
```
---
layout: post
title: "Blogpost Title"
summary: summary here
featured-img: img_name
categories: category_name
tags: [tag1, tag2]
---
```

- `layout` 뒤에는 `_layouts` 폴더에 저장되어 있는 디자인 중 하나를 고르면 됨. 그럼 그 링크는 그 레이아웃을 가짐
- `title` 에는 꼭 `" "`을 넣어줘야 페이지가 제대로 나왔음 
- `tag`나 `categories`가 여러개 있다면 `[]` 안에 넣어주면 됨


## Posting on Github Pages
이제 블로그를 조금 customize 했다면 첫 포스트를 써보자!

가장 간단한 방법은 사실 Github Repo `_posts`에서 Create new file 하는 방법일 수도 있지만... 사진 올리고 markdown쓰고 이러려면 외부 프로그램 사용하는 게 더 편하다.

나는 코드도 VS Code를 사용하기 때문에 블로그도 VS Code를 통해서 쓰고 있다. VS CODE와 블로그를 연동했으면 VS Code에서 `.md` 파일을 만들고 `git commit` 하면 된다. 그 방법은 역시 같은 블로그를 참고했다.


- [VS Code Github 와 연동하기](https://technote.kr/352?category=940649)
- [Github 에 반영하기 (git commit/push)](https://technote.kr/353?category=940649)

### 주의점
- 블로그 포스트는 `YYYY-MM-DD-TITLE.md` 이렇게 저장해줘야 날짜랑 다 잘 맞게 나온다
- VS Code에서 포스트를 `.md` 파일로 생성할때 문서 우측 상단에 돋보기가 있는 아이콘을 클릭하면 markdown을 미리보기 할 수 있다  
   ![preview](https://i.ibb.co/DwnsjZc/Screen-Shot-2021-01-13-at-8-18-47-PM.png)  
- 포스트에 이미지를 넣으려면 이미지를 같은 폴더에 넣어야 한다. 같은 repo 다른 폴더에서 불러오는 방법이 없다 (아니면 이미지 링크/호스팅 통해서)


## 마지막으로...
댓글이나 Contact Me 같은 페이지는 워낙 테마에서 잘 되어있어서 그냥 로그인하고 글만 조금 바꾼 정도로 끝났다.

아직 블로그 포스트 프리뷰 이미지나 카테고리, 태깅하는 방법에 대해서 연구하는 중인데 일단 포스트를 쓰는데에는 문제가 없으니 나름 성공했다고 생각하고 있다.
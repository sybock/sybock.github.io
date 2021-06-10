---
layout: post
title: "How to Use Streamlit"
categories: [Guides]
featured-img: streetlight
tags: [Python]
---

# Streamlit Introduction

Again, I was training a simple chatbot and I wanted to upload it online so that other users could try it as well. I decided to use [Streamlit](https://streamlit.io/), a free app builder that is easy to use and all in Python. I honestly haven't tried using any other online app builders so I can't compare, but Streamlit was extremely easy to use with Python and required almost no HTML knowledge. I do think HTML knowledge would've helped me customize my app and make it prettier, but still, I was very satisfied with the results. So another short guide blog post today to talk about the very basics of using Streamlit.

## What is Streamlit?

Strangely, just like PyTorch Lightning from my previous post, Streamlit is kind of ambiguous on their official website about what they really do (not as much as PyTorch Lightning fortunately). I think all these sites assume you already know their basic functionalities and spend more time coming up with catchy phrases. But anyway, here's the main phrase you see when you enter their site:
> The fastest way to build and share data apps
> 
> Streamlit turns data scripts into shareable web apps in minutes.
> All in Python. All for free. No front‑end experience required.

So I guess it's not that ambiguous. But to put it in my words more simply: **Streamlit is a platform for building and sharing data apps online based on Python, requiring very little front-end coding (aka. HTML).**

As you can tell, they market their product largely for data visualization online. Like I said before, I used it to share my chatbot and I'm sure there are many other apps you can create on the site.

## Why use Streamlit?

I didn't plan for this post to be structured exactly like my PyTorch Lightning post, but it's kind of turning out that way. Because just as I did for PyTorch Lightning, I want to tell you about two main reasons for using Streamlit. 

First, **it requires minimum front-end knowledge**. I think this is a big advantage for amateur and back-end programmers. Front-end is another monster in itself, and really, who can bother shifting through all of the HTML Stack Overflow questions just to make sure your background color is right and that your text is aligned properly? Streamlit makes sure you can deploy your apps in Python and although you can customize bits and pieces, the basic Streamlit layout is already pretty good-looking.

Second, **it supports caching**. Problems with putting big apps on the web usually arise because of memory or computation limitations. Like I said before, I haven't tried out any of the other online app platforms so I can't really speak for them, but Streamlit has this caching decorator that will help your app run faster even if it has some expensive computations. For me, because chatbots and language generation in general is so expensive, I ran into problems with memory limitations even with the caching function but I think for the purpose for which Streamlit is originally built, data visualization and manipulation, this caching function should help your app run smoothly.

I do think there are some things Streamlit can do better. For one, if you're starting from scratch, it's a bit of a daunting task and I had trouble figuring out where to start. They do have some templates that you can refer to but I wish they had some pre-defined modules you could import that would structure your app for you and you could just change some of the content inside. To be honest, I ended up not making my Streamlit app from scratch, but referred to another person's chatbot code and adapted it from there. So I guess as Streamlit grows and more people create more apps, there will be more "templates" to choose from.

# Streamlit Tutorial

So there are a lot of customizable features to Streamlit that I haven't used... I'm just going to write here how I used it, which probably was the way to use it with minimum effort. But their docs are very well-written and easy to follow, so I recommend looking up specific functions on their docs.

0. `pip install streamlit`

    Just do a quick `pip install` and make sure Streamlit is running properly with `streamlit hello`. That second command line should open up a Demo site on your browser where you can look through some of the demos they've made. You can also see the code for the demo, so if your project goals align with their demos, you can actually just copy the code and customize it from there. As you'll see, Streamlit has a clean and neat visual feel.

1. Make your app

    Basically running a streamlit app consists of the following CLI command: `streamlit run your_script.py [-- script args]`. Just like running a Python code, but instead of `python3` in the beginning use `streamlit run`. To quit, use `ctrl+c`. But how do you actually write `your_script.py`?
    
    **Basics**
    - `import streamlit as st`: standard import command for Streamlit, just like `import pandas as pd`. 
    - `st.markdown()`: outputs text on site, can use standard markdown syntax. To customize using HTML, add the HTML to your text block and pass the parameter `unsafe_allow_html = True` to the function. You should customize titles or subtitles to make your site more visually catchy and organized.
    - `st.image()`: adds images to the site; pass a link to the image to the function. You can adjust the size using `width =` and `height =` parameters.
    - `st.dataframe()`: interactive table you can pass Pandas and NumPy dataframes to (the Streamlit version... meaning it's prettier) <br>
    I didn't use Streamlit to build a data app, but that's what it's meant to be used for and there a bunch of functions that helps with that, [here](https://docs.streamlit.io/en/stable/api.html#display-data). I'd like to check out some of its functions... maybe I will build a data app sometime in the future.<br>

    **Widgets**: interactive widgets where you can manipulate data
    - `st.slider()`: pass a description, minimum #, maximum #, and initial # → returns the value
    - `st.button()`: pass text to be displayed on the button → returns bool (True if pressed, False if not)
    - `st.selectbox()`: pass a description, and a tuple/array/Series/Dataframe of options to choose from → returns selected option 
    - `st.text_input()`: pass a description, a value for the text input when it first renders → returns input text  <br>
    There are many more widgets you can use, check them out [here](https://docs.streamlit.io/en/stable/api.html#display-interactive-widgets).<br>

    **Layout**
    - `st.sidebar()`: customize left sidebar... If you want to add text, do `st.sidebar.markdown`. If images, do `st.sidear.image`... and so on. It's a good place to add some of the widgets above to organize your app.<br>

    **Caching**
    - `@st.cache`: Use the decorate when loading data from the web, manipulating large datasets, or performing expensive computations. To use the cache, wrap functions that have heavy computations with the `@st.cache` decorator. For more details, check their [docs](https://docs.streamlit.io/en/stable/caching.html).<br>

2. Deploy your app

    Making the app is the hardest part. Deploying it was very easy. 
    1. Sign up for [Streamlit Sharing](streamlit.io/sharing): You have to be "invited" but usually it only takes about a day for you to be "accepted"
    2. Put your Streamlit app on GitHub: easy for anyone who's ever used GiHub... but make sure your `requirements.txt` is correctly written! Important because Streamlit will initialize using that file
    3. Log in to share.streamlit.io: you can login using GitHub if you used the same e-mail
    4. Click New app (top right blue button): select the appropriate repo, branch, and main file path from your GitHub

    And you're done!

    It takes a while to load the app the first time, but you only need to load it once. 

## Streamlit Example

So I finished training, tweaking my chatbot app and did upload it using Streamlit, you can check out the code [here](https://github.com/sybock/chatbot_streamlit_v2/blob/main/main.py) and the demo [here](https://share.streamlit.io/sybock/chatbot_streamlit_v2/main/main.py). I do not take full credit for the Streamlit app-- I adapted it from another person's [chatbot](https://github.com/bigjoedata/jekyllhydebot), which is also linked in the demo.

You organize the flow just like a Python code with functions, classes, and all the good stuff. For example, just to clean up your code a little, make displaying the main header a function, adapted from the link above: 
```python
def display_app_header(main_txt,sub_txt,is_sidebar = False):
    """
    function to display major headers at user interface
    Parameters
    ----------
    main_txt: str -> the major text to be displayed
    sub_txt: str -> the minor text to be displayed 
    is_sidebar: bool -> check if its side panel or major panel
    """

    html_temp = f"""
    <div style = "background.color:white  ; padding:15px">
    <h2 style = "color:#3c403f; text_align:center;"> {main_txt} </h2>
    <p style = "color:#3c403f; text_align:center;"> {sub_txt} </p>
    </div>
    """
    if is_sidebar:
        st.sidebar.markdown(html_temp, unsafe_allow_html = True)
    else: 
        st.markdown(html_temp, unsafe_allow_html = True)
```

As you can see, a little bit of HTML is involved for styling. In this case, the `<div>` (the division tag) is used for creating a background box for the header, the `<h2>` (the header2 tag) centers the header text and turns it into a dark grey, styled in the header2 style. Lastly, the `<p>` (the paragraph tag) centers the subtitle text and also colors it a dark grey.

Use it in `main()` like this:
```python
main_text = 'Title'
sub_text = 'Subtitle for site`
display_app_header(main_text, sub_text, is_sidebar=False)
```

You can define a similar function to customize your sidebar, or to get inputs from users and on... 

So that was the quick tutorial for streamlit, written mainly so I wouldn't forget how to use the next time I wanted to upload an app online!
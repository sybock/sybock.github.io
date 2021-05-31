---
layout: post
title: "How to use Pytorch Lightning"
categories: [Guides]
featured-img: lightning
tags: [Python]
---

# Pytorch Lightning Introduction

I was training a simple GPT2 chatbot the other day and came across a code that utilized the Pytorch Lightning package to both train the model and create generations from the model. I found it easy to use, streamlining the training, validation, evaluation, and generation processes under one Class. So before I forget how to use it again... I wanted to post a short blog about how to use the package.

## What is Pytorch Lightning?

Surprisingly, when you visit the official [Pytorch Lightning site](https://www.pytorchlightning.ai/), it's difficult to find out what exactly this package is. Here's a quote from their offical site:

> Lightning makes coding complex networks simple.
> Spend more time on research, less on engineering. It is fully flexible to fit any use case and built on pure PyTorch so there is no need to learn a new language. 

Wow, how ambiguous is that? 

To put it simply, I think it's a **package that stuffs all the messy bits of machine learning training (ex. the training steps, optimizer, evaluation steps etc) into one module.**

It's a more streamlined process to train your model through one module while still maximizing customization options.

## Why use Pytorch Lightning?

Obviously there are the detailed functions Lightning does to make your life easier, such as letting you forego the `.to(device)` function and so on, but in my opinion, there are two main high-level advantages to using one module to train, evaluate, and utilize your model. 

First, **it makes your code easier to read.** And this may seem like a minor advantage. Afterall, isn't it just easier to organize your own code rather than learn how to use a whole new package? The answer is maybe... but (a) lightning is super easy to use and (b) it's difficult to organize code! Making sure your code is reproducible and editable by other users is an important part of programming, which is why clear organization and flow is key to a good code. Using lightning makes the code readable and accessible to many other users... including yourself. Also, I find having a simple flow to the code makes it easier to incorporate CLI.

Secondly, in the process of making the code more streamlined, Lightning does not sacrifice detail. **Your code is still extremely customizable**, allowing you full control over the hyperparameters and data collation and so on. But it allows you to customize the code in an extremely simple and high-level method. For the code I was looking at with GPT2, the module also included a generation function that could be called that would allow more customizable generation than the usual HuggingFace `.generate()` method.
<br><br>

# Pytorch Lightning Tutorial

So, the most important question of all: how do you use Lightning?

Again, the official site is suprising in that when you click the tutorials section, you get a bunch of videos. Personally, I don't like video tutorials, especially for programming because I want to go back and forth between the instructions and the example code. But luckily, if you click on the docs, it's well documented and easy to understand.

## How to Use Pytorch Lightning

This section is largely based on their official docs, from a section titled [Lightning in 2 Steps](https://pytorch-lightning.readthedocs.io/en/latest/starter/new-project.html)

0. `pip install pytorch-lightning`
    - I sort of ran into problems installing the package, because there was an error about `No module named builtins` or something like that. After a short Google search, I found you can `pip install future` to solve the issue.

1. Define LightningModule
    - Create a class object that inherits from `pytorch_lightning.LightningModule`
    - Under this class, the minimal required functions are: 
        ```python
        import pytorch_lightning as pl
        class LitModel(pl.LightningModule):

            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(28 * 28, 10)
        
            def forward(self, x):
                return torch.relu(self.l1(x.view(x.size(0), -1)))
        
            def training_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self(x)
                loss = F.cross_entropy(y_hat, y)
                return loss
        
            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=0.02)
        ```
        [Source](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html)

    - Remember this module is a **system** not a model. It groups all the necessary steps and models for training in one place for ease of tracking and accessibility.
    - Although you can change it, the forward function is recommended to be used for inference (predicting) and the training_steps function for training

2. Fit with Lightning Trainer
    - Get a Data Loader
    - Create an instance of your model class that you defined above, inheriting from `LightningModule`
    - Load the trainer and call `.fit(model, dataloader)`
    ```python
    trainer = pl.Trainer()
    trainer.fit(autoencoder, train_loader)
    ```

## How to Customize Pytorch Lightning Modules

Like I said before, Lightning makes your code easier to read but still allows for maximum customization. 

In my project, I added to my module the functions necessary for generation from my pretrained GPT2 model. I also played around with some manual optimization options, which you can control if you set variable `self.automatic_optimization = False` in your `__init__`.

There seem to be other cool functions that you can play around with, including something called [callbacks](https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html), which seems kind of just like a function that you can call across multiple projects. They also provide their own [data module](https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html), which I wish I had sort of known about before I started this chatbot project because I was doing the same data processing to all my data... I hope to check out these functions soon.

## Some Miscellaneous Thoughts

Other reasons why Lightning is useful include...
- Not having to move your model and tensors `.to(device)`
- Allows optimized GPU training with ease and is compatible with DeepSpeed. (But I haven't used this function yet... Want to soon)
- Automatic logging is simple and only prints necessary information

I really thought the Hugging Face Interface was streamlined and simple enough that I wouldn't need another package to just *organize* my Hugging Face models and functions but after using it, I found it much more useful that I thought I would. I look forward to testing out the other cool functions they have, hopefully really soon.
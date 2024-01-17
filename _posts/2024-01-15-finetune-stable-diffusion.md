---
layout: post
comments: true
title: Fine-Tuning Stable Diffusion for Personalized Image Generation
author: Shaodong Wang
---


We're witnessing an AI revolution. The ability of AI models to generate personalized images is not only impressive but has vast potential applications. 
These AI models can not only generate images but also generate images that are intimately tied to specific objects and styles, 
such as portraying a person or emulating a distinctive art style.

Consider some scenarios. 1) Imagine you are an anime artist seeking inspiration. 
You wish to employ image AI to generate images that not only spark creativity but also align with your unique artistic style. 
2) You are a marketing manager. You want to use AI to generate advertising images for your products. 
The generated images should be consistent with the company values and the product styles. 
In both scenarios, the base AI model lacks knowledge of your specific style preferences. 
Training a personalized image AI model is needed.

In this blog, I will introduce the methodologies and tools to fine-tune stable diffusion models (an image generation model) for personalized use cases. 
I will also share my experience and insights of training using my own photos. This approach demonstrates how AI can be tailored to meet personalized needs.

## Training Technique Overview
To finetune the stable diffusion models, we usually feed the models a set of reference images of the target subject. 
The model is trained to link a particular term with this subject. Then, when we include the subject within the text prompt (the model’s input), 
the resulting images will maintain the target subject, e.g. a specific individual or a distinctive art style. 

There are three famous techniques: Dreambooth, Textual Inversion, and LoRA. DreamBooth adjusts model weights for specific styles or subjects, 
while Textual Inversion creates new embeddings related to provided terms. LoRA is a quicker and smaller method that adds new weights to the model.

### DreamBooth
<div style="text-align: center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/sd_finetune/diagram_dreambooth.png" 
  alt="nbeats">
  <figcaption><em> Figure 1: Diagram created by Reddit user, use_excalidraw. Source: https://www.reddit.com/r/StableDiffusion/comments/10cgxrx/wellresearched_comparison_of_training_techniques/ . </em></figcaption>
</div>

The LoRA method was originally developed for Large Language Models, but it also works well for stable diffusion models. LoRA does not change the weights in the base model directly. Instead, it adds sparse weight matrices to the base model to form the new dense weights. To make the training fast and compact, LoRA uses matrix decomposition methods. 

Basically, a large sparse matrix can be decomposed to two low-dimensional matrices. Instead of updating the entire model with huge matrices, LoRA uses the small matrices (red squares) to represent the added weight matrices of high dimensions. Thus, LoRA only needs very limited computation resources and storage space. Usually the LoRA files are around 200MB. 

## Hands-on
There are a lot of tools to train and run stable diffusion models. I highly recommend using Khoya (https://github.com/bmaltais/kohya_ss) 
to train your personalized model and run it on Automatic1111 web UI (https://github.com/AUTOMATIC1111/stable-diffusion-webui). 
We can follow the instructions on their GitHub pages to install the package. 

In my experiment, I will use my own photos to fine-tune a stable diffusion 1.5 model. 
Then, when I include my name within the text prompt, the model is supposed to present my portrait. 

### Dataset preparation
The first step is to collect training images. In my experiment, I collected 20 photos of my face. 
Typically, 15 photos are good enough. Stable diffusion models require images of 512*512 for training, but the khoya will process the images with different resolutions. 

Note that it is *very important* to have your target subject occupy a substantial portion of the image. 
For example, in my case, my face should cover at least 50% of the training photos. 
In the experiment, if the subject, like a face, only occupies a small area of the training photos, 
the quality of the output from the fine-tuned model will be significantly compromised.























# Introduction to Diffusion models

This repository provides an introduction to Diffusion models with simple examples in pytorch.
There are already many good resources explaining how diffusion models work, such as [Lilian Weng's blog post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/).
However, having a basic understanding of how diffusion works is not enough to code them.
Most available diffusion models are aimed at generating images and hence are very complex.
Here we will code a simple diffusion model in just 100 lines of python code.


## What are diffusion models?

Diffusion models belong to the class of generative models as they can create new content and are very performant at image generation.


We train a neural network to remove some level of noise from the data
![](assets/noise_scales.png)

## How to implement a diffusion model in 100 lines of python?



# Introduction to Diffusion models

This repository provides an introduction to Diffusion models with simple examples in pytorch.
There are already many good resources explaining how diffusion models work, such as [Lilian Weng's blog post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/).
However, having a basic understanding of how diffusion works is not enough to code them.
Most available diffusion models are aimed at generating images and hence are very complex.
Here we will code a simple diffusion model in just 100 lines of python code.


## What are diffusion models?

Diffusion models belong to the class of generative models as they can create new content and are very performant at image generation.


We train a neural network to remove some level of noise from the data
![noise scales](assets/noise_scales.png)

We will focus on the implementation of ["Denoising Diffusion Probabilistic Models"](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf) (DDPM).


## How to implement a diffusion model in 100 lines of python?

To code neural networks, we will use Pytorch.
In our first implementation, the only libraries needed are [torch](https://pytorch.org/) for the neural networks, [matplotlib](https://matplotlib.org/) for plotting our results, and [numpy](https://numpy.org/) for the calculations.

We will code `single_DDPM.py` which can be found in the `codes` folder.


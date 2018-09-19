# ML Basic C References

**Machine Learning Basics in C++**

Copyright (c) Pius Braun 2018

[TOC]

-----

#### Neural Networks and Deep Learning

I had the idea for this project while reading [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com) by Michael A. Nielsen. His book is licensed under a Creative Commons Attribution-NonCommercial 3.0 Unported License.

The backpropagation method `void backprop(...)` , the Stochastic Gradient Descent method `void SGD(...)` and the commandline parameteres for my `main(...)` are inspired from his python code. Like him, I use the USPS dataset for my project.

#### The Deep Learning Book

I learned the machine learning basics and most of the Math from the book [Deep Learning](http://www.deeplearningbook.org/) by Ian Goodfellow, Yoshua Bengio and Aaron Courville.

#### The Softmax function and its derivative

Eli Bendersky explains on his [website](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/), how to derive the Softmax function.

His site is © 2003-2018 Eli Bendersky.

#### The MNIST Dataset

The "Modified National Institute of Standards and Technology" dataset is a large set of handwritten digits. Like Michael A. Nielsen, I use the MNIST dataset for test and demostration purpose. 

![](C:\Users\piusb\Documents\tmp\usps1-3.png)

You should download the dataset from Yann Lecun's [site](http://yann.lecun.com/exdb/mnist/), since I use the IDX file format, that is described there.

#### The Eigen Matrix Library

My code uses the Dense Matrix classes from [Eigen](http://eigen.tuxfamiliy.org).

Eigen is primarily MPL2 licensed (see: http://www.mozilla.org/MPL/2.0/).

I installed the latest stable release 3.3.4 from www.tuxfamily.org in the `MinGW\x86_64-w64-mingw32\include` directory.

#### Compiler, O-S, Hardware

I use the MinGW environment (see http://www.mingw.org)

```
c:> g++ --Version
g++ (x86_64-posix-seh-rev0, Built by MinGW-W64 project) 7.1.0
Copyright (C) 2017 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

My ASUS Ultrabook UX305F has an Intel(R) Core(TM) M-5Y10c CPU @0.80 GHz / 998 MHz. It is running on Windows 10 Home (c) 2018 Microsoft Corporation. I use Typora (Version 0.9 Beta) to write these Markdown files. For the pictures, I use a free Version of Autodesk Sketchbook and paint.net.


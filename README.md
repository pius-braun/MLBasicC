# Machine Learning Basics in C++

Copyright (c) Pius Braun 2018

Neural networks are one of the most important methods in machine learning.

This project implements some of the widely used components of neural networks in `C++` using Back Propagation and Stochastic Gradient Descent. 

The result is an executable file that can train a neural network with different neurons (Linear, Sigmoid and Softmax) and cost functions (Quadratic, Binary Cross Entropy and Multiclass Cross Entropy). The parameters of the network can be configured from the command line.

As an example, the network trains the MNIST Dataset and achieves a hit rate of 98.47 %.

-----



## Documentation

| Section                                 | Content                                                      |
| :-------------------------------------- | ------------------------------------------------------------ |
| [Specification](docs/Specification.pdf) | describes the Math behind Neural Networks                    |
| [Implementation](Implementation.pdf)    | explains the Code                                            |
| [References](docs/References.pdf)       | contains the list of books, websites, tools and third party libraries |

-----



## Contributing

If you want to contribute to the project:

1. Write a specification in .pdf, what you intend to achive, and what's the Math behind it.
2. Let me know, what you intend.
3. Update the source code in a sandbox on your own system and test it for bugs.
4. Run the tests similar to my Test section in the documentation.



## To do

There is room for improvements:

- I did not implement the code for validation data.
- The data input is restricted to the IDX format as defined by Jann LeCun. CSV would be better.
- The results are stored to a CSV file without any real useful structure. Maybe there are better ideas to store training results.
- The network is fully connected in all layers. Convolutional networks should be better for some purposes.
- Some matrix operations in `backprop()` and `feedforward()` may run faster.






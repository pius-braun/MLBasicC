# Machine Learning Basics in C++

Copyright (c) Pius Braun 2018

Neural networks are one of the most important methods in machine learning.

This project implements some of the widely used components of neural networks in `C++` using **Back Propagation** and **Stochastic Gradient Descent**. 

The result is an executable file that can train a neural network with different neurons (Linear, Sigmoid and Softmax) and cost functions (Quadratic, Binary Cross Entropy and Multiclass Cross Entropy). The parameters of the network can be configured from the command line.

As an example, the network trains the MNIST Dataset and achieves an accuracy of up to 98.5 %.

-----



## Documentation

| Section                                    | Content                                                      |
| :----------------------------------------- | ------------------------------------------------------------ |
| [Specification](docs/Specification.html)   | describes the Math behind Neural Networks                    |
| [Implementation](docs/Implementation.html) | explains the Code                                            |
| [References](docs/References.html)         | contains the list of books, websites, tools and third party libraries |

-----



## Contributing

If you want to contribute to the project:

1. Write a specification, what you intend to achive, and what's the Math behind it.
2. Send the sepcification to me: pius.braun@t-online.de.
3. Update the source code in a sandbox on your own system and test it for bugs.
4. run the tests similar to my Test section in the documentation.



## To Do

There is room for improvements:

- I did not implement the code for validation data.
- The data input is restricted to the IDX format as defined by Jann LeCun. CSV would be better.
- The results are stored to a CSV file without any real useful structure. Maybe there are better ideas to store training results.
- The network is fully connected in all layers. Convolutional networks should be better for some purposes.
- Some matrix operations in `backprop()` and `feedforward()` may run faster if I could dig deeper into the eigen matrix code.



---
title: "Artificial Neural Network"
date: 2021-12-29T15:35:30+09:00
categories:
  - Mathematics
tags:
  - ANN
---

An **artificial neural network** is a computational learning system that uses a network of functions to find a mathematical representation of information processing. It is composed of neurons stacked in layers. The information from the input is **feed-forward**ed through all combinations of neurons until the output layer. But, to obtain the expected output from some inputs with its function, the network has to be trained first by **backprop**agating the difference between the output of the network and the expected output (target value) information through all neurons.

In this article, we are going to study the feed-forward process, the training process by error backpropagation, and some regularization techniques that are commonly used to prevent overfitting.

## Feed-Forward

<img src="https://aaumar.github.io/Math-Behind-AI/assets/images/perceptron.png"/>

The figure above briefly explains what a neuron is. Let us denote the input of the neuron to be $$\{x_i\}$$ where $$i=1,\dots,D$$ with $$D$$ is the total number of inputs. A neuron will calculate the linear combination of all inputs 

$$a=\sum_{i=1}^D x_iw_i+w_0,$$

where the quantity $$a$$ is called activation, $$\{w_i\}$$ is called the weights, and $$w_0$$ is called biases. To simplify the notation, we can define an additional input variable $$x_0$$ whose value is $$x_0=1$$ so the activation becomes

$$a=\sum_{i=0}^D x_iw_i,$$

This activation is then transformed using a differentiable, nonlinear *activation function* $$h(\cdot)$$ to give

$$z=h(a)$$

where $$z$$ is the final output of a neuron. There are some popular activation functions used in a neural network. Here we will cover 3 types of activation functions and their derivative as we will need them later for training using the gradient descent algorithm.

1. Logistic sigmoid
    
    $$h(a)=\dfrac{1}{1+e^{-a}},\quad h'(a)=h(a)(1-h(a))$$
    
2. Hyperbolic tangent
    
    $$h(a)=\dfrac{e^a-e^{-a}}{e^a+e^{-a}},\quad h'(a)=1-h(a)^2$$
    
3. Rectified Linear Unit (ReLU)
    
    $$h(a)=\max(0,a),\quad h'(a)=\left\{\begin{array}{lll}
    1,&&a>0
    \\ 
    0,&&\rm otherwise
    \end{array}\right.$$
    

The purpose of the activation function is to introduce nonlinearity to the network so it can recognize/deal with nonlinear data as well.

After understanding the mathematics behind one neuron, we are ready to construct the single-layer neural network. A single-layer neural network consists of one input layer, one hidden layer, and one output layer. The term "single" means that the network only has one hidden layer.

<img src="https://aaumar.github.io/Math-Behind-AI/assets/images/feed-forward.png"/>

The figure above describes the connections in a single-layer neural network. The first layer is called the input layer which is just the input information of the network. It is passed through the hidden units and then it is feed-forwarded to the output layers. The value in the hidden layer and output layer is calculated as follow:

$$\begin{aligned}
z_j&=h\left(\sum_{i=0}^D x_iw_{j,i}^{(1)}\right)=h\left(\bold x^T\bold w_j^{(1)}\right),
\\
y_k&=h\left(\sum_{j=0}^M z_jw_{k,j}^{(2)}\right)=h\left(\bold z^T\bold w_k^{(2)}\right),
\end{aligned}$$

where $$D$$ is the total number of inputs and $$M$$ denotes the total number of hidden neurons. Note that $$x_0$$ and $$z_0$$ are just arbitrary variables whose value is $$1$$. In the formulation above, the superscript $$w^{(1)}$$ and $$w^{(2)}$$ denotes the weight matrices for the first and second layer, respectively. For clarity, $$w_{i,j}$$ denotes the weight from neuron $$x_i$$ to neuron $$z_j$$. The final product of the feed-forward process is to get the value $$y$$ based on our input $$x$$.

## Error Backpropagation
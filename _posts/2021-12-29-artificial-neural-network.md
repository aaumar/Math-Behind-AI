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

An artificial neural network can not predict an output from data accurately before it learns from previous data. Error backpropagation is a well-known algorithm that people use to train a neural network. It calculates the gradient of an error function $$E(\bold w)$$ and applies the gradient descent algorithm to find the optimal w, so our network finds the minima of $$E(\bold w)$$. To describe the algorithm, let us consider the error function as follows

$$\begin{aligned}
E(\bold w) &= \sum_{n=1}^{N} E_n(\bold w), \\
E_n(\bold w)&=\dfrac{1}{2}\sum_{k=1}^K \left(y_{nk}-t_{nk}\right)^2,
\end{aligned}$$

where $$N$$ is the total training data pair $$(x_n,\,t_n)$$ we have for training the network and $$K$$ is the number of output we expect from the network. Here we shall consider the problem of evaluating $$\nabla_\bold w E_n(\bold w)$$.

For simplicity, let us imagine having a network with two inputs, three neurons in the hidden layer, and one output. We will summarize the notation as follows

$$\{x_i\}$$: input of the network, with $$i=\{1,2\}$$ and $$x_0=1$$.

$$w^{(1)}$$: weight connecting input and hidden layer

$$w^{(1)}=\left[\begin{array}{lll}
w^{(1)}_{0,1}&w^{(1)}_{0,2} & w^{(1)}_{0,3}\\
w^{(1)}_{1,1}&w^{(1)}_{1,2} & w^{(1)}_{1,3}\\
w^{(1)}_{2,1}&w^{(1)}_{2,2} & w^{(1)}_{2,3}\end{array}\right]=\left[\begin{array}{lll}
\bold w_1^{(1)} &
\bold w_2^{(1)} &
\bold w_3^{(1)}\end{array}\right]_$$

$$a^{(1)}$$: activation of input layer

$$\left.\begin{matrix}
a_1^{(1)}&=\sum_{i=0}^2 w_{i,1}^{(1)}x_i=\bold w_1^{(1)^T}\bold x \\
a_2^{(1)}&=\sum_{i=0}^2 w_{i,2}^{(1)}x_i=\bold w_2^{(1)^T}\bold x \\
a_3^{(1)}&=\sum_{i=0}^2 w_{i,3}^{(1)}x_i=\bold w_3^{(1)^T}\bold x
\end{matrix}\right\}\rightarrow\bold a^{(1)}=W^{(1)^T}\bold x$$

$$\{z_j\}$$: neuron of hidden layer, with $$j=\{1,2,3\}$$ and $$z_0=1$$. Let us assume the activation function is a hyperbolic tangent.

$$z_1=h(a_1^{(1)})=\dfrac{e^{a_1^{(1)}}-e^{-a_1^{(1)}}}{e^{a_1^{(1)}}+e^{-a_1^{(1)}}} \\
z_1=h(a_2^{(1)})=\dfrac{e^{a_2^{(1)}}-e^{-a_2^{(1)}}}{e^{a_2^{(1)}}+e^{-a_2^{(1)}}} \\
z_3=h(a_3^{(1)})=\dfrac{e^{a_3^{(1)}}-e^{-a_3^{(1)}}}{e^{a_3^{(1)}}+e^{-a_3^{(1)}}}$$

$$w^{(2)}$$: weight matrix connecting hidden and output layer

$$w^{(2)}=\left[\begin{array}{l}w^{(2)}_{0,1}\\w^{(2)}_{1,1}\\w^{(2)}_{2,1}\\w^{(2)}_{3,1}\end{array}\right]=\left[\bold w_1^{(2)}\right]$$

$$a^{(1)}$$: activation of hidden layer

$$\begin{aligned}
a_1^{(2)}&=\sum_{j=0}^3 w_{j,1}^{(2)}z_j=\bold w_1^{(2)^T}\bold z \\ 
a_2^{(2)}&=\sum_{j=0}^3 w_{j,2}^{(2)}z_j=\bold w_2^{(2)^T}\bold z
\end{aligned}$$

$$\{y_k\}$$: output of the network, with $$k=\{1,2\}$$. Let us assume that the activation function in the output layer is linear.

$$y_1=h\left(a_1^{(2)}\right)=a_1^{(2)} \\
y_2=h\left(a_2^{(2)}\right)=a_2^{(2)}$$

The activation unit of the output depends on the problem.

1. If the problem is classification, use the SoftMax function.
2. If the problem is regression, use the linear function.

As we already learned from [Optimizers in Neural Network], gradient descent updates the weight using the gradient of error function w.r.t. the weight. In this case, we want to update $W^{(1)}$ and $W^{(2)}$. Let us calculate the gradient of error function $\nabla_\bold wE_n(\bold w)$ w.r.t. $W^{(1)}$ and $W^{(2)}$ by assuming the activation function in the output layer is linear. We have to apply the chain rule in order to derive the cost function as follows.

$$\nabla_\bold wE_n(\bold w)=\begin{bmatrix}
\dfrac{\partial E_n(\bold w)}{\partial w_{i,j}^{(1)}}\\ \\
\dfrac{\partial E_n(\bold w)}{\partial w_{j,k}^{(2)}}
\end{bmatrix}= \begin{bmatrix}
\dfrac{\partial E_n(\bold w)}{\partial y} \cdot \dfrac{\partial y}{\partial a^{(2)}}\cdot \dfrac{\partial a^{(2)}}{\partial z_j}\cdot \dfrac{\partial z_j}{\partial a_j^{(1)}}\cdot \dfrac{\partial a_j^{(1)}}{\partial w_{i,j}^{(1)}}\\ \\
\dfrac{\partial E_n(\bold w)}{\partial y_k}\cdot\dfrac{\partial y_k}{\partial a_k^{(2)}}\cdot\dfrac{\partial a_k^{(2)}}{\partial w_{j,k}^{(2)}}
\end{bmatrix}$$

We will deal with the easy one first, the gradient of error function w.r.t. $w^{(2)}$, to get the intuition of the backpropagation algorithm.

$$\begin{aligned}
\dfrac{\partial E_n(\bold w)}{\partial w_{j,k}^{(2)}}&=\dfrac{\partial E_n(\bold w)}{\partial y_k}\cdot\dfrac{\partial y_k}{\partial a_k^{(2)}}\cdot\dfrac{\partial a_k^{(2)}}{\partial w_{j,k}^{(2)}}\\
&=(y_k-t_k)\cdot1\cdot z_j\\
\dfrac{\partial E(\bold w)}{\partial w_{j,k}^{(2)}}&=\sum_n\dfrac{\partial E_n(\bold w)}{\partial w_{j,k}^{(2)}}&
\end{aligned}$$

The real challenge of backpropagation is when you backpropagate the error further from the output layer. We will derive how to calculate the derivation of the gradient of the error function w.r.t. $w^{(1)}$.

$$\dfrac{\partial E_n(\bold w)}{\partial w_{i,j}^{(1)}}=\dfrac{\partial E_n(\bold w)}{\partial y} \cdot \dfrac{\partial y}{\partial a^{(2)}}\cdot \dfrac{\partial a^{(2)}}{\partial z_j}\cdot \dfrac{\partial z_j}{\partial a_j^{(1)}}\cdot \dfrac{\partial a_j^{(1)}}{\partial w_{i,j}^{(1)}}$$

As you can see above, $y$ and $a^{(2)}$ do not have subscript so you have to calculate the sum such as below for the first three-term on the right-hand side.

$$\dfrac{\partial E_n(\bold w)}{\partial z_j}=\sum_{k=1}^{2}\left(\dfrac{\partial E_n(\bold w)}{\partial y_k}\cdot\dfrac{\partial y_k}{\partial a_k^{(2)}}\cdot\dfrac{\partial a_k^{(2)}}{\partial z_j}\right)$$

Finally, using the equation above, we can reformulate $\dfrac{\partial E_n(\bold w)}{\partial w_{i,j}^{(1)}}$ into

$$\begin{aligned} 
\dfrac{\partial E_n(\bold w)}{\partial w_{i,j}^{(1)}}&=\dfrac{\partial E_n(\bold w)}{\partial z_j}\cdot \dfrac{\partial z_j}{\partial a_j^{(1)}}\cdot \dfrac{\partial a_j^{(1)}}{\partial w_{i,j}^{(1)}} \\
&=\sum_{k=1}^2\left((y_k-t_k )\cdot1\cdot z_j \right )\cdot\left(1-h\right(a_j^{(1)}\left)^2 \right )\cdot x_i\\
\dfrac{\partial E(\bold w)}{\partial w_{i,j}^{(1)}}&=\sum_n \dfrac{\partial E_n(\bold w)}{\partial w_{i,j}^{(1)}}
\end{aligned}$$

By acquiring the $\dfrac{\partial E(\bold w)}{\partial w_{i,j}^{(1)}}$ and $\dfrac{\partial E(\bold w)}{\partial w_{j,k}^{(2)}}$, we could update the weight using the gradient descent.
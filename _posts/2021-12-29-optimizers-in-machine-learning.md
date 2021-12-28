---
title: "Optimizers in Machine Learning"
date: 2021-12-29T15:34:30+09:00
categories:
  - blog
tags:
  - Jekyll
  - update
---

At the beginning of the machine learning era, people tend to use classical gradient descent (GD) to train the model they want to use. However, gradient descent performance will suffer if the data becomes very large. This limitation motivates people to find a better optimizers formulation so the network can train fast in any available data condition. Besides the speed of training, generalization in machine learning is also crucial as predicting the output from new data, which is not part of the training set, is the goal of the training process.

As machine learning develops rapidly, many optimizers have shown their performance in making the training phase faster. Here, we are going to look at some popular optimizers used in applications.

## Gradient Descent (GD)

Gradient descent is an iterative optimization algorithm used in training a model. It minimizes the loss function \(J(w)\) parameterized by the model's parameters $$w$$.

Suppose that we are given a training data set that consists of $N$ observations $\{\bold x_n\}$ and corresponding target values $\{\bold y_n\}$ where $n=1,\dots, N$. The goal of training a network is to achieve the performance when the network can predict the value ${\hat y}$ from any new value of $x$. 

Let us consider a simple linear function that describes the value ${\hat y}$ 

$$\hat y(x,w) = wx$$

where $w$ will be trained to minimize the difference between the network output $\hat y$ and its target $t_n$ corresponding to the observation data $x$.

Let us define the cost function as follow.

$$J(w)=\dfrac{1}{2N}\sum_{n=1}^{N}\left[\left(t_n-\hat y(x_n,w)\right)^2\right]$$

To obtain the weight, we minimize the cost function $J(w)$

$$\min_ w J(w)=\min_w \dfrac{1}{2N}\sum_{n=1}^{N}\left[\left(t_n-\hat y(x_n,w)\right)^2\right]$$

$$\nabla_w J(w)=\dfrac{1}{2N}\sum_{n=1}^{N}\nabla_w\left[\left(t_n-\hat y(x_n,w)\right)^2\right]$$

Gradient descent minimizes the cost function by updating the parameters in the opposite direction of the gradient of the objective function  $\nabla_w J(w)$ w.r.t. the weights.

$$w_{k+1}=w_k-\eta\nabla_w J(w)=w_k-\eta\dfrac{1}{2N}\sum_{n=1}^{N}\nabla_w\left[\left(\hat y(x_n,w)-t_n\right)^2\right]$$

where $\eta$ is called *learning rate* which value is bounded from 0 to 1 so the model learns well.

Choosing a learning rate is very crucial as it will affect the training process. These conditions apply when you pick a learning rate:

1. $\eta$ too small: the model will take a long time to learn.
2. $\eta$ too large: the model will not converge to the minima.
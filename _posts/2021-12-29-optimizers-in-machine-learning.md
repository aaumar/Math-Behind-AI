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

Gradient descent is an iterative optimization algorithm used in training a model. It minimizes the loss function $$J(w)$$ parameterized by the model's parameters $$w$$.

Suppose that we are given a training data set that consists of $$N$$ observations $$\{\mathbf x_n\}$$ and corresponding target values $$\{\mathbf y_n\}$$ where $$n=1,\dots, N$$. The goal of training a network is to achieve the performance when the network can predict the value $${\hat y}$$ from any new value of $$x$$. 

Let us consider a simple linear function that describes the value $${\hat y}$$ 

$$\hat y(x,w) = wx$$

where $$w$$ will be trained to minimize the difference between the network output $$\hat y$$ and its target $$t_n$$ corresponding to the observation data $$x$$.

Let us define the cost function as follow.

$$J(w)=\dfrac{1}{2N}\sum_{n=1}^{N}\left[\left(t_n-\hat y(x_n,w)\right)^2\right]$$

To obtain the weight, we minimize the cost function $$J(w)$$

$$\min_ w J(w)=\min_w \dfrac{1}{2N}\sum_{n=1}^{N}\left[\left(t_n-\hat y(x_n,w)\right)^2\right]$$

$$\nabla_w J(w)=\dfrac{1}{2N}\sum_{n=1}^{N}\nabla_w\left[\left(t_n-\hat y(x_n,w)\right)^2\right]$$

Gradient descent minimizes the cost function by updating the parameters in the opposite direction of the gradient of the objective function  $$\nabla_w J(w)$$ w.r.t. the weights.

$$w_{k+1}=w_k-\eta\nabla_w J(w)=w_k-\eta\dfrac{1}{2N}\sum_{n=1}^{N}\nabla_w\left[\left(\hat y(x_n,w)-t_n\right)^2\right]$$

where $$\eta$$ is called *learning rate* which value is bounded from $$0$$ to $$1$$ so the model learns well.

Choosing a learning rate is very crucial as it will affect the training process. These conditions apply when you pick a learning rate:

1. $$\eta$$ too small: the model will take a long time to learn.
2. $$\eta$$ too large: the model will not converge to the minima.

<img src="https://aaumar.github.io/Math-Behind-AI/assets/images/learning_rate.jpg"/>

Gradient descent does come with some drawbacks:

- Might converges to local minima: In real-life, there are a lot of cases when the data have multiple minima.
- Takes a longer training time when dealing with very large data: As gradient descent calculates the gradients for the whole dataset to perform just one update.

→ As classical gradient descent updates the parameter after evaluating the whole dataset, usually it is called as **batch gradient descent** as it processes the whole **batch** of data.

## Stochastic Gradient Descent

Stochastic gradient descent, as described from its name, introduce randomness into the learning process. While GD computes the whole dataset to update weights, SGD only takes one random point from the data and calculates the gradient from that data. In SGD, the weight $w$ is updated as

$$w_{k+1}=w_k-\eta\nabla_w\left[\left(\hat y(x_n,w)-t_n\right)^2\right]$$

where $n$ is chosen uniformly at random.

This modification leads to:

1. Enabling the learning process to be faster compared to batch gradient descent.
2. A slow convergence rate on a strongly convex function.
3. Having a better generalization rule to work with new data outside the training dataset.
4. Getting out from the local minima as a result of only evaluating the gradient of cost function from one data.

## Mini-batch Gradient Descent

To balance the advantages of batch gradient descent and stochastic gradient descent, mini-batch gradient descent comes with an idea to update the weights for every mini-batch of $m$ training examples where $$1< m < N $$.

$$w_{k+1}=w_k-\eta\dfrac{1}{2m}\sum_{n=1}^{m}\nabla_w\left[\left(\hat y(x_n,w)-t_n\right)^2\right]$$

The number of batches can be chosen freely, but keep in mind that a smaller batch size leads to better generalization. 

<img src="https://aaumar.github.io/Math-Behind-AI/assets/images/GD-type.png">

The illustration above shows the cost evolution of each gradient descent type. We can see that batch gradient descent updates the weights to their optimal very smoothly.

[](https://arxiv.org/pdf/1609.04836.pdf)

---

SGD takes a lot of attention in the literature for its ability to learn quickly. There are a lot of modifications to improve the learning rate based on SGD formulation. Here, we will learn some algorithms that are widely used in deep learning applications.
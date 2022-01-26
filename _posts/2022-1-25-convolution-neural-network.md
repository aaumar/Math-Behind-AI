---
title: "Convolution Neural Network"
date: 2022-01-25T15:37:30+09:00
categories:
  - Mathematics
tags:
  - CNN
---

Convolutional Neural Network (CNN) comes as a solution to a highly number of weight to be trained on a fully connected neural network. CNN architecture is able to reduce the number of parameters. The most important assumption about problems that are solved by CNN should not have features that are spatially dependent. This enables us to adopt a neural network architecture in image processing, where CNN is very popular in this field, as we concern to detect an object from images regardless of its position in the given images.

A convolutional neural network usually is comprised of neurons organized into three dimensions, spatial dimensionality (height and width) and depth. This type of dimension shares the same properties as the dimension of an image which usually comprises 3 dimensions with an RGB channel as the third dimension.

## Overall Architecture of CNNs

CNNs are consist of three types of layers: convolutional layers, pooling layers, and fully-connected layers. When these layers are stacked, a CNN architecture has been formed. The function of each layer type can be broken down into:

1. The input layer of a CNN will hold the pixel values of the image.
2. The **convolutional layer** will determine the output of neurons that are connected to local regions of the input through the calculation of the scalar product between their weights and the region connected to the input volume. The output of each convolutional layer will be activated with the **rectified linear unit (ReLU)** to introduce nonlinearity instead of sigmoid or tanh to prevent the vanishing gradient problem as the network design is deeper.
3. The **pooling layer** is responsible for down-sampling the image in order to reduce the complexity for further layers.
4. The **Fully-connected layer** will perform the same duties found in a traditional neural network. Please note that each of the nodes in the last pooling layer is connected as a vector to the first layer from the fully-connected layer. That is the reason why you have to flatten the last pooling layer from three-dimensional matrices to one-dimensional vectors.

To obtain the full knowledge of CNNs, let us focus on each layer and the training process of CNN.

## The Convolutional Layer

### Convolution process

Let us assume that the input of our neural network is an image such as a color image of a CIFAR-10 dataset with a width and height of $32\times32$ pixels, and a depth of 3 which RGB channel. To connect the input layer to only one neuron in a fully-connected layer, there should be $32\times32\times3$ weight connections for the CIFAR-10 dataset. Therefore, instead of a full connection, it is a good idea to look for local regions in the picture instead of in the whole image.  The figure below shows a regional connection for the next layer. 

<img src="https://aaumar.github.io/Math-Behind-AI/assets/images/convolution.png"/>

Although the size of the connection drastically dropped, it still leaves so many parameters to solve. Another assumption for simplification is to keep the local connection weights fixed for the entire neurons of the next layer. These weights for the convolutional layer are usually called kernel or filter in the literature. The convolution process of the image and the filter can be illustrated as

<img src="https://aaumar.github.io/Math-Behind-AI/assets/images/convolution-2.png"/>

where

$$
\begin{aligned}
O_{11}&=X_{11}F_{11}+X_{12}F_{12}+X_{21}F_{21}+X_{22}F_{22}\\
O_{12}&=X_{12}F_{11}+X_{13}F_{12}+X_{22}F_{21}+X_{23}F_{22}\\
O_{21}&=X_{21}F_{11}+X_{22}F_{12}+X_{31}F_{21}+X_{32}F_{22}\\
O_{22}&=X_{22}F_{11}+X_{23}F_{12}+X_{32}F_{21}+X_{33}F_{22}
\end{aligned}
$$

The output of the convolution process will be activated using rectified linear unit (ReLU) to introduce nonlinearity to the network. Details about ReLU can be obtained [here.](https://aaumar.github.io/Math-Behind-AI/mathematics/artificial-neural-network/#feed-forward)

### Stride

In the previous convolution process, we move the filter through the input one step each time to calculate the output (which is the limit of $3\times3$ input shape). To see clearly the effect of stride, let us assume that our input is a $7\times7$ image. If we move the filter one step each time, it will leave us a$5\times5$ output. But, if we move the filter two steps each time, which is called $2$ stride size, it yields an output with a dimension of $3\times3$. The effect of using stride more than one will decrease the parameters more in a convolutional layer.

To calculate the output size of a convolutional layer, let us assume given the image $N\times N$ dimension and the filter size of the $F\times F$, the output size $O$ is given as

$$
O=1+\frac{N-F}{S}
$$

where $S$ is the stride size.

###  Padding

One of the drawbacks of the convolution step is the loss of information that might exist on the border of the image. Because they are only captured when the filter slides, they never have the chance to be seen. An efficient method to resolve the issue is to use a zero-padding. Another benefit of zero padding is to manage the output size. 

<img src="https://aaumar.github.io/Math-Behind-AI/assets/images/padding.png"/>

The formula to calculate the output size using padding is

$$
O=1+\frac{N-F+2P}{S}
$$

where P is the number of the zero-padding (e.g. P=1 in the above figure). This padding helps us to prevent network output size from shrinking with the depth. Therefore, it is possible to have any number of deep convolutional networks.

## Pooling Layer

The main idea of pooling is down-sampling in order to reduce the complexity for further layers.  This layer operates over each activation mapping of the input and scales its dimensionality using the “MAX” function which is usually called max-pooling layers. It divides the image to sub-region rectangles and returns only the maximum value of the value inside that sub-region. One of the most common sizes used in max-pooling is $2\times2$.

<img src="https://aaumar.github.io/Math-Behind-AI/assets/images/pooling.png"/>

In the figure above, the input is divided into sub-regions with each sub-region having a dimension of $2\times2$. It means that stride 2 is used in the pooling process which will down-sampling the image and we might lose the position information of an object in the image. Therefore, pooling should be applied only when the presence of information is more important than spatial information.


## Fully-Connected (FC) Layer

A fully-connected layer is similar to the way that neurons are arranged in a traditional neural network. The details of this layer can be seen at [Artificial Neural Network post](https://aaumar.github.io/Math-Behind-AI/mathematics/artificial-neural-network/). This layer is usually used for classifying an object of images after performing feature extraction in the previous convolutional layer. There are a lot of classification technique that has been done before using the fully-connected layer setup so it is considered to be simpler to add this layer at the end of the convolutional neural network architecture.

## Training Process

The training process is comprised of two parts: forward pass and backward pass. In this note, we only focus on the convolutional layer as the training process of other layers. If you are interested in the training process of other layers, you can always go back to [here.](https://aaumar.github.io/Math-Behind-AI/mathematics/artificial-neural-network/)

### Forward Pass

Forward pass in the convolutional layer is a convolution operation between an input $X$ and a filter $F$. This process is actually the convolutional process as we already discussed earlier in this note. We will focus on using the same dimension as stated in the convolutional process for this section and yield the below output value.

<img src="https://aaumar.github.io/Math-Behind-AI/assets/images/convolution-2.png"/>

$$
\begin{aligned}
O_{11}&=X_{11}F_{11}+X_{12}F_{12}+X_{21}F_{21}+X_{22}F_{22}\\
O_{12}&=X_{12}F_{11}+X_{13}F_{12}+X_{22}F_{21}+X_{23}F_{22}\\
O_{21}&=X_{21}F_{11}+X_{22}F_{12}+X_{31}F_{21}+X_{32}F_{22}\\
O_{22}&=X_{22}F_{11}+X_{23}F_{12}+X_{32}F_{21}+X_{33}F_{22}
\end{aligned}
$$

### Backward Pass

<img src="https://aaumar.github.io/Math-Behind-AI/assets/images/convolution-3.png"/>

In the backward pass, we want to calculate the gradient of the loss w.r.t. the filter $F$. In this note, we assume that the gradient of the loss from the previous layer is known as $\partial L/\partial O$.  We want to calculate $\partial L/\partial F$ and $\partial L/\partial X$ in the backward pass. These can be calculated using the chain rule as can be seen in the figure above.

But the question is, why do we need to find $\partial L/\partial F$ and $\partial L/\partial X$? The first term,  $\partial L/\partial F$, is used to update the filter using gradient descent as below

$$
F_\text{new} = F_\text{current}-\alpha\cdot\frac{\partial L}{\partial F}
$$

Since $X$ is the output of the previous layer, $\partial L/\partial X$ becomes the gradient for the previous layer.

## Calculation of $\partial L/\partial F$ and $\partial L/\partial X$

### Finding $\partial L/\partial F$

To find this, we need to perform:

- Find the local gradient $\partial O/\partial F$
- Find $\partial L/\partial F$ using the chain rule

*Step 1: Finding the local gradient ($\partial O/\partial F$)*

From our convolution operation, we know that

$$
O_{11}=X_{11}F_{11}+X_{12}F_{12}+X_{21}F_{21}+X_{22}F_{22}
$$

We can find the gradient of $O_{11}$ with respect to the elements of $F$ $(F_{11}, F_{12}, F_{21}, F_{22})$ as

$$
\frac{\partial O_{11}}{\partial {F}_{11}}={X}_{11,} \quad \frac{\partial O_{11}}{\partial {F}_{12}}={X}_{12}, \quad \frac{\partial O_{11}}{\partial {F}_{21}}={X}_{21}, \quad \frac{\partial O_{11}}{\partial {F}_{22}}={X}_{22}
$$

Similarly, we can find the local gradients for $O_{12}, O_{21},$  and $O_{22}$.

*Step 2: Using the chain rule*

We need to find $\partial L/\partial F$ as

$$
\frac{\partial L}{\partial F}=\frac{\partial L}{\partial O} * \frac{\partial O}{\partial F}
$$

The complication of using the chain rule for the above equation can be written as

$$
\frac{\partial L}{\partial F_{i}}=\sum_{k=1}^{M} \frac{\partial L}{\partial O_{k}} * \frac{\partial O_{k}}{\partial F_{i}}
$$

We could expand it to make it clearer in to

$$
\begin{aligned}&\frac{\partial L}{\partial F_{11}}=\frac{\partial L}{\partial O_{11}} * \frac{\partial O_{11}}{\partial F_{11}}+\frac{\partial L}{\partial O_{12}} * \frac{\partial O_{12}}{\partial F_{11}}+\frac{\partial L}{\partial O_{21}} * \frac{\partial O_{21}}{\partial F_{11}}+\frac{\partial L}{\partial O_{22}} * \frac{\partial O_{22}}{\partial F_{11}} \\ \quad \\&\frac{\partial L}{\partial F_{12}}=\frac{\partial L}{\partial O_{11}} * \frac{\partial O_{11}}{\partial F_{12}}+\frac{\partial L}{\partial O_{12}} * \frac{\partial O_{12}}{\partial F_{12}}+\frac{\partial L}{\partial O_{21}} * \frac{\partial O_{21}}{\partial F_{12}}+\frac{\partial L}{\partial O_{22}} * \frac{\partial O_{22}}{\partial F_{12}} \\\quad \\&\frac{\partial L}{\partial F_{21}}=\frac{\partial L}{\partial O_{11}} * \frac{\partial O_{11}}{\partial F_{21}}+\frac{\partial L}{\partial O_{12}} * \frac{\partial O_{12}}{\partial F_{21}}+\frac{\partial L}{\partial O_{21}} * \frac{\partial O_{21}}{\partial F_{21}}+\frac{\partial L}{\partial O_{22}} * \frac{\partial O_{22}}{\partial F_{21}} \\\quad \\&\frac{\partial L}{\partial F_{22}}=\frac{\partial L}{\partial O_{11}} * \frac{\partial O_{11}}{\partial F_{22}}+\frac{\partial L}{\partial O_{12}} * \frac{\partial O_{12}}{\partial F_{22}}+\frac{\partial L}{\partial O_{21}} * \frac{\partial O_{21}}{\partial F_{22}}+\frac{\partial L}{\partial O_{22}} * \frac{\partial O_{22}}{\partial F_{22}}\end{aligned}
$$

If we substitute the value of the local gradients, we get

$$
\begin{aligned}&\frac{\partial L}{\partial F_{11}}=\frac{\partial L}{\partial O_{11}} * X_{11}+\frac{\partial L}{\partial O_{12}} * X_{12}+\frac{\partial L}{\partial O_{21}} * X_{21}+\frac{\partial L}{\partial O_{22}} * X_{22} \\ \quad \\&\frac{\partial L}{\partial F_{12}}=\frac{\partial L}{\partial O_{11}} * X_{12}+\frac{\partial L}{\partial O_{12}} * X_{13}+\frac{\partial L}{\partial O_{21}} * X_{22}+\frac{\partial L}{\partial O_{22}} * X_{23} \\\quad \\&\frac{\partial L}{\partial F_{21}}=\frac{\partial L}{\partial O_{11}} * X_{21}+\frac{\partial L}{\partial O_{12}} * X_{22}+\frac{\partial L}{\partial O_{21}} * X_{31}+\frac{\partial L}{\partial O_{22}} * X_{32} \\\quad \\&\frac{\partial L}{\partial F_{22}}=\frac{\partial L}{\partial O_{11}} * X_{22}+\frac{\partial L}{\partial O_{12}} * X_{23}+\frac{\partial L}{\partial O_{21}} * X_{32}+\frac{\partial L}{\partial O_{22}} * X_{33}\end{aligned}
$$

### Finding $\partial L/\partial X$

*Step 1: Finding the local gradient $(\partial O/\partial X)$*

Similar to how we found the local gradients earlier, we can find $\partial O/\partial X$ as

$$
O_{11}=X_{11}F_{11}+X_{12}F_{12}+X_{21}F_{21}+X_{22}F_{22}
$$

$$
\frac{\partial O_{11}}{\partial {X}_{11}}={F}_{11,} \quad \frac{\partial O_{11}}{\partial {X}_{12}}={F}_{12}, \quad \frac{\partial O_{11}}{\partial {X}_{21}}={F}_{21}, \quad \frac{\partial O_{11}}{\partial {X}_{22}}={F}_{22}
$$

We can find local gradients for $O_{12}, O_{21},$  and $O_{22}$ similarly.

*Step 2: Using the chain rule*

$$
\frac{\partial L}{\partial X_{i}}=\sum_{k=1}^{M} \frac{\partial L}{\partial O_{k}} * \frac{\partial O_{k}}{\partial X_{i}}
$$

By expanding the equation above and substituting the value of local gradients, we get

$$

\begin{aligned}&\frac{\partial L}{\partial X_{11}}=\frac{\partial L}{\partial O_{11}} * {F}_{11} \\ \quad \\&\frac{\partial L}{\partial X_{12}}=\frac{\partial L}{\partial O_{11}} * {F}_{12}+\frac{\partial L}{\partial O_{12}} * {F}_{11} \\ \quad \\&\frac{\partial L}{\partial X_{13}}=\frac{\partial L}{\partial O_{12}} * {F}_{12} \\ \quad \\&\frac{\partial L}{\partial X_{21}}=\frac{\partial L}{\partial O_{11}} * {F}_{21}+\frac{\partial L}{\partial O_{21}} * {F}_{11} \\ \quad \\&\frac{\partial L}{\partial X_{22}}=\frac{\partial L}{\partial O_{11}} * {F}_{22}+\frac{\partial L}{\partial O_{12}} * {F}_{21}+\frac{\partial L}{\partial O_{21}} * {F}_{12}+\frac{\partial L}{\partial O_{22}} * {F}_{11} \\ \quad \\&\frac{\partial L}{\partial X_{23}}=\frac{\partial L}{\partial O_{12}} * {F}_{22}+\frac{\partial L}{\partial O_{22}} * {F}_{12} \\ \quad \\&\frac{\partial L}{\partial X_{31}}=\frac{\partial L}{\partial O_{21}} * {F}_{21} \\ \quad \\&\frac{\partial L}{\partial X_{32}}=\frac{\partial L}{\partial O_{21}} * {F}_{22}+\frac{\partial L}{\partial O_{22}} * {F}_{21} \\ \quad \\&\frac{\partial L}{\partial X_{33}}=\frac{\partial L}{\partial O_{22}} * {F}_{22}\end{aligned}
$$

## Reference

- S. Albawi, T. A. Mohammed and S. Al-Zawi, "Understanding of a convolutional neural network," 2017 International Conference on Engineering and Technology (ICET), 2017, pp. 1-6, doi: 10.1109/ICEngTechnol.2017.8308186.
- K. O’Shea en R. Nash, “An Introduction to Convolutional Neural Networks”, *arXiv*, Nov 2015. Available: [https://arxiv.org/abs/1511.08458](https://arxiv.org/abs/1511.08458)
- “Introduction to Neural Network| Convolutional Neural Network,” Analytics Vidhya, 11-Feb-2020. [Online]. Available: [https://www.analyticsvidhya.com/blog/2020/02/mathematics-behind-convolutional-neural-network/](https://www.analyticsvidhya.com/blog/2020/02/mathematics-behind-convolutional-neural-network/)
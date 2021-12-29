---
title: "Recurrent Neural Network"
date: 2021-12-29T15:37:30+09:00
categories:
  - Mathematics
tags:
  - RNN
  - LSTM
---

In this article, we are going to introduce the Recurrent Neural Network (RNN) and how to train this type of network. In addition, we are going to cover the Long-Short Term Memory (LSTM) network which is developed based on the limitation of RNN.

## Recurrent Neural Network

The recurrent neural network is a type of network architecture that is mainly used to detect patterns in a sequence of data. The difference between RNN and the conventional Feedforward Neural Network(FFNN) is the latter passes information through the network without cycles while the RNN has cycles and transmits information back into itself. It can be visualized as in below.

<img src="https://aaumar.github.io/Math-Behind-AI/assets/images/RNN.png"/>

To describe the RNN, we denote the hidden state and the input at time step $$t$$ as $$H_t \in \mathbb{R}^{n\times h}$$ and $$X_t\in \mathbb{R}^{n\times d}$$ where $$n$$ is the number of samples, $$d$$ is the number of inputs of each sample, and $$h$$ is the number of hidden units. We use a weight matrix $$\mathbf{W}_{xh}\in \mathbb{R}^{d\times h}$$, hidden-state-to-hidden-state matrix $$\mathbf{W}_{hh}\in\mathbb{R}^{h\times h}$$ and a bias parameter $$\mathbf{b}_h\in\mathbb{R}^{1\times h}$$. These notations yield how to calculate the value in the hidden layer and output layer with $$\phi$$ as the activation function.

$$\mathbf{H}_t=\phi_h(\mathbf X_t\mathbf{W}_{xh}+\mathbf{H}_{t-1}\mathbf{W}_{hh}+\mathbf{b}_h)$$

$$\mathbf{O}_t=\phi_o(\mathbf H_t \mathbf W_{ho} + \mathbf b_o)$$

## Backpropagation Through Time

Backpropagation Through Time (BPTT) is the adaption of the backpropagation algorithm for RNNs. In theory, this unfolds the RNN to construct a traditional Feedforward Neural Network where we can apply backpropagation. 

We can define a loss function $$\mathcal L(\mathbf O,\mathbf Y)$$ to describe the difference between all outputs $$\mathbf O_t$$ and target values $$\mathbf Y_t$$.

$$\mathcal L(\mathbf O,\mathbf Y) = \sum_{t=1}^{T}\mathcal \ell_t(\mathbf O_t,\mathbf Y_t)$$

where $$\ell_t$$ can have different definitions based on the specific problem (e.g. Mean Squared Error for regression problem or Cross-Entropy Loss for classification problem).

Since we have three weight matrices $$\mathbf W_{xh}, \mathbf W_{hh}$$ and $$\mathbf W_{ho}$$, we need to compute the partial derivative w.r.t. to each of these weight matrices. Using the chain rule as we see in the backpropagation error, we can get the result as follow.

$$\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{h o}}&=\sum_{t=1}^T\frac{\partial\ell_t}{\partial\mathbf O_t}\cdot\frac{\partial\mathbf O_t}{\partial\phi_o}\cdot \frac{\partial\phi_o}{\mathbf W_{ho}} \\ &=\sum_{t=1}^{T} \frac{\partial \ell_{t}}{\partial \mathbf{O}_{t}} \cdot \frac{\partial \mathbf{O}_{t}}{\partial \phi_{o}} \cdot \mathbf{H}_{t} 
\\ \frac{\partial \mathcal{L}}{\partial \mathbf{W}_{h h}}&=\sum_{t=1}^{T} \frac{\partial \ell_{t}}{\partial \mathbf{O}_{t}} \cdot \frac{\partial \mathbf{O}_{t}}{\partial \phi_{o}} \cdot \frac{\partial \phi_{o}}{\partial \mathbf{H}_{t}} \cdot \frac{\partial \mathbf{H}_{t}}{\partial \phi_{h}} \cdot \frac{\partial \phi_{h}}{\partial \mathbf{W}_{h h}}\\ &=\sum_{t=1}^{T} \frac{\partial \ell_{t}}{\partial \mathbf{O}_{t}} \cdot \frac{\partial \mathbf{O}_{t}}{\partial \phi_{o}} \cdot \mathbf{W}_{h o} \cdot \frac{\partial \mathbf{H}_{t}}{\partial \phi_{h}} \cdot \frac{\partial \phi_{h}}{\partial \mathbf{W}_{h h}}\\
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{x h}}&=\sum_{t=1}^{T} \frac{\partial \ell_{t}}{\partial \mathbf{O}_{t}} \cdot \frac{\partial \mathbf{O}_{t}}{\partial \phi_{o}} \cdot \frac{\partial \phi_{o}}{\partial \mathbf{H}_{t}} \cdot \frac{\partial \mathbf{H}_{t}}{\partial \phi_{h}} \cdot \frac{\partial \phi_{h}}{\partial \mathbf{W}_{x h}}\\&=\sum_{t=1}^{T} \frac{\partial \ell_{t}}{\partial \mathbf{O}_{t}} \cdot \frac{\partial \mathbf{O}_{t}}{\partial \phi_{o}} \cdot \mathbf{W}_{h o} \cdot \frac{\partial \mathbf{H}_{t}}{\partial \phi_{h}} \cdot \frac{\partial \phi_{h}}{\partial \mathbf{W}_{x h}}
\end{aligned}$$

Since each $$\mathbf H_t$$ depends on the previous time step, we can substitute the derivation of cost function $$\mathcal L$$ w.r.t. $$\mathbf W_{hh}$$ and $$\mathbf W_{xh}$$ into

$$\begin{aligned}\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{h h}}&=\sum_{t=1}^{T} \frac{\partial \ell_{t}}{\partial \mathbf{O}_{t}} \cdot \frac{\partial \mathbf{O}_{t}}{\partial \phi_{o}} \cdot \mathbf{W}_{h o} \prod_{k=1}^{t} \frac{\partial \mathbf{H}_{t}}{\partial \mathbf{H}_{k}} \cdot \frac{\partial \mathbf{H}_{k}}{\partial \mathbf{W}_{h h}} \\
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{x h}}&=\sum_{t=1}^{T} \frac{\partial \ell_{t}}{\partial \mathbf{O}_{t}} \cdot \frac{\partial \mathbf{O}_{t}}{\partial \phi_{o}} \cdot \mathbf{W}_{h o} \prod_{k=1}^{t} \frac{\partial \mathbf{H}_{t}}{\partial \mathbf{H}_{k}} \cdot \frac{\partial \mathbf{H}_{k}}{\partial \mathbf{W}_{x h}}\\

\end{aligned}$$

# Vanishing Gradient Problem

In the two equations above, the term $$\dfrac{\partial\mathbf H_t}{\partial\mathbf H_k}$$ is basically introducing multiplication over the (potentially very long) sequence. If there are small values $$(<1)$$ in this multiplication, the gradient decrease over time step and finally vanishes. Otherwise, if we have large values $$(>1)$$, it will cause an exploding gradient which will make the training process very bad.

## Long-Short Term Memory (LSTM)

<img src="https://aaumar.github.io/Math-Behind-AI/assets/images/LSTM.png"/>

LSTM cell was designed to handle the vanishing gradient problem in RNN architecture. The core of the LSTM cell is the memory cell $$\mathbf C_t$$ which encodes the information of the inputs that gave been observed up to that step. The information flow is controlled via some gates inside the LSTM cell. The input gate $$\mathbf I_t$$ and output gate $$\mathbf O_t$$ control the input information to the memory unit and the output information from the unit. More specifically, the output of the LSTM cell $$\mathbf H_t$$ can be shut off via the output gate. The forget gate $$\mathbf F_t$$ can reset the memory cell with a sigmoid function (the output of a sigmoid function is $$[0,1]$$). The computation for the input gate, output gate, forget gate, and candidate memory cell $$\mathbf {\tilde C}_t$$ are as follows.

$$\begin{aligned}\mathbf{O}_{t}&=\sigma\left(\mathbf{X}_{t} \mathbf{W}_{x o}+\mathbf{H}_{t-1} \mathbf{W}_{h o}+\mathbf{b}_{o}\right) \\\mathbf{I}_{t}&=\sigma\left(\mathbf{X}_{t} \mathbf{W}_{x i}+\mathbf{H}_{t-1} \mathbf{W}_{h i}+\mathbf{b}_{i}\right) \\\mathbf{F}_{t}&=\sigma\left(\mathbf{X}_{t} \mathbf{W}_{x f}+\mathbf{H}_{t-1} \mathbf{W}_{h f}+\mathbf{b}_{f}\right)
\\ \tilde{\mathbf{C}}_{t}&=\tanh \left(\mathbf{X}_{t} \mathbf{W}_{x c}+\mathbf{H}_{t-1} \mathbf{W}_{h c}+\mathbf{b}_{c}\right)\end{aligned}$$

We introduce old memory content $$\mathbf C_{t-1}$$ which together with the introduced gates controls how much of the old memory content we want to preserve to get to the new memory content $$\mathbf C_t$$. The computation of the new memory content is

$$\mathbf{C}_{t}=\mathbf{F}_{t} \odot \mathbf{C}_{t-1}+\mathbf{I}_{t} \odot \tilde{\mathbf{C}}_{t}$$

Similar to the recurrent neural network, the hidden state is passed to the next step. The computation of hidden state $$\mathbf H_t$$ is

$$\mathbf{H}_{t}=\mathbf{O}_{t} \odot \tanh \left(\mathbf{C}_{t}\right)$$

Since the LSTM cell actually uses the same architecture as RNN, the training process is actually similar using backpropagation through time but with additional backpropagation inside the LSTM cell.

## Reference

- Schmidt, Robin M. "Recurrent Neural Networks (RNNs): A gentle Introduction and Overview." *Arxiv.* 2019. Available: [https://arxiv.org/abs/1912.05911](https://arxiv.org/abs/1912.05911)
- Gang Chen. *"*A Gentle Tutorial of Recurrent Neural Network with Error Backpropagation." *Arxiv.* 2016. Available: [https://arxiv.org/abs/1610.02583](https://arxiv.org/abs/1610.02583)
- Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory." *Neural computation* 9.8 (1997): 1735-1780. Available: [https://www.bioinf.jku.at/publications/older/2604.pdf](https://www.bioinf.jku.at/publications/older/2604.pdf)
# Positional encoding
The full sentence is passed to the encoder. From the encoder's view, these are just embeddings without any order. Therefore, positional encoding is used to tell the encoder the original position of each word in a sentence.

![image](https://github.com/guyuxuan9/Transformer-from-scratch/assets/58468284/c218d9f1-49b3-4ad2-ba0a-2b4fe1625503)


**Original paper**:

![image](https://github.com/guyuxuan9/Transformer-from-scratch/assets/58468284/53fd61b6-a49f-424c-a0cb-9f8b5e738f23)

**Code implementation** (replace $(10k)^{power}$ with $10k e^{power}$, prob easier computation):


$ div\_term = e^{-2i \frac{log(10k)}{d_{model} } } = \frac{1} { 10k \cdot e^{\frac{2i}{d_{model}}}}$,

$even\_pos = sin(position \times div\_term)$, $odd\_pos = cos(position \times div\_term)$

# MultiHeadAttention
**Intuition**: Each head captures different relationships between words.

_NB_: The following code and explanation implements the multihead attention a bit differently from that in the original paper. $Q,K,V \in \mathbb{R}^{seq, d_{model}}$

Original paper:
-  $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}, \quad \text{and} \quad W^O \in \mathbb{R}^{h \cdot d_v \times d_{\text{model}}}$
- These are weight matrix for each head

Code below:
- $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_{model} }, \quad W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_{model} }, \quad W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_{model} }, \quad W^O \in \mathbb{R}^{h \cdot d_v \times d_{\text{model}}}$
- Only one big weight matrix for $Q,K,V$.
- Split the heads after linear transformation.

![image](https://github.com/guyuxuan9/UROP_robotic_arm/assets/58468284/5e12311f-dee3-4d09-9dce-90e81c93458c)

## Linear Layer for $Q$, $K$ and $V$

$Q_{original} = \begin{bmatrix}
    q_{1,1} & q_{1,2} & \dots & q_{1,d_{model}} \\
    q_{2,1} & q_{2,2} & \dots & q_{2,d_{model}} \\
    \vdots & \vdots & \ddots & \vdots \\
    q_{\text{seq},1} & q_{\text{seq},2} & \dots & q_{\text{seq},d_{model}} \\
\end{bmatrix}$,    $K_{original} = \begin{bmatrix}
    k_{1,1} & k_{1,2} & \dots & k_{1,d_{model}} \\
    k_{2,1} & k_{2,2} & \dots & k_{2,d_{model}} \\
    \vdots & \vdots & \ddots & \vdots \\
    k_{\text{seq},1} & k_{\text{seq},2} & \dots & k_{\text{seq},d_{model}} \\
\end{bmatrix}$

The weight matrix $W_Q$ and $W_K$ have the learnable parameters. They are described by the _nn.Linear_ function. They all have dimension ($d_{model}, d_{model}$).

$W_Q = \begin{bmatrix}
\vdots & \vdots & \dots & \vdots \\
w_1^Q & w_2^Q & \dots & w_{d_{model}}^Q \\
\vdots & \vdots & \dots & \vdots
\end{bmatrix}$ $W_K = \begin{bmatrix}
\vdots & \vdots & \dots & \vdots \\
w_1^K & w_2^K & \dots & w_{d_{model}}^K \\
\vdots & \vdots & \dots & \vdots
\end{bmatrix}$

By multiplying the original embeddings with the learnable weights, the network can learn more patterns, increasing the expressive power than self-attention.

$Q = Q_{original}W_Q = \begin{bmatrix}
    q_{1,1}w_{1,1}^Q & q_{1,2}w_{1,2}^Q & \dots & q_{1,d_{model}}w_{1,d_{model}}^Q \\
    q_{2,1}w_{2,1}^Q & q_{2,2}w_{2,2}^Q & \dots & q_{2,d_{model}}w_{2,d_{model}}^Q \\
    \vdots & \vdots & \ddots & \vdots \\
    q_{\text{seq},1}w_{\text{seq},1}^Q & q_{\text{seq},2}w_{\text{seq},2}^Q & \dots & q_{\text{seq},d_{model}}w_{\text{seq},d_{model}}^Q \\
\end{bmatrix} = \begin{bmatrix}
    q_{1,1}' & q_{1,2}' & \dots & q_{1,d_{model}}' \\
    q_{2,1}' & q_{2,2}' & \dots & q_{2,d_{model}}' \\
    \vdots & \vdots & \ddots & \vdots \\
    q_{\text{seq},1}' & q_{\text{seq},2}' & \dots & q_{\text{seq},d_{model}}' \\
\end{bmatrix}$,

$K = K_{original}W_K = \begin{bmatrix}
    k_{1,1}' & k_{1,2}' & \dots & k_{1,d_{model}}' \\
    k_{2,1}' & k_{2,2}' & \dots & k_{2,d_{model}}' \\
    \vdots & \vdots & \ddots & \vdots \\
    k_{\text{seq},1}' & k_{\text{seq},2}' & \dots & k_{\text{seq},d_{model}}' \\
\end{bmatrix}$

## Split heads
(batch size, seq length, $d_{model}$) --> (batch size, # heads, seq length, $d_k$)

$\begin{bmatrix}
    q_{1,1}' & q_{1,2}' & \dots & q_{1,d_{model}}' \\
    q_{2,1}' & q_{2,2}' & \dots & q_{2,d_{model}}' \\
    \vdots & \vdots & \ddots & \vdots \\
    q_{\text{seq},1}' & q_{\text{seq},2}' & \dots & q_{\text{seq},d_{model}}' \\
\end{bmatrix}$ --> $\begin{bmatrix}
    q_{1,1}' & \dots & q_{1,k}'  \\
    q_{2,1}' & \dots &q_{2,k}'\\
    \vdots & \ddots & \vdots   \\
    q_{\text{seq},1}' & \dots & q_{\text{seq},k}' \\
\end{bmatrix}$ $\begin{bmatrix}
    q_{1,k+1}' & \dots & q_{1,2k}'  \\
    q_{2,k+1}' & \dots &q_{2,2k}'\\
    \vdots & \ddots & \vdots   \\
    q_{\text{seq},k+1}' & \dots & q_{\text{seq},2k}' \\
\end{bmatrix}$ ...

## Attention calculation

$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_{k}}}\right) V$
- **Q**: Why scaled? **A**: dot product $q.k = \sum_{i=1}^{d_k} q_ik_i$. Assume $q$ and $k$ are independent, zero mean and unit variance. $E\{q.k\} = 0$, $Var(q.k) = d_k$. If dot products gets larger, it will enter the saturation region of softmax --> vanishing gradient
- Attention has shape: (batch size, # heads, seq length,  $d_k$ )

## Combine heads
(batch size, # heads, seq length,  dk) --> (batch size, seq length, $d_{model}$)

# Position-wise Feed-Forward Networks
![image](https://github.com/guyuxuan9/UROP_robotic_arm/assets/58468284/4c2f54fb-27d0-4e19-a99f-0562123240d7)

# Encoder Layer

![image](https://github.com/guyuxuan9/Transformer-from-scratch/assets/58468284/28b7d53f-2d51-40db-9853-76a6ce400e7b)

**Batch Normalisation vs. Layer Normalisation** (Why layer norm?)

![image](https://github.com/guyuxuan9/Transformer-from-scratch/assets/58468284/5099cb58-16cf-48dd-be40-b79fc263c56e)

- Layer Norm: normalise across all features of each input word
- Batch Norm: normalise across batch of each feature

The problem of Batch Norm in NLP is that the input sentence might have various length, which is indicated by figure below.

![image](https://github.com/guyuxuan9/Transformer-from-scratch/assets/58468284/c6cd143c-bede-4a43-8c1a-0fb7781cf86b)

If the sequence length is less than the max length, zero paddings will be used to fill in the empty positions. However, the zeros added will change the mean and variance of the batch (batch statistics). Therefore, batch norm is not used in NLP tasks.

# Transformer
- **Nx** in encoder and decoder means there are several layers. The output of the previous layer is fed to the input of the current layer.
- **Purpose of the mask**: in output prediction, we don't want to look ahead, i.e. predict only based on the previous input

![image](https://github.com/guyuxuan9/Transformer-from-scratch/assets/58468284/20259fc2-ccf9-4104-bda0-5be29a9a96e3)


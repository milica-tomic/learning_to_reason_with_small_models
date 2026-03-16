# Learning to reason with small models

Implementation of the Word2Vec training loop from scratch using only NumPy with a focus on parameter efficiency and capacity analysis for small-scale language modeling. 

## Project Overview

This project implements the Skip-Gram with Negative Sampling (SGNS) variant of Word2Vec, as described in Mikolov et al. (2013b). The implementation covers the optimization procedure: forward pass, loss computation, gradient derivation, and parameter updates (with NumPy).

Additionaly, it is added small improvoments compering to standard word2vec models:

- Weight Tying: Reducing parameter count by 50%
- SVD Capacity Analysis: Measuring effective dimensionality and model saturation.

## Requirements

- Python 3.8+
- `numpy`
- `os`
- `json`
- `zipfile`
- `urllib`


## Instructions

First, if you don't have some of the libraries mentioned in the [Requirements](#requirements) section, please do as follows:
```
pip install numpy
```
After you check that all libraries are installed, run the first cell in the code.

The [Text8](https://mattmahoney.net/dc/text8.zip) and [Google Analogy](https://github.com/tmikolov/word2vec) datasets will download automatically during executon. 

You may also download the w2v_model  to independently try and evaluate the model's performance.

## Model Architecture

The model consists of two embedding matrices (both of shape `(V, D)`). Based on the configuration, it supports two choices:

1. Standard SGNS:

    - **`W_in`** — center-word embeddings, used as final word vectors
    - **`W_out`** — context-word embeddings, used only during training
    
    Both matrices are initialized with small random uniform values:
    
    ```python
    W_in  = (random(V, D) - 0.5) / D
    W_out = (random(V, D) - 0.5) / D
    ```
    
    Only `W_in` is used for evaluation. The difference, compared to Mikolov et al. (2013b), is the initialization of W_out. In this implementation, it is initialized with random values, whereas in the original paper, it is initialized only with zeros. This choice was made to facilitate faster symmetry breaking and prevent numerical plateaus during the initial stages of training, similar to the approach used in **Gensim**.

2. Weight - Tied SGNS:

   `W_out` shares the same memory as `W_in`. This reduces the model size by half, serving as a base line for parameter-efficent "small models".

   ```python
    if self.tie_weights:
    self.W_out = self.W_in
    ```

## Training Procedure

### Skip-Gram Objective

For each center word $v_c$, predict its surrounding context words within a window.  
The **dynamic window** $c \sim \text{Uniform}[1, \text{window}]$ gives closer words a higher effective weight.

### Negative Sampling Loss (SGNS)

Instead of a full softmax over the vocabulary, SGNS replaces the problem with $K$ binary classification tasks:

$$\mathcal{L} = -\left[ \log \sigma(\mathbf{u}_o \cdot \mathbf{v}_c) + \sum_{k=1}^{K} \log \sigma(-\mathbf{u}_{n_k} \cdot \mathbf{v}_c) \right]$$

where $\sigma$ is the sigmoid function, **u_o** is the true context vector, **u_n_k** are $K$ noise word vectors sampled from the $\text{unigram}^{0.75}$ distribution.

This reduces the per-step cost from $O(V \cdot D)$ (full softmax) to $O(K \cdot D)$.

### Gradient Derivation

Applying the chain rule through the sigmoid $\sigma(x)$, using $\frac{d}{dx}\log\sigma(x) = 1 - \sigma(x)$:

**Center word vector** $\mathbf{v}_c$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}_c} = \underbrace{(\sigma(s_{pos}) - 1)}_{\text{positive error}} \cdot \mathbf{u}_o + \sum_{k=1}^{K} \underbrace{\sigma(s_{neg_k})}_{\text{negative error}} \cdot \mathbf{u}_{n_k}$$

**True context vector** $\mathbf{u}_o$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{u}_o} = (\sigma(s_{pos}) - 1) \cdot \mathbf{v}_c$$

**Noise vector** $\mathbf{u}_{n_k}$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{u}_{n_k}} = \sigma(s_{neg_k}) \cdot \mathbf{v}_c$$

The error signals $(\sigma - 1)$ and $\sigma$ drive the updates: when the model is correct, the errors approach zero and the update vanishes.

These gradients map directly to the code:

```python
err_pos = (sig_pos - 1.0) 
err_neg = sig_neg 

grad_vc    = err_pos[:, None] * u_pos + np.einsum('bk,bkd->bd', err_neg, u_neg)
grad_u_pos = err_pos[:, None] * vc
grad_u_neg = np.einsum('bk,bd->bkd', err_neg, vc)
```

### Parameter Updates

SGD (Stochastic Gradient Descent) with linear learning rate decay (as in the original paper):

```python
lr = max(lr_start * (1 - processed / total), lr_min)
```

Updates use `np.add.at` instead of plain indexing to correctly accumulate gradients when the same word index appears multiple times in a batch:

```python
np.add.at(W_in,  centers,  -lr * grad_vc)
np.add.at(W_out, contexts, -lr * grad_u_pos)
```

### Subsampling of Frequent Words

Frequent words (the, of, a) are discarded with probability:

$$P(\text{keep}) = \sqrt{\frac{t}{f(w)}}, \quad t = 10^{-5}$$

This reduces corpus size and improves vector quality for rare words.

### Noise Distribution

Negative samples are drawn from the $\text{unigram}^{0.75}$ distribution:

$$P(w) \propto f(w)^{0.75}$$

The exponent 0.75 reduces the dominance of frequent words as negatives without fully equalising to a uniform distribution.


## Batch Vectorisation

The training loop processes `B` pairs simultaneously instead of one by one (in the original Word2Vec C implementation processes samples one by one). This shifts the computational load from the slow Python level to NumPy's highly optimized C backend. 

The mathematics are identical to the per-sample case — dimensions are larger (due to vectorisation)

The negative dot products require `np.einsum`:

```python
s_neg = np.einsum('bd,bkd->bk', vc, u_neg)
# for each b: s_neg[b,k] = vc[b] · u_neg[b,k]
```

## Capacity Analysis (SVD)

After training, the effective dimensionality of `W_in` is measured via Singular Value Decomposition (SVD). This quantifies how much of the nominal embedding dimension is actually used to encode meaningful structure.

The singular values $s_1 \geq s_2 \geq \ldots \geq s_D$ of `W_in` are computed and the explained variance ratio for each component is:

$$\text{ExpVar}(i) = \frac{s_i^2}{\sum_{j=1}^{D} s_j^2}$$

The **effective dimension** ($k^*$) is the smallest $k$ such that the cumulative explained variance reaches a set threshold (default: 90%):

$$k^* = \min \lbrace k : \sum_{i=1}^{k} \text{ExpVar}(i) \geq 0.90 \rbrace$$

k* is the recommended value for `DIM` in the next training run — it is the number of dimensions that actually carry information given the corpus and task. Setting `DIM` = k* avoids wasting compute on directions that contribute nothing, while a `DIM` much larger than k* indicates the model is over-parameterized for the current data.

## Evaluation

### 1. Nearest Neighbours

Cosine similarity between a query vector and all vectors in `W_in`:

```
Nearest neighbours of 'king':
  queen    0.78
  prince   0.71
  ...
```

Note that `vii` and `viii` appear due to frequent co-occurrence with
  royal names (Henry VIII, Louis VII) in Wikipedia text — a direct
  consequence of distributional semantics rather than explicit encoding.

### 2. Analogy Accuracy

Tests the linear structure of the embedding space: **a − b + c = ?**

```
king - man + woman  →  queen    ✓
paris - france + germany  →  berlin   ✓
```

Accuracy = % of analogies where top-1 prediction matches the expected word.

The benchmark used is the original Google word analogy dataset (Mikolov et al., 2013a), containing 19,544 analogies split into semantic categories (e.g. capital cities, currency) and syntactic categories (e.g. plural forms, verb tenses). Words absent from the vocabulary are skipped and counted separately.

**Result: 1.4% accuracy (241/17,827 evaluated, 1,717 skipped)**

This low accuracy is expected and consistent with the literature.
The original Word2Vec paper reports ~70% accuracy, but was trained on
100 billion tokens, approximately 25,000x more data than Text8 (~4M tokens).

> **Note:** Accuracy is expected to be modest at `DIM=100` with 3 epochs on Text8. Larger dimension, more epochs, and untied weights improve this significantly.


## References

- Mikolov, T. et al. (2013a). *Efficient Estimation of Word Representations in Vector Space.* arXiv:1301.3781
- Mikolov, T. et al. (2013b). *Distributed Representations of Words and Phrases and their Compositionality.* NeurIPS 2013

# Character-Level Feedforward Language Model in PyTorch

A from-scratch exploration of **autoregressive language modeling**, progressing from classical **n-gram language models** to a **character-level Feedforward Neural Language Model** implemented with PyTorch.

The project focuses on understanding how language models evolve from simple frequency-based statistical models to neural models that learn distributed representations and predict the next character from a fixed context window.

---

## Overview

A language model estimates the probability of the next token given the previous context:

$$
P(x_t \mid x_1, x_2, \ldots, x_{t-1})
$$

This project demonstrates several approaches to autoregressive language modeling:

1. Unigram Language Model
2. Bigram Language Model
3. Trigram Language Model
4. General N-Gram Language Model
5. Feedforward Neural Language Model
6. Character-Level Feedforward Language Model in PyTorch

The main neural model uses an embedding layer followed by fully connected layers to predict the next character from a fixed-size context.

---

## Project Goals

The main goals of this project are to:

- Understand autoregressive language modeling
- Implement classical n-gram models from scratch
- Understand the limitations of count-based language models
- Introduce neural language modeling
- Learn how embeddings allow models to generalize across contexts
- Implement a character-level Feedforward Language Model using PyTorch
- Train and validate the model
- Evaluate the model using loss and perplexity
- Generate text using the trained model
- Study the effect of sampling temperature on generated text

---
## Running the Notebook

Clone the repository:

```bash
git clone https://github.com/NESHADUR/character-level-feedforward-language-model.git
```

## Learning Outcomes

This project provides a practical progression from statistical to neural language modeling and demonstrates:

- Chain-rule factorization of language models
- N-gram modeling
- Maximum likelihood estimation
- Neural embeddings
- Feedforward neural networks
- Next-character prediction
- Cross-entropy training
- Validation and test evaluation
- Perplexity
- Autoregressive text generation
- Temperature-based sampling

---

## Future Improvements

Possible extensions include:

- RNN-based language modeling
- LSTM language model
- GRU language model
- Transformer language model
- Larger training corpus
- Subword tokenization
- Beam search
- Top-k sampling
- Nucleus (top-p) sampling
- Better experiment tracking
- More systematic hyperparameter experiments

---
This project was developed as an educational exploration of language modeling and neural text generation using PyTorch.
---

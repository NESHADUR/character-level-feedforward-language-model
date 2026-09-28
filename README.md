# Text Generation with Language Models

**From unigrams to GRUs: building autoregressive language models from the counts up.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An exploration of autoregressive language modeling, progressing from classical word-level n-gram language models to character-level neural language models: feedforward, RNN, LSTM, and GRU, all implemented with PyTorch.

The project focuses on understanding how language models evolve from simple
frequency-based statistical models to neural models that learn distributed
representations and predict the next character.

## Overview

A language model estimates the probability of the next token given the previous
context:

$$
P(x_t \mid x_1, x_2, \dots, x_{t-1})
$$

That single definition *is* text generation. Start from a few words, repeatedly
sample the next token from the model's distribution, feed that prediction back in
as context, and repeat. The model never "writes" anything — it emits one
probability distribution at a time, and the text is the sum of those choices.
Feed it *"the harpooners arrived to the"* and the differences between models show
up immediately in what comes next.

This project demonstrates several approaches to autoregressive language modeling:

- Unigram Language Model
- Bigram Language Model
- Trigram Language Model
- General N-Gram Language Model
- Feedforward Neural Language Model
- Recurrent Neural Network Language Model
- LSTM Language Model
- GRU Language Model

### The neural models in PyTorch

The feedforward model uses an embedding layer followed by fully connected layers
to predict the next character from a fixed-size context. The three recurrent
models replace that fixed window with a hidden state carried across the
sequence, so the context is no longer bounded. Both variants end at the same
place: a softmax over the character vocabulary, followed by temperature and
top-k sampling.

## The four models

All four neural approaches answer the same question and differ only in **how they
remember what came before**.

| Model                         | How it remembers                                                                       | How it generates                                                                                         | What it can't do                                                               |
| ----------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **N-gram** (unigram → 4-gram) | Counts the exact last *n−1* words it has seen before                                   | Emits the most common continuation of that phrase. A `[/s]` end token teaches it where a sentence stops. | An unseen phrase gets probability exactly **0**, so the sentence is impossible |
| **Feedforward**               | Embeds a fixed window of the last *n* characters, flattens it through one hidden layer | Picks the next character from that window's representation                                               | Only sees the window. More text means a bigger window, and it still forgets    |
| **RNN**                       | Carries a hidden state forward, rewriting it at every step                             | Reads the whole prefix, however long                                                                     | The state decays — distant context fades to noise                              |
| **LSTM**                      | Adds a cell state and three gates that decide what to **keep** and what to **forget**  | Reads the whole prefix, and the gates stop distant context from fading                                   | Four times the parameters of an RNN                                            |
| **GRU**                       | Collapses the LSTM's two states into one, with two gates                               | Same accuracy as the LSTM here, with **~24% fewer** parameters                                           | Slightly less capacity in principle                                            |

**Two notebooks. Six public-domain novels.** No `transformers`, no `nltk`, no
Hugging Face. You write the counting, the special tokens, the loss, the gradient
step, and the perplexity each model actually reached.

> Sequence-to-sequence and attention live in a **separate repository**, because
> they are about translation rather than language modeling.

---

## Project goals

- Understand autoregressive language modeling and the chain-rule factorization
- Implement classical n-gram models from scratch, by counting
- Understand the limitations of count-based models — specifically, the zero
  probability assigned to anything unseen
- Introduce neural language modeling and learned embeddings
- See how embeddings let a model generalize across contexts that never co-occurred
- Implement and train a character-level feedforward neural language model
- Understand why maximum likelihood estimation *is* cross-entropy loss
- Evaluate models with loss, accuracy and perplexity
- Generate text from a trained model
- Study the effect of sampling temperature and top-k on generated text
- Move from a fixed context window to a recurrent state, and see what that buys
- Understand why gated cells (LSTM, GRU) beat a vanilla RNN

## Learning outcomes

After working through both notebooks you should be able to:

- Factorize a sequence probability by the chain rule and explain what the Markov
  assumption truncates
- Build a conditional probability table by counting, and explain why the
  denominator is a context count
- Derive perplexity from cross-entropy and interpret it as an effective
  vocabulary size
- Explain why a trigram assigns probability exactly zero to a sensible sentence
- Write a language model as a `nn.Module` and train it with a hand-written loop
- Explain what a logit table of shape `V^(n−1) × V` represents
- Explain why learned embeddings fix the zero-probability problem
- Distinguish a fixed context window from a recurrent hidden state
- Describe what the LSTM's forget, input and output gates each do
- Explain why the GRU reaches comparable quality with fewer parameters
- Generate text and explain what temperature and top-k actually change
- State which of your results would not survive a rerun, and why

---

## The progression

Each family exists because of a specific failure in the one before it.

| Family                                   | Idea                                            | Dies because                                                             |
| ---------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------ |
| **Count-based** (n-gram)                 | Estimate `P(x_t \| x_1…x_{t-1})` by counting    | Cannot generalize — an unseen combination gets probability exactly **0** |
| **Neural, feedforward**                  | Learned embeddings, fixed context window        | Fixed context — cannot attend further back than the window               |
| **Neural, recurrent** (RNN → LSTM → GRU) | Carry a hidden state across arbitrary distances | Information decays with distance; cannot parallelize                     |

The order is the point. Each notebook ends by naming the exact failure of the
model it just trained, and the next one opens with that failure.

---

## The notebooks

### [`Statistical Language Model.ipynb`](Statistical%20Language%20Model.ipynb)

**Word-level, count-based, no neural networks at all.**

1. **Autoregressive formulation:** the chain rule, and what "context" means.
2. **The Markov assumption:** why truncating history to *n−1* tokens is a choice,
   and what it costs.
3. **Evaluation:** log-likelihood, per-word log-likelihood, and perplexity,
   derived from each other, with a worked numerical example.
4. **Unigram → Bigram → Trigram → N-gram:** each built twice: from scratch with
   `Counter` so the count ratios are visible, then as a learnable PyTorch model
   trained by gradient descent, so it is visible that the learned distribution
   converges to the counted one.
5. **The bridge:** why a trigram assigns exactly zero to a sensible sentence,
   and why learned embeddings are the answer.

### [`Neural Network-based Language Models.ipynb`](Neural%20Network-based%20Language%20Models.ipynb)

**Character-level, trained on ~4.2M characters of six public-domain novels.**

Trains the feedforward, RNN, LSTM and GRU models from the table above, each with
its own data pipeline, `Dataset`, training loop with early stopping, test
perplexity, and training curves.

> **What is hand-written and what is not.** The feedforward model, its embedding
> layer, its training loop, its evaluation and its sampler are all written out in
> the notebook. The three recurrent models **use PyTorch's built-in `nn.RNN`,
> `nn.LSTM` and `nn.GRU`** as their recurrent layer — what you write is the
> embedding, the output projection, the loss, the training loop, and the
> generator around them. Hand-written `ManualLSTMCell` and `ManualGRUCell`
> implementations, each with a numerical check against the corresponding
> `nn.*Cell`, are included in the notebook but are **commented out** and do not
> run.

Maximum likelihood becomes cross-entropy here: the feedforward section derives
both, so you can see that the loss the optimizer minimizes is the negative
log-likelihood of the training corpus, nothing more.

#### Results

Character-level, 80/10/10 split, from the saved notebook outputs.

| Model       | Vocab | Context | Batch | Params  | Test loss | Test acc | **Test PPL** |
| ----------- | -----:| -------:| -----:| -------:| ---------:| --------:| ------------:|
| Feedforward | 120   | 40      | 256   | 362,616 | 1.5829    | 0.5408   | **4.87**     |
| RNN         | 114   | 100     | 128   | 274,290 | 1.4478    | 0.5690   | **4.25**     |
| LSTM        | 114   | 100     | 128   | 965,490 | 1.3109    | 0.5985   | **3.71**     |
| GRU         | 114   | 100     | 128   | 735,090 | 1.3059    | 0.6005   | **3.69**     |

Shared settings: AdamW, lr 1e-3, weight decay 1e-4, cosine schedule, gradient
clipping 1.0, early stopping patience 5, up to 80 epochs, `SEED = 42` for the
three recurrent models. **All recorded runs are CPU runs.** No weight tying.

The ordering is the expected one — recurrence beats a fixed window, and gated
cells beat a vanilla RNN. The GRU reaches LSTM quality with ~24% fewer
parameters, which is the point of the two-gate design.

> ⚠️ **Do not read the feedforward row as part of that comparison.** Its split is
> taken over the concatenated corpus, so *Alice's Adventures in Wonderland*
> (3.5% of the corpus) lands entirely in the test set and is never trained on, and
> that section sets no random seed. Treat 4.87 as indicative only.

---

## Setup

Python 3.11 is what the notebooks were last run against; 3.9+ should work.

```bash
git clone https://github.com/NESHADUR/ngram-to-recurrent-language-models.git
cd ngram-to-recurrent-language-models

python -m venv .venv
# Windows:     .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
pip install jupyter
```

Or with conda: `conda env create -f environment.yml && conda activate ngram-to-recurrent-language-models`.

## Running

Launch Jupyter **from the repository root** — that is what makes the book
filenames resolve:

```bash
cd ngram-to-recurrent-language-models
jupyter notebook
```

Then **Kernel → Restart Kernel and Run All**.

Notebook 01 finishes in seconds. Notebook 02 trains four models from scratch and
takes considerably longer; a CPU-only run of the recurrent half is measured in
hours. It picks CUDA automatically if a GPU is present.

### What gets written where

| Artifact                                | Written by                   | Tracked?        |
| --------------------------------------- | ---------------------------- | --------------- |
| `*_final.pt` checkpoints                | `torch.save` in notebook 02  | No — gitignored |
| `*_curves.png` in the working directory | `plt.savefig` in notebook 02 | No — gitignored |
| `images/*.png`                          | Curated figures              | Yes             |

---

## Layout

```
.
├── Statistical Language Model.ipynb
├── Neural Network-based Language Models.ipynb
├── scripts/
│   └── lstm_full_implementation.py     # Standalone character-level LSTM LM
├── images/                             # 18 figures, descriptively named
├── .github/                            # CI, issue templates, PR template
├── LICENSE  CITATION.cff  Makefile  environment.yml  requirements.txt
└── README.md  .gitignore  .editorconfig
```

Notebooks and the six `.txt` files sit at the root **on purpose**: the notebooks
open the books by bare filename, so they must share a working directory.

---

## Corpus

Six Project Gutenberg novels, public domain in the United States.

| File                                    | Author             | Raw           | Cleaned       |
| --------------------------------------- | ------------------ | -------------:| -------------:|
| `The Mysterious Island.txt`             | Jules Verne        | 1,112,362     | 1,112,264     |
| `The Modern Prometheus.txt`             | Mary Shelley       | 438,841       | 419,342       |
| `The Whale.txt`                         | Herman Melville    | 1,238,279     | 1,218,947     |
| `The Adventures of Sherlock Holmes.txt` | Arthur Conan Doyle | 581,601       | 562,213       |
| `Pride and Prejudice.txt`               | Jane Austen        | 748,167       | 728,750       |
| `Alice's Adventures in Wonderland.txt`  | Lewis Carroll      | 163,950       | 144,604       |
| **Total**                               |                    | **4,283,285** | **4,186,205** |

The feedforward section uses the raw text, which keeps the Gutenberg licence header and footer. That is why its vocabulary is 120 characters rather than 114; the footer contributes ™, ‎, ‏, and •. The cleaned column strips everything between the *** START OF and *** END OF markers, and is what the RNN, LSTM, and GRU sections use. The two halves of the notebook therefore model slightly different data.

To move the books into a `data/` directory, change one line per notebook:

```python
DATA_DIR = pathlib.Path("data")
...
with open(DATA_DIR / path, "r", encoding="utf-8") as f:
```

and keep launching Jupyter from the repository root. That change is not applied
here because it modifies code cells.

To rebuild the corpus from scratch:

```bash
mkdir -p data && cd data
for id in 1268 84 2701 1661 1342 11; do
  curl -L -o "${id}.txt" "https://www.gutenberg.org/cache/epub/${id}/pg${id}.txt"
done
```

Then rename each file to the title the notebooks expect.

---

## Citation

```bibtex
@software{ngram_to_recurrent_language_models,
  author  = {NESHADUR},
  title   = {From N-Grams to Recurrent Networks: Language Models Built From Scratch},
  year    = {2026},
  url     = {https://github.com/NESHADUR/ngram-to-recurrent-language-models},
  license = {MIT}
}
```

## License

Code and prose: [MIT](LICENSE). The six Project Gutenberg novels are **not**
covered by that license — they are public domain in the United States, but Project
Gutenberg's trademark and distribution terms are separate from the copyright
status of the works. See [Corpus](#corpus).

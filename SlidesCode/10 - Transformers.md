---
marp: true
title: Week 6 - RNN and LSTM Models for Time Series
theme: default
class: default
size: 4:3
---

# Transformers and Modern LLMs

Following [D2L Chapter 11](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html)

---

# Kernel Density Estimation

A(nother) way to estimate non-linear regressions:
- Weight the observations based on their distance from the current location
- One version of this is LOWESS (local weighted sum of squares), which we use in `plotly`

![](https://d2l.ai/_images/output_attention-pooling_d5e6b2_18_0.svg)

---

Regression Lines under KDE

![](https://d2l.ai/_images/output_attention-pooling_d5e6b2_63_0.svg)

How slope is estimated

![](https://d2l.ai/_images/output_attention-pooling_d5e6b2_78_0.svg)

---

# So what?

While KDE methods are really great for semi-parametric models, some smart people (Vaswani et al, 2017) decided to generalize these concepts to even broader classes of models
- How can we selectively attend to different data points and inputs in a neural network?
- Can we make our model focus on more important words in a query, or critical characteristics in an image?

Yes! We can!

---

# Defining Terms

$\mathcal{D}$: a database of key-value pairs
$k$: a key
$v$: a value
$q$: a query where we want to look up what the correct value is given a key

This is easy where all queries exist in $\mathcal{D}$ (like in Python's dictionaries), but what do we do when we have a query that doesn't exist? We don't want to be like Python and just throw an error when a new question is asked.

---

# Attention

$$\alpha(q, k_i) = softmax(a(q, k_i)) = \frac{exp(q^\intercal k_i/\sqrt{d})}{\sum_j exp(q^\intercal k_j/\sqrt{d})}$$

In English, we provide the output that our weights indicate is most applicable across our set of inputs as the relevant feature for our model to leverage at any particular point

---

# Attention Diagram

![](https://d2l.ai/_images/attention-output.svg)

Our weights help us to choose which pieces of information we pass forward through our network, and softmax makes us focus on a single key for each query in a given layer

---

# Encoder and Decoder

Our attention models can have two parts

1) Encoder - Where we take our query and encode the information in it into our information set (embedding). Trained to **ingest** data
2) Decoder - Where we take our ingested data and use it to generate an output

So an attention model is actually TWO neural networks that are connected to each other: one encoder and one decoder

---

# So what does Attention do?

Our model will get REALLY big if we try to make it memorize all of the input information in order to remember how to generate output information.

Instead, we teach our model how to pay attention to the important parts to streamline its calculations and memory requirements

---

# Multi-head Attention

Just like our inception models, we can create parallel streams to model our inputs so that we can explore the data from different angles

![](https://d2l.ai/_images/multi-head-attention.svg)

---

# Positional Encoding

We also include information about WHERE a specific token appears in a sequence as **separate inputs**

- This allows us to parallel process our data rather than work sequentially!
- Reduces computation time drastically, and allows for much greater scaling of models than was previously possible

---

# The Transformer

![](https://d2l.ai/_images/transformer.svg)

---

# Let's Make One!

Follow the code here:

[https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html)

Data looks like this: [https://www.manythings.org/bilingual/fra/1.html](https://www.manythings.org/bilingual/fra/1.html)

---

# Modern Models - BERT

Encoder only!

![](https://d2l.ai/_images/bert-encoder-only.svg)

---

# T5

Encoder-Decoder

![width:600](https://d2l.ai/_images/t5-encoder-decoder.svg)

---

# T5 Training

![width:800](https://d2l.ai/_images/t5-finetune-summarization.svg)

---

# ChatGPT (1 and 2)

![width:600](https://d2l.ai/_images/gpt-decoder-only.svg)

---

# Training GPT3, and "shots"

![width:600](https://d2l.ai/_images/gpt-3-xshot.svg)


---

# Lab time!
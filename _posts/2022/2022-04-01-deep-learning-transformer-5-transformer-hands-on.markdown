---
layout: post
title: Deep Learning - Transformer Series 5 - Transformer Hands On
date: '2022-04-01 13:19'
subtitle: Hands-On Transformer Training and Validation
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Tasks and Data

It's common practice to pad input sequences to `MAX_SENTENCE_LENGTH`. Therefore,

- the input is always [batch_size, max_sentence_length]
- `NUM_KEYS = NUM_QUERIES = max_sentence_length` since neither the encoder nor the decoder changes the `max_sentence_length` dimension

In practice, one can apply below methods to reduce padding:

- Bucketing - bucketing is to group sentences of similar lengths to reduce sentence lengths.
- Packed Sequences: PyTorch's `pack_padded_sequence` and `pad_packed_sequence` utilities (more common in RNNs) to handle variable-length sequences.

Applications

- Machine Translation (using World-Machine-Translation datasets)
- Named Entity Recognition (like extracting "phone number" from resumes)

## Quick Word on Teacher Forcing

In seq2seq, we use outputs y(t-1) as x[t]. This is common in machine translation. However, this method suffers slow convergence and less stability. Teacher Forcing is to use the teacher signal `tgt[t]` as the input `x[t]`

Without autoregression in training, this would look like:

```python
logits = model(src_spanish_tokens, ground_truth_english_tokens...)   #[batch_size, sentence_length, output_embedding_dims]
criterion(logits, target)    #
```

## Lessons Learned From Hands-On Training and Validating

I assembled my own version of the transformer, and tested them against the torch implementations. There were some bugs so I decided to assemble a test bench that could work on `nn.Transformer`. Later, if I can make sure the I/O are consistent in my custom version, there's a good chance my custom transformer will work too.

### Lesson 1 - Model Loss Was High In Training

When I started training, the loss got stuck quite early on. At first, I was thinking "how could I debug this? This is like a black box". It turned out that there were still things that can **narrow down the bugs**.

1. Try overfitting a small batch and see if it works. This made debugging much faster. I got to validate this problem.
2. **Turn on a debugger and understand the data flow**. It was in this step that I finally found that when feeding logits and target batch into `nn.CrossEntropyLoss`, the dimensions of my target batch were swapped. No errors were thrown because I needed to flatten them for loss calculation. However, due to the wrong dimensions, the data were wrong.

- It was also in this step that I got a much better understanding of how the model really forms the attention with this data.
- The VSCode debugger is really handy. It's definitely a **good time investment to look into a handy debugger for programming projects.** The visibility of the system is gold.

### Lesson 2 - Vanishing Gradients

My model started outputting 'NaN' after a few batches. This is the same problem as this [StackOverflow Post](https://stackoverflow.com/questions/66542007/transformer-model-output-nan-values-in-pytorch). **The training architecture is most likely well-defined considering the first few batches are good.**

1. I turned on `torch.autograd.set_detect_anomaly(True)`. I could see that it was the decoder in `nn.Transformer` where `NaN` arose. But this is not enough, we need to examine what caused it. **I'm thankful that the VSCode Python Debugger actually shows all variable values**. I saw the positional embedding layer before the decoder output a vector full of zeros. That made sense, because in the decoder's multi-head attention module, `softmax` would NOT like all zero inputs. AH!
2. Later, I also saw that with all non-zero inputs, transformer still output `NaN`. **That's a strong indication** that some parameters were `inf` or `NaN`.

- So I started hunting for `NaN` and `inf` occurences in model parameters. I read that it was not uncommon and gradient clipping was a common practice despite it could slow down the training. **I gave it a try anyways. I still see NaN during training, but they do not jeopardize training anymore.**
- Along the way, I found out that it was quite common in Transformer variants (GPT, BERT) to use the same embedding layer for input and output tokens. This is called "weight tying". This way, the model can learn a better memory efficient model. However, before getting a fully functioning training and validation bench, I'd stick to two separate embedding layers to learn separate input and output embeddings.

### Lesson 3 - Padding Masks Are Important

At this stage, shapes of my data should be good. However, my model's loss was still relatively high. It started to output words (that is, not `<EOS>, <SOS>, <PAD>`), but the training loss was stuck. I realized that I did not add padding mask.

- I added padding mask to help the model not to focus on the padding. This helped the model learn the actual target sentences.
- `nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)` was also helpful.

Side note: On the internet, one might encounter [similar tutorials like this one](https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1). I was inspired by them and built out a general training skeleton. However, they were testing with randomly generated values and simply training for "copying". That's a simpler task than translation. So, **please add the padding masks.**

### Lesson 4 - Teacher Forcing Is Subject To Exposure Bias

The most stark difference I encountered in [tutorials like this one](https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch) and my hands-on training experience is the use of **teacher forcing**. Teacher forcing is to feed the entire ground truth into the transformer, specifically, the decoder, so the model learns the parameters to output target values, when **it sees the entire ground truth**. This is great for speeding up learning, and as a quick way to validate that the model, the data, the general training framework works. However, in validation, we feed the decoder outputs back into the transformer one at a time (a.k.a autoregression). The performance there is **VERY BAD**: the model outputs a bunch of `<EOS>, <SOS>, <PAD>` as if it had not learned anything. This is called "Exposure Bias"

- I added the `look_ahead mask (attn_mask)` in the belief that it could help the model learn from existent timesteps. However, that did not solve this issue.
  - I actually ran into a bunch of `NaN` again. I read that one should use a small negative value like `-1e9` for `attn_mask` instead of `-inf` to avoid underflow

- I read that architecture might be too small. So I tried a larger architecture with 8 encoder layers, 8 heads. The training took forever, the same problem still existed. I think it's wise **to train with small training data + a small architecture, and validate with trained data.**
- At this stage, it seems that mixing in decoder data is very important, though the target value could be noisier. The training loss was more "rugged", but it went steadily down. Progress! 😊

  - Here in training, I could see something like:

    ```text
    prediction: ['SOS', 'SOS', 'i', 'i', 'not', 'working', 'for', 'work', 'EOS', 'PAD', 'PAD'], target: ['SOS', 'i', 'm', 'not', 'working', 'for', 'tom', 'EOS', 'PAD', 'PAD']
    ```

  - In Validation, I see something like:

    ```text
    # For "Eres grande."
    Prediction: ['SOS', 'big', 'EOS', 'you', 're', 'SOS', 'big', 'big', 'you', 'big', 'PAD']
    ```

### Lesson 5: Early Training Termination Upon `<EOS>` Makes Training Harder

The model's performance at this step this is still not ready for the small test data. The model doesn't seem to learn the positioning of `<EOS>`  well.

To try different hypothesis, I was really looking to speed up the training. One thought came to my mind: "why don't we terminate training of a batch, when all output sentences already have an `<EOS>`"? This method made training stuck again.

My theory is that this method could introduce unnatural learning result. So, I'm trying to see if without termination, the model can ultimately learn the correct `<EOS>` position.

### Lesson 6: Regression Test Is Really Helpful For Debugging

In the meantime, I tried setting `teacher-forcing-ratio` to 1 so I always had teacher forcing. This is a "regression test" that helps identify any potential bug in teacher forcing mixing. Strangely, this time I didn't see a successful learning result there. This led me correct a small indexing issue I have there.

### Lesson 7: There Must Be Autoregression In Inference

One thought crossed into my mind was "what about having "one-shot" prediction?" That is, during inference, we read all logits predicted and use them as an output. I had this thought because the output already has all timesteps `[batch_size, time_steps, output_dim]`. Also, a look-ahead mask already omitted `[t+1, MAX_SENTENCE_LENGTH]` terms when calculating attention.

This idea will make inference impossible, because an attention looks at what **is in the output up until the current timestep**. When we start out by feeding `tgt=[<SOS>, ... <PAD>]` into the decoder, the decoder will use this `tgt` as the key, value, and query in its self attention. Only `<SOS>` is a meaningful input, so only the attention for `t=0` is meaningful.

For Pure Teacher Forcing Training, it's Not Necessary, But We are not doing that for the exposure bias.

### Lesson 8: A Larger Batch Size Can Almost Likely Speed Up Training

I had "accumulated batching" on so back-propagation is carried out only when an accumulated batch size is achieved. However, despite that in my various experiments, I still noticed an almost linear relationship between batch_size and training speed. **So, trying as large batch sizes as possible shouldn't be a bad idea**

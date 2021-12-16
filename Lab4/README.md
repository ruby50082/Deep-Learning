# Introduction
- Conditional Sequence-to-Sequence VAE
- In this lab, you need to implement a conditional seq2seq VAE for English tense conversion and generation.
- VAE has been applied to many NLP generation task such as text summarization and paraphrase.
- Specifically, your model should be able to do English tense conversion and text generation.
- For example, when we input the input word ‘access’ with the tense (the condition) ‘simple present’ to the encoder, it will generate a latent vector z.
- Then, we take z with the tense ‘present progressive’ as the input for the decoder and we expect that the output word should be ‘accessing’.
- In addition, we can also manually generate a Gaussian noise vector and feed it with different tenses to the decoder and generate a word those tenses.

![image](https://user-images.githubusercontent.com/39916963/146381200-5b0adff4-f11a-4a46-b77b-2b4f34b1e7f1.png)

- More details can be found in spec.

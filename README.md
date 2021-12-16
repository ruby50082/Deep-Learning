# Deep-Learning

# Lab1
### Introduction
- In this lab, you will need to understand and implement simple neural networks with forwarding pass and backpropagation using two hidden layers.
- Notice that you can only use Numpy and the python standard libraries, any other framework (ex : Tensorflow、PyTorch) is not allowed in this lab.
<img src=https://user-images.githubusercontent.com/39916963/146375325-b86debdd-3678-4026-b37a-3ff50dd4eef4.png width="60%" height="60%">

### Input
![image](https://user-images.githubusercontent.com/39916963/146375418-e613c735-a97c-4533-afdc-0109c56b474e.png)

### Purpose
- To split two kinds of data correctly.
<img src=https://user-images.githubusercontent.com/39916963/146375520-0f759486-b9a9-4a81-8d7b-d2e604ae7d86.png width="40%" height="40%">

---

# Lab2
### Introduction
- In this lab, you will need to implement simple EEG classification models which are EEGNet, DeepConvNet with BCI competition dataset.
- Additionally, you need to try different kinds of activation function including ReLU, Leakly ReLU, ELU.

### Input data
![image](https://user-images.githubusercontent.com/39916963/146378561-f8d48713-e6b7-4114-9650-1e9c36ab1593.png)

---

# Lab3
### Introduction
- In this lab, you will need to analysis diabetic retinopathy (糖尿病所引發視網膜病變) in the following three steps.
- First, you need to write your own custom DataLoader through PyTorch framework.
- Second, you need to classify diabetic retinopathy grading via the ResNet architecture.
- Finally, you have to calculate the confusion matrix to evaluate the performance.
- Loss function: Cross entropy

### Dataset
- Diabetic Retinopathy Detection (kaggle)
- Format: .jpg

<img src="https://user-images.githubusercontent.com/39916963/146379678-f0db5f58-51b2-4935-a825-dfb280caf11e.png" width="50%" height="50%">

---

# Lab4
### Introduction
- Conditional Sequence-to-Sequence VAE
- In this lab, you need to implement a conditional seq2seq VAE for English tense conversion and generation.
- VAE has been applied to many NLP generation task such as text summarization and paraphrase.
- Specifically, your model should be able to do English tense conversion and text generation.
- For example, when we input the input word ‘access’ with the tense (the condition) ‘simple present’ to the encoder, it will generate a latent vector z.
- Then, we take z with the tense ‘present progressive’ as the input for the decoder and we expect that the output word should be ‘accessing’.
- In addition, we can also manually generate a Gaussian noise vector and feed it with different tenses to the decoder and generate a word those tenses.

![image](https://user-images.githubusercontent.com/39916963/146381200-5b0adff4-f11a-4a46-b77b-2b4f34b1e7f1.png)

- More details can be found in spec.

---

# Lab5
### Introduction
- In this lab, you need to implement a conditional GAN to generate synthetic images according to multi-label conditions.
- To achieve higher generation capacity, especially in computer vision, generative adversarial network (GAN) is proposed and has been widely applied on style transfer and image synthesis.
- In this lab, given a specific condition, your model should generate the corresponding synthetic images.
- For example, given “red cube” and “blue cylinder”, your model should generate the synthetic images with red cube and blue cylinder and meanwhile, input your generated images to a pre-trained classifier for evaluation.

![image](https://user-images.githubusercontent.com/39916963/146383006-6ab7c3bf-5f8e-4bdb-882d-3aa35b6ade69.png)

---

# Lab6
### Introduction
- Deep Q-Network and Deep Deterministic Policy Gradient
- In this lab, you will learn and implement two deep reinforcement algorithms by completing the following two tasks: 
  (1) solve LunarLander-v2 using deep Q-network (DQN), and 
  (2) solve LunarLanderContinuous-v2 using deep deterministic policy gradient (DDPG).

# Introduction
- We would like to build an image extension model, which has broad applications in image editing, computational photography and computer graphics.
- Comparing with inpainting model "Context Encoder" as baseline, we implement conditioning in a GAN to train an image extension model.
- We add a pretrained model "Inception_v3" to select image features and compare with the discriminator results, in order to improve the model’s performance.
- Experiment results show that models trained with conditioning have better performance than baseline.

# CAPTCHA-verifier
Welcome to CAPTCHA Verifier repository! This open-source project is your one-stop solution for leveraging the power of deep learning and transformer-based models to tackle the challenging task of CAPTCHA recognition and verification.

## Overview
CAPTCHAs (Completely Automated Public Turing tests to tell Computers and Humans Apart) are widely used to protect websites and online services from automated bots. This repository provides a comprehensive set of tools and code to fine-tune state-of-the-art transformer models for the task of recognizing and verifying CAPTCHAs which can be used for automated web scrapping as an example. Whether you're interested in improving the security of your online platform or conducting research in the field of computer vision and deep learning, this repository has you covered.

## Approach

In this notebook, we will explore the fine-tuning of a pre-trained TrOCR model on two distinct datasets: the [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) and the [CAPTCHA dataset](https://www.kaggle.com/datasets/parsasam/captcha-dataset) from Kaggle.

### Data usage

#### 1. IAM Handwriting Database
The [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) is a comprehensive collection of annotated images containing handwritten text. This dataset provides a rich source of diverse handwriting samples, making it ideal for training models to recognize and transcribe handwritten text in various contexts.

#### 2. CAPTCHA Dataset
The [CAPTCHA dataset](https://www.kaggle.com/datasets/parsasam/captcha-dataset) is an extensive repository comprising over 113,000 colorful 5-character images. CAPTCHAs, which stand for "Completely Automated Public Turing tests to tell Computers and Humans Apart," are commonly used to differentiate between human users and automated bots on websites and online services. This dataset offers a valuable resource for training and evaluating models designed to tackle CAPTCHA recognition tasks.

### Model used

In this notebook, we will employ the `VisionEncoderDecoderModel` class to fine-tune a pre-trained TrOCR model. TrOCR represents an encoder-decoder architecture, with its encoder weights initialized from a pre-trained BEiT model and its decoder weights initialized from a pre-trained RoBERTa model. The cross-attention layer's weights were initially random, and the model underwent further pre-training on millions of partially synthetic annotated images of handwritten text.

This figure gives a good overview of the model (from the original paper):

![TrOCR](https://miro.medium.com/v2/resize:fit:786/format:webp/1*yCoAnJAjspTdltTkMYQ0cA.png)

## References

- TrOCR Paper: [Read Paper](https://arxiv.org/abs/2109.10282)
- TrOCR Documentation: [View Documentation](https://huggingface.co/transformers/master/model_doc/trocr.html)

To gain a deeper understanding of warm-starting encoder-decoder models, which is the approach employed by the TrOCR authors, you can refer to Patrick's insightful [blog post](https://huggingface.co/blog/warm-starting-encoder-decoder).

We will carry out the fine-tuning process using native PyTorch.

## Set-up Environment

To begin, let's install the necessary libraries:

- Transformers (for the TrOCR model)
- Datasets & Jiwer (for evaluation metrics)

While working through this notebook, we will not utilize HuggingFace Datasets for data preprocessing. Instead, we will create a conventional PyTorch Dataset.

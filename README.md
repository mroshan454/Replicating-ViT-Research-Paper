# Replicating-ViT-Research-Paper & Plant Leaf Disease Detection using Vision Transformers 🍃🍂🧑‍🔬📝💻

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mroshan454/Replicating-ViT-Research-Paper/blob/main/Replicating_The_ViT_Paper_from_Scratch.ipynb)

# 1. Project Overview 📝

This project replicates the **Vision Transformer (ViT)** Architecture from the groundbreaking 2020 research paper , "An Image is Worth 16x16 Words" and applies it to plant disease classification.

I replicated a ViT from scratch using PyTorch including patch embedding, transformer encoder layers, and classification head — to deeply understand its mechanics. 

To test the architecture , I trained it on a 3-Class subset of Plant Village Dataset(Tomato Early Blight, Tomato Healthy, Tomato Late Blight). 
Due to the limited dataset size (~500 train / 150 test images per class), the scratch model underfit — mirroring the ViT paper’s emphasis on the need for large-scale pretraining.

Then I fine-tuned a Pre-trained ViT(VIT B-16) on the same dataset , which achieved a strong performance and correctly classified unseen plant disease image I found online.
This confirmed the transfer learning's advantage of pre-trained transformers in low-computation and low-data constraints.

The learnings from this replication will be applied in my upcoming mobile plant disease detection app using a lightweight model for 8+ classes.
 

# 2. Motivation 🏃‍♂️‍➡️📈📖

The ViT paper was a landmark in Computer Vision , that proved Transformer architecture which initially used for NLP can outperform CNNs if you give enough data.

However in real-world domains like Agriculture we don't have millions of labelled images to train from scratch.

By replicating the ViT from scratch and fine-tune a pretrained ViT from scratch on a small-scale agriculture dataset , I aimed for:

- Understand every components , equations and math behind the VIT architecture by building it myself.
- Compare the custom ViT I built vs Transfer Learning Performance.
- Highlight The data requirements of ViT for High Accuracy.
- Prepare for Deploying the model trained using Transfer Learning for Real World use.

This project blends both research paper replication with practical application development, demostrating both theoretical depth and deployement-focused thinking.

# 3. Dataset 📊📚

The dataset is derived from **PlantVillage** Dataset,  a large-scale collection of plant leaf images used for plant disease detection research.

### Original Dataset Details:
- Classes: 39 (various crop species, multiple disease types, and healthy leaves, plus a “background without leaves” class).
- Total Images: 61,486.
- Augmentation Techniques Applied in Original Dataset:
  * Horizontal & vertical flipping
  * Gamma correction
  * Noise injection
  * PCA color augmentation
  * Rotation
  * Scaling
#### Examples of Classes in the Original Dataset:
- Apple (Scab, Black Rot, Cedar Apple Rust, Healthy)
- Pepper Bell (Bacterial Spot, Healthy)
- Potato (Early Blight, Healthy, Late Blight)
- Tomato (Bacterial Spot, Early Blight, Healthy, Late Blight, and more)
- Plus several other fruit, vegetable, and background categories.
### Classes Used for This Project’s ViT Replication:
For the purpose of replicating the Vision Transformer paper under limited compute and training time constraints, I selected 3 classes from the original dataset:
 - Tomato Early Blight
 - Tomato Healthy
 - Tomato Late Blight
#### Sampling Strategy for This Experiment:
- Training Set: 500 images per class.
- Test Set: 150 images per class.
Balanced sampling ensures equal representation of all classes during training and evaluation.
### 📌 Why Reduce the Dataset?
The primary goal was to replicate the ViT paper’s architecture from scratch and compare it against a pretrained ViT model under a controlled, smaller-scale setup.
Using fewer classes and fewer images allowed faster iteration and debugging without requiring massive GPU resources.






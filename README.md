# Replicating-ViT-Research-Paper: Plant Leaf Disease Detection using Vision Transformers 🍃🍂🧑‍🔬📝💻

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mroshan454/Replicating-ViT-Research-Paper/blob/main/Replicating_The_ViT_Paper_from_Scratch.ipynb)


This is my attempt to replicate the Vision Transformer Paper **"AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE"** from scratch using *PyTorch*.

For this project I'm using ViT to classify Plant Leaf Diseases for Tomato 🍅 , BellPepper🫑 and Potato🥔. This project is heavily inspired by the Mr. Daniel Bourke's Online Pytorch Course, which helped me in providing foundation and approach for implementing ViT in a modular and scalable way.

**📊 Project Goals**

- ✅ Replicate the Vision Transformer architecture
- ✅ Train a ViT model from scratch
- ✅ Fine-tune a pretrained ViT model
- ✅ Classify Plant Leaf Disease using Healthy and Affected Leaf Images 
- ✅ Compare Performance of ViT Built from scratch vs. Pretrained ViT 


**🗂️Repository Structure**

* data/
* going_modular/   #Contains modular functions in .py script file .
    * |---/data_setup.py
    * |---/engine.py
    * |---/helper_function.py
    * |---/image_sampling.py
    * |---/utils.py
* images/          #Contains images explaining equations and figures in ViT Paper , and also images for custom predictions.
* Modular_Functions_For_Pytorch.ipynb          #Notebook explaining the modular functions for pytorch workflow.
* Replicating_The_ViT_Paper_from_Scratch.ipynb      #The main ViT Paper Replication Notebook.


## 📖 Understanding Vision Transformers

### 🧠 What is Vision Transformer (ViT)?

ViT treats image patches like tokens in NLP and applies Transformer encoders to classify images.

ViT processes an image by:
1. Splitting it into patches
2. Embedding each patch
3. Passing through Transformer blocks
4. Using the [CLS] token for classification

### Why Vision Transfomers?
- Traditional Convolutional Neural Networks(CNNS) dominates the computer vision due to their ability to capture local patterns. However, CNNs have limitations in modeling long-range dependencies and lack flexibility. Vision Transformers leverage self-attention to overcome this by treating images like sequences, similar to how NLP models treat sentences.
  
### Transformers in NLP
- Transformers were originally introduced in the famous paper `Attention is All you Need(2017)` and was originally designed for NLP tasks. They use self-attention to weigh relationships between tokens in a sequence.


## Project Implementation 💻📐

### From Words -> Patches to Image -> Patches 
- In the original transformer architecture they break down words into patch embeddings and create sequence and then pass it through the transformer layers , Similarly in ViT we break down a 2-Dimensional picture into patches of equal size (16x16 in this case) and combine them to form a linear sequence and then pass it into transformer layers to classify them. 


I recreated key visuals from the ViT paper to understand how it works under the hood:

- **Figure 1 Explaining the ViT architecture**
  
 ![ViT architecture](images/1.png)
### Equation 1 - Splitting the image into patches and flattening them
**Equation 1 Explaination🧠**
  This equation turns the image into patch embeddings and add an extra learnable token and add position embeddings

  Before the input image can be passed through the Vision Transformer (ViT), it needs to be converted into a sequence format, similar to how words are processed in NLP.

- The input image is split into fixed-size patches (e.g., 16x16).
- Each patch is flattened into a 1D vector.
- A special learnable [CLS] token is prepended to represent the entire image.
- Positional embeddings are added to retain information about the order of patches.

This entire sequence becomes the model's input, just like tokens in a sentence for a language model.
 
  ![Equation 1](images/2.png)
  **Turning Equation 1 into Usable Code**
  ![Equation 1 to Code](images/Equation1_to_Code.png)
### Equation 2&3
- **Equation 2 Explaination🧠**
After forming the input token sequence from Equation 1, we feed it into a standard Transformer encoder block. The first step in that block is the Multi-Head Self-Attention (MSA) mechanism.
- Each token attends to all other tokens, including itself.
- This helps the model learn relationships between different image patches.
- MSA allows the model to jointly focus on different representation subspaces.
- This is followed by Layer Normalization and a skip connection (residual connection).
  ![Equation 2](images/3.png)
  **Turning Equation 2 into Usable Code**
  ![Equation 2 to Code](images/Equation2_to_Code.png)
- **Equation 3 Explaination🧠**
- After applying MSA, the result is passed through a position-wise feedforward network — essentially a *2-layer MLP* with a *non-linear activation* (usually GELU or ReLU).
- This helps transform the representations in a more complex way.
- Again, we apply Layer Normalization and a residual connection.
  ![Equation 3](images/4.png)
  **Turning Equation 3 into Usable Code**
  ![Equation 3 to Code](images/Equation3_toCode.png)
### Equation 4
- **Equation 4 Explaination🧠**
- At the very beginning (Eq. 1), we prepended a special [class] token to the patch embeddings.
- After passing through L Transformer layers, we take the output corresponding to the [class] token (i.e.,z_0^L) as the final image representation.
- This vector now goes through a classification head:
- During pretraining: it's an MLP with one hidden layer.
- During fine-tuning: it's a single linear layer.
- The result is the final logits used for classification (e.g., happy/sad/angry).

![Equation 4](images/5.png)

### Putting All together to Form Entire ViT Architecture to Usable PyTorch Code
![All together](images/Entire_ViT_in_Code.png)
![All together](images/Whole_ViT_in_PyTorch_Code.png)


















# Replicating-ViT-Research-Paper: Emotion Detection from Facial Images using Vision Transformers🎭📝💻

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/mroshan454/Replicating-ViT-Research-Paper/blob/main/Replicating_The_ViT_Paper_from_Scratch.ipynb)


This is my attempt to replicate the Vision Transformer Paper **"AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE"** from scratch using *PyTorch*.

For this project I'm using ViT to classify human emotions (Happy , Angry , Sad). This project is heavily inspired by the Mr. Daniel Bourke's Online Pytorch Course, which helped me in providing foundation and approach for implementing ViT in a modular and scalable way.

**📊 Project Goals**

- ✅ Replicate the Vision Transformer architecture
- ✅ Train a ViT model from scratch
- ✅ Fine-tune a pretrained ViT model
- ✅ Classify human emotions using facial expression images
- ✅ Deploy the final model (coming soon)


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

I recreated key visuals from the ViT paper to understand how it works under the hood:

- **Figure 1 Explaining the ViT architecture**
  
 ![ViT architecture](images/1.png)
- **Equation 1**
  This equation turns the image into patch embeddings and add an extra learnable token and add position embeddings
  ![Equation 1](images/2.png)
- **Equation 2&3**
  The Transformer contains the alternating layer of MSA and MLP Blocks , where LayerNorm(LN) is added before and residual connection is added after.
  ![Equation 2](images/3.png)
  ![Equation 3](images/4.png)
- **Position Embeddings**









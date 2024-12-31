![image](https://github.com/user-attachments/assets/493b6617-413e-410e-98a8-e054f64a2b84)


<h1 align="center">🖼️ Generative Adversarial Network (GAN) for Face Generation 🖼️</h1>

<p align="center">
<strong>A Deep Learning project utilizing GANs to generate realistic faces with TensorFlow and Keras.</strong>
</p>

<p align="center">
<a href="https://github.com/Aman-Devs/Generative-Adversarial-Network-GAN">
<img src="https://img.shields.io/github/license/its-amann/Face-Generation-with-Deep-Convolutional-Generative-Neural-Network.svg" alt="License">
</a>
<a href="https://github.com/Aman-Devs/Generative-Adversarial-Network-GAN/issues">
<img src="https://img.shields.io/github/issues/its-amann/Face-Generation-with-Deep-Convolutional-Generative-Neural-Network.svg" alt="Issues">
</a>
<a href="https://github.com/Aman-Devs/Generative-Adversarial-Network-GAN/stargazers">
<img src="https://img.shields.io/github/stars/its-amann/Face-Generation-with-Deep-Convolutional-Generative-Neural-Network.svg" alt="Stars">
</a>
</p>

🚀 Overview

Welcome to the Generative Adversarial Network (GAN) Project, a sophisticated implementation of a GAN that generates realistic faces. This project is designed to explore the power of adversarial networks in creating synthetic data that closely resembles real-world examples. It's a deep dive into how generator and discriminator networks work together in a competitive yet collaborative environment to produce lifelike imagery. This repository contains the necessary code and setup to understand, train, and use your own GAN model for generating faces.

✨ Key Highlights:

Adversarial Training: Leverages the core principles of GANs, where two neural networks (generator and discriminator) compete to improve each other.

High-Quality Output: Generates visually compelling and realistic face images using a complex architecture.

TensorFlow & Keras: Built with cutting-edge deep learning frameworks for optimal performance and flexibility.

Customizable: Designed to be easily modified and extended to fit other data generation tasks.

End-to-End Project: Provides a complete pipeline from data loading to image generation, ideal for learning and experimentation.

![image](https://github.com/user-attachments/assets/3c09337f-2280-4a09-ad91-d0972ba73e34)

<h3 align="center">Generative Adversarial Network Workflow</h3>

🛠 Features

Data Download & Preprocessing: Automatically downloads the CelebA dataset and prepares it for training, including resizing and normalization.

Custom Layers and Models: Utilizes TensorFlow and Keras to build a highly tailored GAN architecture.

Dynamic Image Generation: The model is trained to create new, realistic face images that are different from the training dataset.

Visual Progress Tracking: Includes a callback function to save generated images at the end of each epoch, making it easy to monitor the progress of the GAN during training.

Loss Visualization: Plots the generator and discriminator losses, allowing for an in-depth analysis of the training process.

Flexible Parameters: Allows the user to modify important parameters such as the learning rate, latent dimension and number of epochs.

Modular Code: The codebase is neatly organized into separate functions and classes for easy comprehension and maintenance.

TensorBoard Integration: Enables the use of TensorBoard for enhanced visualization of training parameters and network graph.

Model Checkpoints: Implements saving and loading model weights for each training session, ensuring progress is not lost and allowing resuming training.

### 📸 Screenshots
![image](https://github.com/user-attachments/assets/ff16c41b-20ae-4809-b587-15fa31148d26)

<h3 align="center">Generated faces after 100 Epochs of training</h3>

![image](https://github.com/user-attachments/assets/9fe84f75-0f28-47ae-9091-d197bb37fe39)

<h3 align="center">Generated faces after 200 Epochs of training</h3>

![image](https://github.com/user-attachments/assets/d295284e-824f-4995-b203-4cfeea113e03)

<h3 align="center">GAN loss visualization</h3>


## 🔧 Installation

### Prerequisites

-   **Python 3.x** installed on your machine. [Download Python](https://www.python.org/downloads/)
-   **TensorFlow 2.x** installed. Install using pip:

    ```bash
    pip install tensorflow
    ```
-   **Keras** installed. Keras is usually installed with Tensorflow but can also be installed with pip:
    ```bash
    pip install keras
    ```
-   **NumPy** library for Python. Install using pip:

    ```bash
    pip install numpy
    ```
-   **Matplotlib** library for plotting. Install using pip:
    ```bash
    pip install matplotlib
    ```
- **Kaggle API key:** To download the CelebA dataset directly using Kaggle CLI, you need to have your Kaggle API key set up. Follow the steps from the "Download Data" section.

### Steps

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/Aman-Devs/Generative-Adversarial-Network-GAN.git
    cd Generative-Adversarial-Network-GAN
    ```

2.  **Set up Kaggle API Key:**
    - First you need to get your kaggle api key from the kaggle website in your profile under "Account".
    - Then you need to replace "kaggle.json" with the name of the downloaded file in the code below.

    ```bash
    pip install -q kaggle
    mkdir ~/.kaggle
    cp kaggle.json ~/.kaggle/
    chmod 600 /root/.kaggle/kaggle.json
    ```

3.  **Install Libraries:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application:**

    ```bash
    python gan.py
    ```

---

## 💻 Usage

Upon running the application, the training process for the GAN will start. The images generated during each epoch will be saved in the "generated" directory, and the loss graph will be displayed after the training is completed.

### Steps:

1.  **Run `gan.py`:** Execute the main script to start the training process:
    ```bash
        python gan.py
    ```
2.  **Monitor Training:** Observe the training progress directly from the console, or by visualizing the training metrics using TensorBoard. The generated images at the end of every epoch will provide a visual indication of the training progression.
3.  **Generated Images:** The generated images during each epoch are saved in the `generated/` folder. These images can be used to assess the training quality.
4.  **Loss Plot:** A graph containing the loss values of the discriminator and the generator will be displayed to visualize their evolution during training.
5. **Model Checkpoints:** The weights for the generator and discriminator models will be saved at the end of each epoch in the `model_checkpoints` directory

---

## ⚙️ Codebase Overview

### Main Modules

1.  `gan.py`: The main script that orchestrates the GAN training process. It includes data loading, model definition, and training loops, making it the central point for running the application.

2.  **Model Definition:** The code includes classes and functions for defining the generator and discriminator models:
    -   The generator uses a series of `Conv2DTranspose` layers with Batch Normalization and Leaky ReLU activation to upscale the latent space into an image. This architecture is responsible for creating synthetic images based on random noise.
    -   The discriminator employs `Conv2D` layers with Batch Normalization and Leaky ReLU to downsample the images. This is designed to assess the authenticity of the generated images.

3.  **Training Loop:**
    -  Uses a customized training loop with `train_step` method to ensure proper training of GAN.
     - Implements a separate optimizers for the discriminator and the generator.
    -   Uses the binary cross-entropy loss function for the adversarial training.

4.  **Data Loading:**
    -   Automatically downloads the CelebA dataset from Kaggle.
    -  Uses `tf.data.Dataset` to efficiently load and preprocess images.

5.  **Callbacks:**
    - Includes a custom `ShowImage` callback to monitor training progress by saving the generated images at the end of each epoch. This provides a visual aid to evaluate how well the generator is working.
    - Implements a `ModelCheckpoint` callback that saves the model weights after every epoch. This ensures that the progress is not lost during training.

6. **Loss Visualization:**
    - Includes a section that displays the loss graphs for both the generator and the discriminator.

### Key Design Choices

-   **Leaky ReLU:** The choice of Leaky ReLU over traditional ReLU is to prevent the issue of vanishing gradients, which is common in GANs. This promotes a more stable and faster learning process.
-   **Batch Normalization:** Batch Normalization is used to normalize layer inputs, which leads to faster convergence and more stable training.
-   **Adam Optimizer:** The Adam optimizer is selected due to its adaptive learning rate characteristics, which helps with the training stability of GANs.
-   **Convolutional Layers:** Convolutional layers are used to capture spatial hierarchies within the image data, allowing the model to learn complex features effectively.
-   **Transposed Convolutions:** Transposed Convolutions are used in the generator to upscale the image while learning the features needed for generating new images

---

## 🤝 Contributing

We welcome contributions to enhance this GAN project! Here's how you can help:

1.  **Fork the Project**
2.  **Create your Feature Branch:** `git checkout -b feature/AmazingFeature`
3.  **Commit your Changes:** `git commit -m 'Add some AmazingFeature'`
4.  **Push to the Branch:** `git push origin feature/AmazingFeature`
5.  **Open a Pull Request**

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🙏 Acknowledgments

-   This project is inspired by the original GAN paper by Ian Goodfellow et al.
-   The CelebA dataset is used for training the model and can be obtained from Kaggle.
-   Tools used:
    -   TensorFlow & Keras for model creation and training.
    -   Matplotlib for data visualization.
    -   Numpy for numerical computation

<p align="center">
  Made with ❤️ by Aman
</p>


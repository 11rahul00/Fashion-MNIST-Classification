# Fashion-MNIST-Classification
---
# Fashion MNIST Classification using Transfer Learning with VGG16

This project demonstrates the power of transfer learning for image classification using PyTorch. A pre-trained VGG16 model, originally trained on the large-scale ImageNet dataset, is adapted and fine-tuned to classify images from the much simpler Fashion MNIST dataset.

The core idea is to leverage the powerful feature extraction capabilities of a deep network without having to train it from scratch, resulting in faster training times and high accuracy.

***

## Key Features

* **Transfer Learning**: Implements a classic transfer learning workflow using PyTorch and `torchvision.models`.
* **Pre-trained VGG16**: Utilizes the VGG16 architecture to extract high-level features from images.
* **Feature Freezing**: The convolutional base of the VGG16 model is frozen, and only the final classification layers are trained.
* **Custom Classifier**: The original VGG16 classifier is replaced with a new, custom-built dense network tailored for the 10 classes of the Fashion MNIST dataset.
* **Data Preprocessing**: Includes a robust data pipeline using a custom PyTorch `Dataset` that handles image reshaping, grayscale to 3-channel conversion, and standard transformations.

***

## Methodology

The project follows these key steps:

### 1. Data Loading & Preparation
* The dataset is loaded from a CSV file into a Pandas DataFrame.
* The data is split into training and testing sets using Scikit-learn.

### 2. Custom PyTorch Dataset
* A custom `Dataset` class is created to process the data on the fly.
* Each 784-pixel row is reshaped into a 28x28 image.
* The single-channel grayscale image is converted into a 3-channel image by stacking the channel three times. This is a crucial step to make the data compatible with the VGG16 model.
* Standard `torchvision` transforms are applied to resize the image to 224x224 and normalize it.

### 3. Model Adaptation (Transfer Learning)
* The pre-trained VGG16 model is loaded from `torchvision.models`.
* **Feature Extractor is Frozen**: The weights of all convolutional layers (`vgg16.features`) are frozen by setting `param.requires_grad = False`.
* **Classifier is Replaced**: The final fully connected layers (`vgg16.classifier`) are replaced with a new `nn.Sequential` block designed for the 10 classes of Fashion MNIST.

### 4. Training
* The `Adam` optimizer is configured to update only the weights of the new classifier.
* The model is trained using `CrossEntropyLoss` as the criterion on a GPU for efficiency.

***

## Results

The training loop tracks the average loss for each epoch. The final model is then evaluated on the test set to calculate its classification accuracy. By using transfer learning, this model is expected to achieve high accuracy quickly, demonstrating the effectiveness of leveraging pre-trained weights for new tasks.

* **Final Test Accuracy:** 98%

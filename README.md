# CIFAR-10 Object Recognition Project

## Overview

This project presents a comprehensive exploration and evaluation of computer vision algorithms using the CIFAR-10 dataset. It compares the performance of Convolutional Neural Networks (CNNs) and traditional computer vision techniques like SIFT.

## Dataset

**CIFAR-10**:
- Comprises 60,000 32x32 color images in 10 different classes, with 6,000 images per class.
- Divided into 50,000 training images and 10,000 test images.

## Methods

### 1. Convolutional Neural Network (CNN)

#### Architecture

- **Input Layer**: (32, 32, 3)
- **Convolutional Layers**: 32, 64, 128 filters with ReLU activation and Batch Normalization.
- **Pooling Layers**: MaxPooling2D.
- **Dropout**: Applied progressively (0.2 to 0.5) to prevent overfitting.
- **Fully Connected Layers**: Flatten followed by Dense layers with Softmax activation for classification.

#### Hyperparameters

- **Optimizer**: Adam
- **Batch Size**: 64
- **Epochs**: 100
- **Data Augmentation**: Rotation, shift, shear, zoom, and flip.

#### Comparison of CNN Architectures

| Test No. | Batch Size | Epochs | Data Augmentation                            | Parameters                    | Training Accuracy | Test Accuracy | Notes                                                                                           |
|----------|------------|--------|----------------------------------------------|-------------------------------|-------------------|---------------|-------------------------------------------------------------------------------------------------|
| 1        | 128        | 10     | None                                         | Basic CNN architecture        | 65.9%             | 65.82%        | Moderate performance                                                                            |
| 2        | 128        | 20     | Heavy (rotation, shift, zoom, flip)          | Deeper CNN architecture       | 61.27%            | 61.08%        | Worse results with fluctuating accuracies                                                       |
| 2        | 128        | 20     | Moderate (shift, flip)                       | Deeper CNN architecture       | 67.56%            | 67.05%        | Improved accuracy with reduced augmentation                                                     |
| 3        | 64         | 40     | Batch Normalization, Moderate augmentation   | Increased depth and training  | 82.46%            | 82.3%         | Better performance with balanced augmentation                                                   |
| 3        | 32         | 50     | Batch Normalization, Moderate augmentation   | Increased depth and training  | 82.0%             | 83.24%        | Slight improvement with reduced batch size                                                      |
| 3        | 64         | 100    | Batch Normalization, Moderate augmentation   | Increased depth and training  | 86.31%            | 88.01%        | Best test accuracy with balanced complexity and regularization                                  |
| 4        | 128        | 50     | Batch Normalization, Moderate augmentation   | Additional convolution layers | 85.4%             | 85.01%        | Performance saturated with additional layers                                                    |
| 5        | 64         | 100    | Batch Normalization, Moderate augmentation   | Incremented convolution stage | 86.19%            | 85.8%         | Best training accuracy but similar test accuracy to Test 3, indicating mild overfitting |

#### Results

- Achieved highest test accuracy of **88.01%** with moderate data augmentation and Batch Normalization.
- The selected architecture (Test 3) balances model complexity and performance, demonstrating high accuracy and excellent class-wise metrics.

### 2. Traditional Computer Vision Techniques

#### Feature Detection

- **Harris Corner Detector**
- **Laplacian of Gaussian (LoG)**
- **Difference of Gaussians (DoG)**

#### Feature Descriptor

- **SIFT (Scale-Invariant Feature Transform)**

#### Classification

- **k-NN (k=5)**
- **SVM** (yielded higher accuracy compared to k-NN)

#### Results

- Best accuracy achieved with SIFT combined with DoG and SVM: **36.32%**

### Comparison

- **Deep Learning (CNN)**:
  - Automatically extracts hierarchical features.
  - Achieved significantly higher accuracy (**88.01%**) compared to traditional methods.
  - Requires substantial computational resources.

- **Traditional Methods**:
  - Manually engineered features.
  - Lower accuracy (best **36.32%** with SVM).
  - More transparent and require fewer resources.

## Future Work

- Experiment with different combinations of augmentation techniques.
- Test deeper architectures like ResNet and DenseNet.
- Implement k-fold cross-validation to ensure robust performance.

## Conclusion

The project demonstrates that CNNs significantly outperform traditional methods in object recognition tasks on the CIFAR-10 dataset, highlighting the advantages of deep learning models in learning complex patterns and generalizing data.

## References

1. Schilling, F. (2016). The Effect of Batch Normalization on Deep Convolutional Neural Networks (Dissertation).
2. https://arxiv.org/pdf/1412.6980
3. Harris, C.G. and Stephens, M.J. (1988). A Combined Corner and Edge Detector. Alvey Vision Conference.
4. https://scikit-learn.org/stable/modules/svm.html
5. Szeliski, R. (2010). Computer Vision: Algorithms and Applications. Springer-Verlag.
6. Correll, N., et al. (2022). Introduction to Autonomous Robots. MIT Press.
7. Krizhevsky, A., et al. (2012). ImageNet Classification with Deep Convolutional Neural Networks. Advances in Neural Information Processing Systems.


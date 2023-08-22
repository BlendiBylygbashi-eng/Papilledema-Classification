# Papilledema Classification via Deep Learning

Welcome to the Papilledema Classification repository! Here, we will be using Convolutional Neural Networks (CNNs) to diagnose Papilledema from images of the back of the eye. The goal is to use the power of deep learning to accurately and non-invasively detect Papilledema (swollen optic disc) which is an indicator of intra-cranial hypertemsion.

Through this repository, you'll find a thorough breakdown of each segment of our pipeline, ranging from data preparation and augmentation to model creation, training, and validation. We aim to provide a comprehensive guide, enabling you to understand the methodologies employed and to adapt or extend them as needed.

---

## Table of Contents

1. [Data Processing](./data-processing)
2. [Model Creation and Training](./training)
3. [Model Evaluation](./evaluation)
4. [Predictions on New Data](./predictions)
5. [Utilities and Helper Functions](./utils)
6. [Required Libraries](./required-imports.py)

---

## Data Processing

The success of a deep learning model heavily depends on the quality and structure of the data it's trained on. In this project, we've emphasized rigorous data preprocessing to ensure optimal performance.

### Highlights:

- **Data Source**: Our first dataset has been sourced from [Kim, U. (2018, August 1). Machine learning for Pseudopapilledema.](https://doi.org/10.17605/OSF.IO/2W5CE). Our second dataset was sourced from [The William F. Hoyt Neuro-Ophthalmology Collection](https://novel.utah.edu/Hoyt/collection.php). We appreciate and credit the contributors for making this valuable data available.

- **Data Augmentation**: To increase the robustness of our model and counteract overfitting, we've used data augmentation techniques such as simple rotations, scaling and translation. This artificially expands the dataset by introducing minor alterations to the existing images, with the aim of increasing the extent to which the model is able to generalize to new data.

- **Normalization**: All images have been normalized to ensure consistent pixel value ranges, facilitating better and faster convergence during training.

- **Data Split**: The dataset is partitioned into training, validation, and test sets. This separation ensures that our model is evaluated on unseen data, giving a genuine measure of its performance.

- **Handling Imbalanced Data**: Special attention has been paid to handle class imbalances, ensuring that each category has a fair representation in the training process.

For a more detailed walkthrough of our data preprocessing steps and to view the code, check out the [Data Processing directory](./data-processing).

---



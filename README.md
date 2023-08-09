


Tumor Classification using EfficientNetB0


Overview
This GitHub repository contains the code and resources for a tumor classification project using the EfficientNetB0 architecture. The objective of this project is to accurately classify tumors into one of the four specified classes. The dataset used for training and evaluation was obtained from Kaggle and consists of labeled images representing different tumor types.

Project Highlights
Utilizes the state-of-the-art EfficientNetB0 architecture for image classification tasks.
Implements data preprocessing, augmentation, and model training pipelines.
Achieves classification accuracy and precision across four tumor classes.
Provides Jupyter notebooks showcasing the step-by-step process from data preparation to model evaluation.
Dataset
The dataset used in this project is sourced from Kaggle and consists of a collection of images representing various tumor types. The data has been split into four classes, each corresponding to a specific tumor category. The dataset includes both training and testing subsets, enabling robust model evaluation.

Dataset Link: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

Code Structure:
data_preprocessing.ipynb: Notebook for data preprocessing, including image resizing.
efficientnetb0_model.ipynb: Notebook for building, training, and evaluating the EfficientNetB0 model.
inference.ipynb: Notebook for making predictions using the trained model.

Requirements:
Python 3.x
TensorFlow 2.x
Additional packages listed in requirements.txt
Getting Started
Clone this repository: git clone https://github.com/your-username/tumor-classification.git
Install required packages: pip install -r requirements.txt
Follow the notebooks (data_preprocessing.ipynb, efficientnetb0_model.ipynb) to preprocess the data and train the model.
Use the inference.ipynb notebook for making predictions on new images.

Acknowledgments:
EfficientNet: Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ArXiv:1905.11946

License:
This project is licensed under the MIT License.


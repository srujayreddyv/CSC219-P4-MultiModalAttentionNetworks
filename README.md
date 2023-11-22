# CSC219 Project 4: Multi-Modal Attention Networks for Fake News Detection

## Overview
In the CSC219 Machine Learning course, we embarked on an ambitious project to address the pressing issue of fake news detection. Our approach leverages a multi-modal framework, combining both text and image data, and employs advanced attention mechanisms to enhance detection accuracy.

## Objective
The primary goal is to improve the detection of fake news by learning and exploiting the correlations between visual and textual content through various attention mechanisms.

## Dataset
The project utilizes the comprehensive Fakeddit dataset, comprising over 1 million multimodal samples that include text, images, metadata, and comments. 

**Link for Original Dataset**: [Fakeddit Dataset](https://github.com/entitize/Fakeddit)

## Models Implemented
1. **Baseline Model**: This foundational model uses convolutional layers for image processing and an LSTM for text analysis.
2. **Self-Attention Model**: Building upon the baseline, this model incorporates self-attention mechanisms for enhanced feature extraction.
3. **Co-Attention Model**: A sophisticated approach that uses co-attention mechanisms to process text and image data in tandem.
4. **Cross-Attention Model**: This model employs cross-attention mechanisms to better understand the interplay between text and image modalities.
5. **ResNet Model**: An advanced variant that utilizes a pretrained ResNet50 for image processing, coupled with an LSTM for text analysis.

## Usage
Run the `CSC219TeamProject_4.ipynb` Jupyter notebook to train and evaluate the models. This notebook contains all necessary code for model implementation, training, and evaluation.

## Key Features
- Advanced machine learning techniques tailored for fake news detection.
- Exploration of different attention mechanisms to assess their impact on model performance.
- Evaluation metrics include Precision, Recall, and F1 Score, providing a comprehensive view of model efficacy.

## Results
Our models have shown a notable improvement in detecting fake news. The incorporation of attention mechanisms, especially co-attention, has been particularly effective in enhancing model performance.

## Future Work
- Hyperparameter tuning and optimization to further refine model accuracy.
- Investigating additional techniques such as transfer learning and BERT embeddings for more sophisticated feature extraction.
- Conducting a deeper analysis to understand the specific impacts of various attention mechanisms on the model's ability to detect fake news.

## Acknowledgements
Special thanks to the instructors and fellow students of the CSC219 Machine Learning course for their invaluable support and insights throughout this project.

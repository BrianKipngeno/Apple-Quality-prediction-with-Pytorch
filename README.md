# Apple-Quality-prediction-with-Pytorch

This project aims to build a machine-learning model that classifies apples as either "Good" or "Bad" based on their features. The model is trained using PyTorch and achieves an accuracy of 87.67%. It can be further improved with techniques like hyperparameter tuning and cross-validation.

**Table of Contents**

- Business Understanding
- Data Preparation
- Model Building
- Training
- Evaluation
- Results and Findings
- Future Improvements
  
**Business Understanding**

The objective of this project is to help stakeholders easily classify apples as "Good" or "Bad" based on a set of input features (e.g., size, weight, sweetness). This binary classification task can optimize quality control in the apple supply chain, improving efficiency and reducing waste.

**Data Preparation**

The data is preprocessed by splitting it into training and test sets. Key preprocessing steps include:

- Normalization of input features
- Conversion of data into PyTorch tensors
- Use of DataLoader for efficient batch processing
- Model Building
  
The model is a simple neural network built using PyTorch. It consists of the following:

- Input layer: Based on the number of features in the dataset
- Two hidden layers: Both with ReLU activations
- Output layer: Two units for binary classification (Good vs. Bad) with softmax activation
- The loss function used is nn.CrossEntropyLoss(), and the optimizer is SGD with a learning rate of 0.001 and momentum of 0.9.

**Training**

The model is trained over 1,000 epochs. During training:

Forward propagation, loss computation, and backpropagation are performed

Gradients are cleared after each step using optimizer.zero_grad()

Loss is calculated and monitored over the epochs, with periodic updates printed

**Evaluation**

Model performance is evaluated using accuracy, confusion matrix, and classification report. The model achieved an accuracy of 87.67% on the test data. Below is a summary of key metrics:

Precision and Recall for both Good and Bad apples

Visualization of the confusion matrix

**Results and Findings**

Correctly predicted 532 "Good" and 520 "Bad" apples

Misclassified 73 "Bad" apples as "Good" and 75 "Good" apples as "Bad"

Overall accuracy: 87.67%

**Future Improvements**

Experiment with different algorithms such as Random Forest or Gradient Boosting to improve accuracy

Apply hyperparameter tuning to improve performance

Incorporate cross-validation for better model generalization

Explore the reasons why certain apples are classified as "Good" or "Bad" based on their features

**Conclusion**

This project showcases how a binary classifier can be built using PyTorch to predict the quality of apples. While the model performs well, further optimization can improve its performance.

**How to Run**

Clone this repository

Install dependencies using pip install -r requirements.txt

Run the model training script

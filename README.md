# KNN Classifier Project

## Overview

This project implements a K-Nearest Neighbors (KNN) classifier from scratch using Python. It includes functionality for training the model, making predictions, evaluating performance, and visualizing results. The implementation is designed to be educational and demonstrates core concepts of the KNN algorithm and machine learning workflow.

## Features

- Custom KNN Classifier implementation
- Support for Euclidean and Manhattan distance metrics
- Data preprocessing with StandardScaler
- Model evaluation with accuracy, confusion matrix, and classification report
- Visualization of decision boundaries
- Synthetic dataset generation for testing

## Requirements

- Python 3.7+
- NumPy
- scikit-learn
- Matplotlib

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Elotmanix/knn-classifier-project.git
   cd knn-classifier-project
   ```

2. (Optional) Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install numpy scikit-learn matplotlib
   ```

## Usage

Run the main script to train the model, evaluate its performance, and visualize the results:

```
python knn_classifier.py
```

This will:
1. Generate a synthetic dataset
2. Split the data into training and test sets
3. Train the KNN classifier
4. Evaluate the model's performance
5. Display the accuracy, confusion matrix, and classification report
6. Show a plot of the decision boundary

## Customization

You can modify the following parameters in the `main()` function:

- Number of neighbors (K) in the KNN algorithm
- Distance metric ('euclidean' or 'manhattan')
- Dataset parameters (in the `create_dataset()` function)

## Contributing

Contributions to improve the implementation or add new features are welcome. Please feel free to submit a pull request or open an issue to discuss potential changes.


## Contact

If you have any questions or feedback, please open an issue on the GitHub repository.

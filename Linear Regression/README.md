# Multivariate Linear Regression from Scratch (NumPy)

This repository features a high-performance, object-oriented implementation of **Multivariate Linear Regression** built entirely from the ground up using Python and NumPy.

## 🎯 Project Purpose

The goal of this project is to demonstrate a deep understanding of the mathematical foundations of machine learning. By avoiding high-level libraries like `scikit-learn` for the model logic, this implementation showcases:

* **Vectorized Math**: Efficiently handling $O(n)$ operations using matrix multiplication ($X\theta$).
* **Optimization**: Implementing the Gradient Descent algorithm to minimize Mean Squared Error (MSE).
* **Software Engineering**: Transitioning from functional scripts to a clean, class-based API.

---

## 🛠️ Technical Implementation

### Core Algorithm

The model uses **Batch Gradient Descent** to optimize weights ($w$) and bias ($b$). The gradients are calculated using the following vectorized forms:

* **Weight Gradient ($dw$)**: $\frac{1}{m} X^T \cdot (Y_{pred} - Y)$
* **Bias Gradient ($db$)**: $\frac{1}{m} \sum (Y_{pred} - Y)$

### Optimization Features

* **Feature Scaling**: Implemented `StandardScaler` to normalize input features, ensuring the cost function remains spherical for faster convergence.
* **Vectorization**: Leveraged NumPy's `@` operator for dot products, significantly outperforming Python `for-loops`.
* **Cost Tracking**: The class maintains a `cost_history` to monitor convergence and debug learning rate issues.

---

## 📊 Performance Visualization

The effectiveness of the scratch implementation is verified by plotting the **Mean Squared Error** over iterations. A successful run shows the typical "L-curve" convergence, proving the gradients are correctly updating the parameters toward the global minimum.

---

## 🏗️ Architectural Refactor: Functional to OOP

Originally developed as a series of functions, the project was refactored into a **Class-based structure** to follow industry best practices:

* **Encapsulation**: Weights, bias, and hyperparameters are stored as object attributes.
* **Predictability**: Adheres to the standard `.fit()` and `.predict()` interface used in professional ML pipelines.
* **Reusability**: The `LinearRegression` class can be easily imported into other projects or extended for Regularization (Lasso/Ridge).

---

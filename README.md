# 🤖 Machine Learning Algorithms Collection

Welcome to my personal repository containing implementations of various machine learning algorithms and exploratory data analysis projects. This repository aims to demonstrate foundational machine learning concepts through hands-on Python code and real-world datasets.

## 🚀 Project Overview

### 1. Linear Regression Implementation 🔍

A from-scratch implementation of a Linear Regression model without relying on external machine learning libraries like scikit-learn.

- **Location:** `LinearRegression/MyRegression.py`  
- **Features:**  
  - Custom Linear Regression class  
  - Feature scaling support  
  - Multiple regression capability  
  - Performance metrics including MSE, RMSE, MAE, and R-squared  
- **Purpose:**  
  Understand the inner workings of Linear Regression by building it from the ground up.

### 2. Exploratory Data Analysis (EDA) Projects 📊

Projects focused on data exploration and preliminary machine learning modeling on different datasets.

- **Heart Disease Analysis:**  
  - **Notebook:** `Exploratory Data Analysis/heart_disease.ipynb`  
  - **Dataset:** `datasets/heart_2020_cleaned.csv` (~24 MB)  
  - **Description:** Data cleaning, visualization, feature engineering, and predictive modeling (Logistic Regression, Decision Trees, Random Forest) to assess heart disease risk.

- **Rental Price Prediction:**  
  - **Notebook:** `Exploratory Data Analysis/rent_price.ipynb`  
  - **Dataset:** `datasets/rent_price_dataset_buca.csv` (~5 KB)  
  - **Description:** Analysis and modeling of rental prices in the Buca area using regression algorithms like Linear Regression and KNN Regressor.

### 3. Datasets 📁

Raw datasets used across the projects are stored here for easy access and reproducibility.

- `datasets/heart_2020_cleaned.csv` — Cleaned heart disease dataset  
- `datasets/rent_price_dataset_buca.csv` — Rental price dataset from Buca

### 4. Deep Learning 🔍
Implementation of a Neural Network model that predicts MNIST data from scratch using Numpy and Pandas, without the need for deep learning libraries.

- **Location:** `deepLearning/neuralNetwork.ipynb`  
- **Features:**  
  - Custom Neural Network  
  - Batch Gradient Descent  
  - High Accuracy  
- **Purpose:**  
  To understand the logic of deep learning and artificial neural networks by building a neural network from scratch.
## 🛠 Installation & Usage

1. Clone this repository:
    ```bash
    git clone https://github.com/yusufduman78/MachineLearningAlgorithms.git
    cd MachineLearningAlgorithms
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(If `requirements.txt` does not exist, install essential libraries manually:)*  
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn jupyter
    ```

3. Launch Jupyter Notebook for EDA projects:
    ```bash
    jupyter notebook
    ```
    Open the desired `.ipynb` file in your browser.

4. Run the linear regression script directly:
    ```bash
    python LinearRegression/MyRegression.py
    ```

## 🤝 Contributions

Contributions are welcome! Please feel free to fork the repository, create a branch, and open a pull request.  
Before opening a new issue, kindly check if it has already been reported.

## 📄 License

This repository is licensed under the [MIT License](LICENSE).

---

Made with ❤️ by [Yusuf Duman](https://github.com/yusufduman78)

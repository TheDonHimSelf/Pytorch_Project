Sentiment Classification on IMDb Dataset 🎬

This project explores sentiment analysis on the IMDb movie reviews dataset using both classical machine learning and deep learning models. The goal is to compare baseline methods with more advanced neural network architectures to evaluate their performance on text classification.

📌 Project Overview

Dataset: IMDb dataset (50,000 movie reviews labeled as positive or negative).

Models Implemented:

Logistic Regression (baseline).

Deep Neural Network with Embedding → GRU → Dense layers.

Frameworks/Libraries:

TensorFlow/Keras

Scikit-learn

NumPy, Pandas, Matplotlib

⚙️ Features

Data preprocessing: tokenization, padding, and splitting into training/validation/test sets.

Baseline classification with Logistic Regression.

Deep learning model with embedding and GRU layers.

Hyperparameter tuning for better accuracy.

Performance evaluation with accuracy and loss curves.

Reproducibility ensured with fixed random seeds.

🚀 Getting Started
Prerequisites

Make sure you have Python 3.8+ installed and Jupyter Notebook.

Installation

Clone the repository:

git clone <your-repo-url>
cd <your-repo-folder>


Install dependencies:

pip install -r requirements.txt

Run the Project

Open the notebook and execute all cells:

jupyter notebook DLproject_FInal.ipynb

📊 Results

Logistic Regression: Quick baseline model with moderate performance.

Deep Learning Model (Embedding + GRU + Dense): Outperformed the baseline, demonstrating the effectiveness of sequence models for sentiment classification.

🔧 Hyperparameters

Key tuned parameters include embedding dimension, GRU units, batch size, and learning rate.

💻 Hardware Used

Trained on Google Colab (T4 GPU).

📜 License

This project is for academic purposes. You are free to use and adapt it with proper attribution.

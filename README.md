# Credit Card Fraud Detection using GANs (Vanilla, WGAN, LSGAN)

## 📌 Project Overview
This project addresses the problem of **imbalanced datasets** in credit card fraud detection. Since fraudulent transactions are extremely rare compared to normal ones, standard machine learning models often struggle to detect them.

To solve this, I implemented and compared three different **Generative Adversarial Networks (GANs)** to generate synthetic fraud data and balance the dataset:
1.  **Vanilla GAN**
2.  **WGAN (Wasserstein GAN)**
3.  **LSGAN (Least Squares GAN)**

The performance of these techniques was evaluated using a **Random Forest Classifier**.

## 📂 Dataset
The project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).
- **Features:** 28 PCA-transformed features (V1-V28) + Time + Amount.
- **Target:** Class (0 = Normal, 1 = Fraud).
- **Imbalance:** The dataset is highly unbalanced, with fraud cases accounting for a very small percentage.

## 🛠️ Methodologies Implemented

### 1. Data Preprocessing
- Scaling features using `StandardScaler`.
- Dropping irrelevant columns (`Time`).
- Preparing data tensors for PyTorch.

### 2. Generative Models (GANs)
- **Vanilla GAN:** The standard GAN architecture with Binary Cross Entropy Loss.
- **WGAN:** Uses Wasserstein loss with weight clipping to improve training stability.
- **LSGAN:** Uses Least Squares loss (MSE) to generate higher quality samples and prevent vanishing gradients.

### 3. Evaluation
- The generated synthetic data is combined with the original dataset.
- A **Random Forest Classifier** is trained on:
    1. Original Imbalanced Data.
    2. Data balanced by Vanilla GAN.
    3. Data balanced by WGAN.
    4. Data balanced by LSGAN.
- **Metrics used:** Precision, Recall, F1-Score, and ROC-AUC.

## 📊 Results & Visualization
The project includes several visualizations to analyze performance:
- **Class Distribution:** Before and after balancing.
- **PCA Plots:** To visualize the distribution of Real vs. Generated Fraud data.
- **Confusion Matrices:** To view True Positives and False Negatives.
- **Bar Charts:** Comparison of F1-Scores and Accuracy across all models.

*Note: Based on the experiments, LSGAN and WGAN generally showed better stability and distribution coverage compared to Vanilla GAN.*

## 🚀 How to Run
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)

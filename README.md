# CODSOFT — Machine Learning Internship Projects

This repository contains the machine learning tasks completed as part of the **CodSoft ML Internship**. Each task is a self-contained Jupyter notebook covering data preprocessing, exploratory data analysis (EDA), feature engineering, model training/comparison, and evaluation.

## Projects

| # | Notebook | Task | Type |
|---|---|---|---|
| 1 | [`movie-genre-classification-.ipynb`](movie-genre-classification-.ipynb) | Predict a movie's genre from its plot description | Multi-class text classification (NLP) |
| 2 | [`credit-card-fraud-detection.ipynb`](credit-card-fraud-detection.ipynb) | Detect fraudulent credit card transactions | Binary classification (imbalanced data) |
| 3 | [`ham-or-spam-classifier-using-nlp-techniques.ipynb`](ham-or-spam-classifier-using-nlp-techniques.ipynb) | Classify SMS/email messages as spam or ham | Binary text classification (NLP) |

---

## 1. Movie Genre Classification

Predicts a movie's genre (27 possible classes — action, drama, comedy, documentary, western, etc.) from its plot summary text.

- **Dataset**: `train_data.txt` / `test_data_solution.txt`, `:::`-delimited plot descriptions with genre labels (10,843 test rows). Genre distribution is heavily imbalanced.
- **Preprocessing**: NLTK tokenization, stopword removal, Porter stemming, regex/punctuation cleaning, then `TfidfVectorizer` for feature extraction.
- **Models trained**: SVM (SVC) and Logistic Regression (imports also included Decision Tree, Random Forest, Extra Trees, Naive Bayes variants, and XGBoost for experimentation).
- **Results**:

  | Model | Accuracy | Train/Predict Time |
  |---|---|---|
  | SVM (SVC) | 0.577 | ~110 min |
  | Logistic Regression | 0.582 | ~3.3 min |

  Both models perform well on frequent genres (documentary F1 ≈ 0.75, drama ≈ 0.64, western ≈ 0.81) but struggle on rare genres (biography, crime, fantasy, history — F1 ≈ 0) due to severe class imbalance.
- **Takeaway**: Logistic Regression was preferred — near-identical accuracy to SVM at over 30x the training speed.

## 2. Credit Card Fraud Detection

Detects fraudulent transactions from anonymized (PCA-transformed) credit card transaction data.

- **Dataset**: 284,807 transactions × 31 columns (`Time`, `Amount`, `V1`–`V28`, `Class`). Extremely imbalanced — fraud is a tiny minority of transactions.
- **Preprocessing**: `StandardScaler` on `Amount`; class imbalance addressed via **random undersampling** and **SMOTE oversampling**.
- **Models trained**: Logistic Regression, Decision Tree, Random Forest, SVC, KNN, Gaussian Naive Bayes, AdaBoost, Gradient Boosting, Bagging, Extra Trees, SGD, and a Voting Classifier ensemble.
- **Results** (best performers):

  | Data strategy | Model | Accuracy | Precision | Recall | F1 |
  |---|---|---|---|---|---|
  | Full imbalanced data | Extra Trees / Voting Classifier | 0.9995 | 0.931 | 0.736 | 0.822 |
  | SMOTE-balanced data | Decision Tree | 0.998 | 0.998 | 0.999 | **0.998** |

- **Takeaway**: SMOTE oversampling combined with a Decision Tree gave the best, most reliable results and was chosen as the final model, exported with `joblib` as `credit_card_model.pkl`.

## 3. Spam / Ham Classifier (NLP)

Classifies SMS/email text messages as **spam** or **ham** (legitimate).

- **Dataset**: SMS Spam Collection dataset (5,572 messages, deduplicated).
- **Preprocessing**: Character/word/sentence count features, punctuation & stopword removal, Porter stemming, word clouds for EDA, and `TfidfVectorizer` (max 3,000 features) for vectorization.
- **Models trained**: GaussianNB, MultinomialNB, BernoulliNB.
- **Results**:

  | Model | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) |
  |---|---|---|---|---|
  | GaussianNB | 0.865 | 0.48 | 0.89 | 0.63 |
  | MultinomialNB | 0.978 | 0.99 | 0.83 | 0.90 |
  | BernoulliNB | **0.982** | 0.98 | 0.87 | 0.92 |

- **Takeaway**: MultinomialNB was chosen as the final saved model (`model.pkl`, with `vectorizer.pkl`) for its strong precision — minimizing the risk of flagging real messages as spam.

---

## Tech Stack

- **Language**: Python
- **Data handling**: pandas, numpy
- **Visualization**: matplotlib, seaborn, wordcloud
- **NLP**: nltk (tokenization, stopwords, Porter stemming)
- **Machine learning**: scikit-learn, imbalanced-learn (SMOTE), XGBoost
- **Model persistence**: joblib, pickle

## Repository Structure

```
CODSOFT/
├── movie-genre-classification-.ipynb
├── credit-card-fraud-detection.ipynb
├── ham-or-spam-classifier-using-nlp-techniques.ipynb
└── README.md
```

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/ELANGOSTUDENT/CODSOFT.git
   cd CODSOFT
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn nltk wordcloud imbalanced-learn xgboost joblib
   ```
3. Download NLTK data (required for the two NLP notebooks):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```
4. Update the dataset file paths at the top of each notebook to point to your local copy of the dataset, then run the cells in order.

## About

These projects were built as part of the **CodSoft Machine Learning Internship**, focusing on end-to-end ML workflows: data cleaning, EDA, feature engineering, model comparison, and evaluation across both structured (tabular) and unstructured (text/NLP) data.

**Author**: [ELANGOSTUDENT](https://github.com/ELANGOSTUDENT)

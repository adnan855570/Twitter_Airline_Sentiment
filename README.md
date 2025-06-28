# Twitter Airline Sentiment Analysis

This project analyzes public sentiment toward airlines based on tweets, using Natural Language Processing and Machine Learning.  
We classify tweets about airlines as **positive**, **neutral**, or **negative** using both classic and neural models.  

---

## Table of Contents

- [Introduction](#introduction)
- [Project Workflow](#project-workflow)
  - [1. Install and Import Libraries](#1-install-and-import-libraries)
  - [2. Load and Inspect Dataset](#2-load-and-inspect-dataset)
  - [3. Clean and Normalize Text](#3-clean-and-normalize-text)
  - [4. TF-IDF Vectorization and Dataset Splitting](#4-tf-idf-vectorization-and-dataset-splitting)
  - [5. Naive Bayes Classifier](#5-naive-bayes-classifier)
  - [6. MLP Classifier](#6-mlp-classifier)
  - [7. Final Evaluation on Test Set](#7-final-evaluation-on-test-set)
  - [8. Vector Semantics & POS Tagging](#8-vector-semantics--pos-tagging)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)
- [License](#license)

---

## Introduction

**Goal:**  
Classify the sentiment of airline-related tweets into *negative*, *neutral*, or *positive* using NLP and ML.

We use the [Twitter Airline Sentiment Dataset](https://huggingface.co/datasets/osanseviero/twitter-airline-sentiment), process the data, build models (Naive Bayes & MLP), evaluate performance, and analyze language semantics.

---

## Project Workflow

### 1. Install and Import Libraries

We use pandas, numpy, scikit-learn, NLTK, matplotlib, seaborn, and more.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import nltk
# ... (see code below for specifics)
```

## 2. Load and Inspect Dataset

Loads the dataset from Hugging Face.

Shows the initial shape and a preview of tweets.

```python
from datasets import load_dataset
import pandas as pd

data = load_dataset("osanseviero/twitter-airline-sentiment")
df = pd.DataFrame(data['train'])
print("Initial Dataset Shape:", df.shape)
print(df[['text', 'airline_sentiment', 'airline_sentiment_confidence']].head().to_markdown())
```

Filter the data: Keep only rows with 100% confidence and at least 5 words.

```python
df = df[df['airline_sentiment_confidence'] == 1.0]
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
df = df[df['word_count'] >= 5]
df = df.reset_index(drop=True)
print("Filtered by Confidence == 1.0:", df.shape)
print("Filtered by Word Count >= 5:", df.shape)
```

### 3. Clean and Normalize Text

Lowercases, removes links/mentions/hashtags, punctuation, numbers, stopwords, and lemmatizes words.

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

df['cleaned_text'] = df['text'].apply(clean_text)
print(df[['text', 'cleaned_text']].head(5).to_markdown(index=False))
```

### 4. TF-IDF Vectorization and Dataset Splitting

- Vectorizes tweets with TF-IDF (top 4000 words).
- Maps sentiment labels to numbers (0 = negative, 1 = neutral, 2 = positive).
- Splits into 80% train, 10% validation, 10% test.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

vectorizer = TfidfVectorizer(max_features=4000)
X = vectorizer.fit_transform(df['cleaned_text'])
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
y = df['airline_sentiment'].map(label_map)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=10)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=10)

print(f"Train Shape: {X_train.shape}, Validation Shape: {X_val.shape}, Test Shape: {X_test.shape}")
```

### 5. Naive Bayes Classifier

Trains a Multinomial Naive Bayes classifier on TF-IDF features.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
val_preds = nb_model.predict(X_val)

print(classification_report(y_val, val_preds, target_names=["Negative", "Neutral", "Positive"]))
```

### 6. MLP Classifier

Trains a Multi-layer Perceptron (neural net) classifier.

```python
from sklearn.neural_network import MLPClassifier

mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=30, random_state=10)
mlp_model.fit(X_train, y_train)
val_preds = mlp_model.predict(X_val)

print(classification_report(y_val, val_preds, target_names=["Negative", "Neutral", "Positive"]))
```

### 7. Final Evaluation on Test Set

Evaluates both models on test data.  
Shows classification report (precision, recall, F1) and confusion matrix plots.

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Naive Bayes
nb_test_preds = nb_model.predict(X_test)
print(classification_report(y_test, nb_test_preds, target_names=["Negative", "Neutral", "Positive"]))
cm_nb = confusion_matrix(y_test, nb_test_preds)
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues')
plt.title("Naive Bayes - Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# MLP
mlp_test_preds = mlp_model.predict(X_test)
print(classification_report(y_test, mlp_test_preds, target_names=["Negative", "Neutral", "Positive"]))
cm_mlp = confusion_matrix(y_test, mlp_test_preds)
sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Oranges')
plt.title("MLP - Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

### 8. Vector Semantics & POS Tagging 

#### 8.1 Word Similarity with GloVe

Loads GloVe embeddings.  
Shows cosine similarities between word pairs.

```python
print("happy - joy: 0.9213")
print("computer - laptop: 0.8157")
# ...
```

#### 8.2 Nearest Neighbors in Vector Space
Shows top 5 most similar words for a given word.

```python
# Example output:
Top Neighbors for 'language':
  languages: 0.95
  linguistics: 0.93
  # ...
```

#### 8.3 POS Tagging with Brown Corpus
Tags first 100 sentences and shows most frequent tags.


``` python
# Example output:
NN: 150
IN: 80
DT: 76
# ...
```

#### 8.4 POS Ambiguity
Examples of ambiguous sentences and their tags.

``` python
# Example output:
Sentence 1: The group presented a musical program .
Word      NLTK Tag
The       DT
group     NN
presented VBD
# ...
```

#### Installation
```python
git clone https://github.com/yourusername/twitter-airline-sentiment.git
cd twitter-airline-sentiment
```

## Dataset

- Loaded automatically via Hugging Face Datasets (`osanseviero/twitter-airline-sentiment`)
- GloVe embeddings can be downloaded from [GloVe site](https://nlp.stanford.edu/projects/glove/) for bonus tasks.


## Usage

- Open `Twitter_Airline_Sentiment.ipynb` in Jupyter Notebook or Google Colab.
- Run the notebook cells in order.
- All outputs (tables, metrics, plots) will display as above.


## Dependencies

- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- seaborn
- datasets (Hugging Face Datasets)


## Troubleshooting

- **Dataset not found:** Make sure you have internet for `load_dataset()` from Hugging Face.
- **NLTK errors:** Use `nltk.download()` as shown in the notebook/code.
- **GloVe file not found:** Download `glove.6B.50d.txt` for vector tasks.
- **Other errors:** Double-check package versions.


## Contributors

This project was developed solely by **Muhammad Adnan Mushtaq**.





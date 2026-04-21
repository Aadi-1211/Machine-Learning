# Netflix Content Classifier

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Project%20Ready-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A professional machine learning project that classifies Netflix titles as **Movie** or **TV Show** using **natural language processing (NLP)** and **supervised machine learning**.

This project was developed to demonstrate an end-to-end applied ML workflow, including data preprocessing, TF-IDF feature engineering, model development, evaluation, and deployment readiness. The repository is designed to present the work clearly to recruiters, hiring managers, and collaborators.

## Project Overview

The classifier was built on Netflix title metadata and text fields to distinguish between the two primary content types: **Movie** and **TV Show**. The workflow included:

- Exploratory data analysis on 7,787 Netflix titles
- Handling text-based features for content classification
- TF-IDF vectorisation with unigram and bigram features
- Logistic Regression model training
- Evaluation using classification-focused metrics such as F1 score
- Model serialization for reuse and deployment

## Key Highlights

- Built an NLP classification pipeline using **scikit-learn**
- Used **TF-IDF** with up to 8,000 features and **1–2 gram** range
- Trained a **Logistic Regression** classifier with reproducible settings
- Exported the trained model as a serialized `.pkl` artifact
- Repository structured for portfolio presentation and future deployment

## Repository Structure

```text
Netflix-Content-Classifier/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── app.py
├── model/
│   └── netflix_classifier.pkl
├── src/
│   └── predict.py
├── notebooks/
│   └── training_notebook.ipynb   # optional: add your original notebook here
└── assets/
    └── github-banner.png         # optional: add project screenshots here
```

## Model Information

The uploaded model file is a serialized **scikit-learn Pipeline** containing:

- `TfidfVectorizer(max_features=8000, ngram_range=(1, 2))`
- `LogisticRegression(max_iter=1000, random_state=42)`

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/<your-username>/Machine-Learning.git
cd Machine-Learning
pip install -r requirements.txt
```

## Usage

### Run the simple demo app

```bash
python app.py
```

### Example prediction in Python

```python
from joblib import load

model = load("model/netflix_classifier.pkl")
text = ["A group of teenagers uncover dark secrets in a small town."]
prediction = model.predict(text)
print(prediction[0])
```

## Suggested Inputs

Try inputs such as:

- `A detective investigates a murder in a small coastal town.`
- `A family comedy film about a talking dog.`
- `A long-running drama series following hospital staff.`

## Tech Stack

- Python
- scikit-learn
- joblib
- pandas
- numpy

## Professional Portfolio Summary

This repository demonstrates practical skills in:

- Machine learning workflow design
- Text preprocessing and NLP feature engineering
- Binary classification modelling
- Model persistence and reproducibility
- GitHub project presentation for a professional portfolio

## Recommended GitHub Enhancements

To make the repository look stronger on GitHub, add the following:

- A notebook showing training and evaluation
- A screenshot of results or app interface in `assets/`
- A short **About** description on GitHub
- Topics such as `machine-learning`, `nlp`, `scikit-learn`, `classification`, `python`
- A pinned repository on your GitHub profile

## GitHub About Section

Use this in the repository **About** box:

> NLP-based Netflix content classifier built with TF-IDF and Logistic Regression using scikit-learn.

## Topics to Add on GitHub

```text
python, machine-learning, nlp, scikit-learn, text-classification, logistic-regression, tfidf, netflix-dataset
```

## Future Improvements

- Add a Streamlit or Gradio web interface
- Compare multiple classifiers in the repository
- Add confusion matrix and classification report visuals
- Deploy as a lightweight web app
- Add unit tests for prediction functions

## License

This project can be released under the MIT License.

---

**Author:** Aditi Chakravarthy  
**Focus:** Machine Learning | NLP | Applied Analytics

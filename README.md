This repository provides an implementation of machine learning techniques to detect fake news articles. The goal of this project is to build a reliable and accurate model that distinguishes fake news from real news based on text data.

Overview

Fake news detection has become increasingly important in combating the spread of misinformation. This project uses machine learning models to classify news articles as fake or real, utilizing techniques like natural language processing (NLP) and text classification.

Features

Text preprocessing, including tokenization, stemming, and stopword removal

Vectorization of text data using:

Bag of Words (BoW)

Term Frequency-Inverse Document Frequency (TF-IDF)


Implementation of machine learning models:

Logistic Regression

Naive Bayes

Support Vector Machines (SVM)

Random Forest


Model evaluation using metrics such as accuracy, precision, recall, and F1-score

Visualization of performance metrics


Dataset

The project uses the Fake News Detection Dataset from Kaggle, which contains labeled news articles.

Dataset Details:

Number of articles: ~20,000

Columns:

id: Unique ID for each news article

title: The title of the news article

text: The body of the news article

label: 1 for fake news, 0 for real news



Technologies Used

Python

Libraries:

Pandas, NumPy for data manipulation

Matplotlib, Seaborn for data visualization

Scikit-learn for machine learning models

NLTK and SpaCy for NLP preprocessing



Installation

1. Clone the repository:

git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection


2. Set up a virtual environment and install dependencies:

python3 -m venv env
source env/bin/activate   # On Windows: env\Scripts\activate
pip install -r requirements.txt



Usage

1. Download the dataset from Kaggle and place it in the data/ folder.


2. Run the Jupyter notebook for data exploration and model training:

jupyter notebook notebooks/fake_news_detection.ipynb


3. To test the model on custom input, use:

python scripts/predict.py --text "Your news article text here"



Project Structure

fake-news-detection/
│
├── data/                   # Dataset folder
├── notebooks/              # Jupyter notebooks for analysis
├── scripts/                # Python scripts for preprocessing, training, and prediction
├── models/                 # Trained machine learning models
├── results/                # Evaluation results
├── requirements.txt        # List of dependencies
├── README.md               # Project documentation
└── LICENSE                 # License file

Results

Logistic Regression achieved the best performance with:

Accuracy: 94.2%

Precision: 92.8%

Recall: 93.7%

F1-Score: 93.2%


Other models like Naive Bayes and SVM also performed well.


Future Work

Implement deep learning models (e.g., LSTMs, BERT) for improved accuracy.

Enhance the dataset with additional labeled news articles.

Build a web application for real-time fake news detection.

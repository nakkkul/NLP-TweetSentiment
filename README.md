# 📝 NLP-TweetSentiment

![NLP](https://img.shields.io/badge/NLP-SentimentAnalysis-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen) ![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
NLP-TweetSentiment is a sentiment analysis project that classifies tweets into positive, negative, or neutral sentiments. This project leverages **TF-IDF, BoW, Word2Vec, and BERT** embeddings along with machine learning models to predict tweet sentiment.

## 🚀 Features
- Preprocessing of tweets (tokenization, stopword removal, lemmatization)
- Sentiment classification using **Logistic Regression, SVM, and Random Forest**
- Comparison of different text vectorization techniques (**TF-IDF, BoW, Word2Vec**)
- Implementation of a **BERT-based model** for deep learning-based sentiment analysis
- Model performance visualization and comparison

## 📂 Dataset
The dataset consists of tweets labeled as **positive, negative, or neutral**. A sampled dataset (`Sampled_data.csv`) is used for better efficiency.

## 🔧 Tech Stack
- **Programming Language:** Python 🐍
- **NLP Libraries:** NLTK, spaCy, scikit-learn, Transformers
- **Machine Learning Models:** Logistic Regression, SVM, Random Forest
- **Deep Learning Model:** BERT (Trained for 1 Epoch due to hardware limitations)
- **Visualization:** Matplotlib, Seaborn

## 📊 Model Performance
| Model                | Vectorizer  | Accuracy |
|----------------------|------------|----------|
| Logistic Regression | TF-IDF      | 77.13%   |
| SVM                | TF-IDF      | 77.08%   |
| SVM                | BoW         | 77.06%   |
| Logistic Regression | BoW         | 76.44%   |
| Random Forest      | BoW         | 75.34%   |
| Random Forest      | TF-IDF      | 75.17%   |
| Logistic Regression | Word2Vec    | 67.62%   |
| SVM                | Word2Vec    | 66.65%   |
| Random Forest      | Word2Vec    | 64.59%   |
| **BERT**           | Transformers | **50.0%** (1 Epoch) |

⚠️ **Note:** The **BERT model** was trained for only **1 epoch** due to system limitations, leading to lower accuracy.

## 📌 Installation & Usage
### 1️⃣ Clone the Repository
\`\`\`bash
git clone https://github.com/nakkkul/NLP-TweetSentiment.git
cd NLP-TweetSentiment
\`\`\`
### 2️⃣ Install Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`
### 3️⃣ Run the Model
\`\`\`bash
python model_train.py
\`\`\`

## 📈 Results & Visualizations
- Model comparison using bar charts 📊
- ROC-AUC curves for different classifiers 📈
- Feature importance analysis for machine learning models 🏆

## 🤝 Contributing
Feel free to fork the repo, create a new branch, and submit a pull request. Suggestions and improvements are welcome! 🎯

## 📞 Contact
For any questions or collaboration, reach out via:
- **GitHub:** [@nakkkul](https://github.com/nakkkul)
- **Email:** kumar.nakul111@gmail.com

---

🚀 *NLP-TweetSentiment - An AI-Powered Sentiment Analysis Project* 🚀

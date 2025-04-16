# 📊 LinkedIn Review Sentiment Analysis

Analyze the sentiment behind LinkedIn reviews with the power of Natural Language Processing and Machine Learning!  
This project explores how well various models can classify sentiments (positive or negative) based on real-world review data from LinkedIn.

---

## 📌 Project Overview

With the growing volume of user-generated content on professional platforms like LinkedIn, understanding public sentiment can provide valuable insights for companies, job seekers, and researchers.

In this project, I built a **sentiment analysis pipeline** to classify LinkedIn reviews using:
- Preprocessing (cleaning, tokenizing, removing stopwords, etc.)
- Feature Extraction via **TF-IDF**
- Lexicon-based scoring
- Machine Learning Models: **Random Forest**, **SVM**, and **Logistic Regression**

---

## 🛠️ Technologies Used

- **Python 3**
- **Pandas**, **NumPy** for data processing
- **Scikit-learn** for ML modeling
- **NLTK**, **Sastrawi** for Indonesian text preprocessing (optional if reviews are in Bahasa)
- **Matplotlib**, **Seaborn** for visualization
- **Jupyter Notebook / Google Colab** for experimentation

---

## 📁 Dataset

> ⚠️ *Due to privacy and platform terms, the dataset isn't included in this repo. Replace with your own review data or contact me for more info.*

The dataset was split into:
- **Training & Testing (80/20)**
- **Training & Testing (70/30)**

Each review was labeled with a sentiment (`positive`, `negative`).

---

## 🔄 Preprocessing Steps

- Lowercasing
- Removing punctuation, links, numbers
- Tokenization
- Stopword removal
- Stemming (if needed)
- Vectorization using **TF-IDF**

---

## 🤖 Models & Results

| Model                 | Data Split | Accuracy      |
|-----------------------|------------|---------------|
| Random Forest         | 80/20      | 83.33%        |
| Random Forest         | 70/30      | 82.65%        |
| SVM (Linear Kernel)   | 80/20      | 85.29%        |
| SVM (Linear Kernel)   | 70/30      | 84.98%        |
| Logistic Regression   | 80/20      | 84.52%        |
| Logistic Regression   | 70/30      | 83.95%        |
| Conv Neural Network   | 80/20      | 86.26%        |
| Conv Neural Network   | 70/30      | 85.44%        |
| Gated Recurrent Unit  | 70/30      | 85.30%        |

✅ Inference supported  
✅ Lexicon-based scoring available  
✅ Easy to plug in new data

---

## 🔍 Sample Inference

```python
review = "Lingkungan kerja yang sangat suportif dan peluang belajar terbuka lebar"
text_vector = vectorizer.transform([review])
predicted_sentiment = model.predict(text_vector)
print(predicted_sentiment)
```

---

## 📈 Future Improvements

- Incorporate deep learning (e.g., **GRU**, **BERT Indo**)
- Add interactive dashboard for live inference
- Expand lexicon-based analysis with hybrid techniques

---

## 💡 Conclusion

This project proves that traditional ML models with proper preprocessing can achieve competitive results in sentiment classification tasks. With SVM leading the pack, this pipeline is ready for deeper integration into analytics tools or dashboards!

---

## 📬 Contact

Feel free to reach out or collaborate on similar NLP projects:
- 💼 [LinkedIn](https://linkedin.com/in/naufalarsa)
- ✉️ naufal.arsa.27@gmail.com
- 🧠 Portfolio: [GitHub](https://github.com/NaufalArsa)

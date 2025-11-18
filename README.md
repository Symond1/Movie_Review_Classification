---

# 🎬 Movie Review Classification using Naive Bayes

A simple IMDB movie review classification project using **Multinomial Naive Bayes** and **Gaussian Naive Bayes**, along with a **Streamlit dashboard** to visualize results.

---

## 📌 Features

* IMDB movie review dataset
* Text preprocessing using TF-IDF
* Two models:

  * Multinomial Naive Bayes
  * Gaussian Naive Bayes
* Comparison of:

  * Accuracy
  * Precision
  * Recall
  * F1-Score
* Confusion matrices
* Streamlit web interface

---

## 📁 Project Structure

```
movie-bayes-project/
│
├── main.py
├── naive_bayes_multinomial.py
├── naive_bayes_gaussian.py
├── compare_and_visualize.py
├── IMDB_Dataset.csv
│
└── README.md
```

---

## 🚀 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/movie-bayes-project.git
cd movie-bayes-project
```

### 2️⃣ Install required Python libraries

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas scikit-learn matplotlib seaborn streamlit
```

---

## ▶️ Running the Project

### **To run the backend comparison:**

```bash
python compare_and_visualize.py
```

This generates:

* `comparison_chart.png`
* `confusion_matrix_multinomial.png`
* `confusion_matrix_gaussian.png`

---

### **To launch the Streamlit dashboard:**

```bash
streamlit run main.py
```

This project uses the publicly available IMDB Movie Review Dataset (50,000 labeled reviews).
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews


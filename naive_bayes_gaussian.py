# naive_bayes_gaussian.py


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def run_gaussian_nb():
    df = pd.read_csv("IMDB_Dataset.csv")
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["sentiment"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=2500)
    X_train_vec = vectorizer.fit_transform(X_train).toarray()  # must convert to array
    X_test_vec = vectorizer.transform(X_test).toarray()

    model = GaussianNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy, report, cm

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
from pathlib import Path



def train_nb():
    file_path = f"{Path(__file__).parent.parent}"
    print
    df = pd.read_csv(f"{file_path}/data/data.csv")

    df = df.dropna()

    df['label'] = df['label'].map({'spam': 1, 'not spam': 0})
    df['label'] = df['label'].astype(int) 

    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_tfidf, y_train)

    with open(f"{file_path}/weights/tfidf_vec.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)

    with open(f"{file_path}/weights/nb.pkl", 'wb') as f:
        pickle.dump(nb_classifier, f)

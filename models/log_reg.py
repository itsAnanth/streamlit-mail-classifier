import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pickle 
from pathlib import Path

def train_lg():
    file_path = f"{Path(__file__).parent.parent}"
    df = pd.read_csv(f"{file_path}/data/data.csv")

    df = df.dropna()
    df['label'] = df['label'].map({'spam': 1, 'not spam': 0})
    df['label'] = df['label'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    log_reg = LogisticRegression()
    log_reg.fit(X_train_tfidf, y_train)

    test_lg(log_reg, X_test_tfidf, y_test)
    
    with open(f"{file_path}/weights/lg.pkl", 'wb') as f:
        pickle.dump(log_reg, f)
        
    return log_reg

def test_lg(model: LogisticRegression, xtest, ytest):
    pred = model.predict(xtest)
    accuracy = accuracy_score(ytest, pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        ytest, 
        pred, 
        average='weighted'
    )
    
    print(f"\nModel: LogisticRegression")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    
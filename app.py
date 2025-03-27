import streamlit as st
import pickle
from pathlib import Path
file_path = Path(__file__).parent

with open(f"{file_path}/weights/nb.pkl", 'rb') as f:
    nb_model = pickle.load(f) 

with open(f"{file_path}/weights/tfidf_vec.pkl", 'rb') as f:
    vectorizer = pickle.load(f) 

with open(f"{file_path}/weights/lg.pkl", 'rb') as f:
    log_reg_model = pickle.load(f) 



st.title("Email Spam Classifier 🚀")

email_text = st.text_area("Enter the email content:")

model_choice = st.selectbox("Choose a model:", ["Naïve Bayes", "Logistic Regression"])

if st.button("Classify Email"):
    if email_text:
        email_tfidf = vectorizer.transform([email_text])
        if model_choice == "Naïve Bayes":
            model = nb_model
        else:
            model = log_reg_model
        prediction = model.predict(email_tfidf)[0]

        if prediction == 0:
            result = "✅ Not Spam"
        else:
            result = "🚨 Spam Email" 
        st.subheader(f"Prediction: {result}")
    else:
        st.warning("Please enter an email to classify.")

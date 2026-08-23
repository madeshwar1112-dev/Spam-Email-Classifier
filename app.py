import streamlit as st
import joblib

model=joblib.load("spam_classifier.pkl")
vectorizer=joblib.load("vectorizer.pkl")

st.set_page_config(page_title="Spam Email Classifier",page_icon="📧")

st.title("📧 Spam Email Classifier")
st.write("AI-powered email spam detection system")

email=st.text_area("Enter your email/message:",height=200)

if st.button("Check Email"):
    if email.strip()=="":
        st.warning("Please enter an email or message.")
    else:
        data=vectorizer.transform([email])
        prediction=model.predict(data)[0]
        probability=model.predict_proba(data)[0]

        if prediction==1:
            st.error("🚨 SPAM EMAIL")
            st.write("Spam Probability:",round(probability[1]*100,2),"%")
        else:
            st.success("✅ NOT SPAM")
            st.write("Not Spam Probability:",round(probability[0]*100,2),"%")
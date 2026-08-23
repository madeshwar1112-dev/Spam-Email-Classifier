Project Title: Spam Email Classifier

Overview: A Python-based Machine Learning project designed to process and classify text emails into "Spam" or "Ham" (Not Spam) using Natural Language Processing (NLP) techniques.

Key Components:

train_model.py: Script used to train the machine learning classification model on dataset text.

spam.csv: Dataset containing labeled email/SMS messages used for training and testing.

vectorizer.pkl: Pre-trained text feature extractor (TF-IDF / CountVectorizer) saved for inference.

spam_classifier.pkl: Trained Machine Learning model for instant spam detection.

app.py: Web interface / API application to interactively test and classify new emails in real-time.

Tech Stack: Python, Scikit-learn, Pandas, NLTK, Flask/Streamlit.

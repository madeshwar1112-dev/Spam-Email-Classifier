import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report
import joblib

data=pd.read_csv("spam.csv")

data["label"]=data["label"].map({"ham":0,"spam":1})

X=data["message"]
y=data["label"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

vectorizer=TfidfVectorizer(stop_words="english")
X_train_vectorized=vectorizer.fit_transform(X_train)
X_test_vectorized=vectorizer.transform(X_test)

model=MultinomialNB()
model.fit(X_train_vectorized,y_train)

prediction=model.predict(X_test_vectorized)

print("Accuracy:",accuracy_score(y_test,prediction))
print(classification_report(y_test,prediction))

joblib.dump(model,"spam_classifier.pkl")
joblib.dump(vectorizer,"vectorizer.pkl")

print("Model trained successfully!")
import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    a=[]
    for i in text:
        if i.isalnum():
            a.append(i)
    text=a[:]
    a.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            a.append(i)
    text=a[:]
    a.clear()
    for i in text:
        a.append(ps.stem(i))
    return " ".join(a)
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))
st.title('SMS spam classifier')
input_sms=st.text_input("enter the Message : ")
if st.button('Predict'):
    #1 preprocessing
    transformed_sms=transform_text(input_sms)
    #2 vectorize
    vector_input=tfidf.transform([transformed_sms])
    #3 predict
    result=model.predict(vector_input)[0]
    #4 Display
    if result == 1:
        st.header("spam")
    else:
        st.header("not spam")

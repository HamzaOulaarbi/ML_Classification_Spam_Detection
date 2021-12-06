import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


st.title("""Spam Prediction APP""")
st.write('Cette application vous permettra de savoir si le message introduit est un spam ou pas')
st.write('Entrer votre message ensuite cliquer sur Submit')
st.sidebar.header('Inputs')
input=st.sidebar.text_area(label='Please enter your message here :',value='Hi Hamza,\n Click here or call us by phone on: 0800080121 if you want to earn  millions $$$ of dollars\nRegards', height=200)
input=pd.Series(input)

# Modelling with CountVectorizer,adding 3 features and LogisticRegression
def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')
#Processing
vect=pickle.load(open('my_vectorizer.pkl','rb'))
# add features as training data
Xtraintrans=vect.transform(input)
if st.sidebar.button('Submit'):
    Xtraintrans_add1=add_feature(Xtraintrans,input.str.len())
    Xtraintrans_add2=add_feature(Xtraintrans_add1, input.str.findall(r'\d').str.len())
    Xtraintrans_add3=add_feature(Xtraintrans_add2, input.str.findall(r'\W').str.len())

    # Reads in saved classification model
    load_clf = pickle.load(open('LR_clf.pkl', 'rb'))

    #Predictions
    prediction=load_clf.predict(Xtraintrans_add3)
    prediction_proba = load_clf.predict_proba(Xtraintrans_add3)
    prediction_proba=pd.DataFrame(prediction_proba,columns=['Not a spam','Spam'])
    st.subheader('Prediction : ')
    mydict={1:'Attention le message introduit est un **Spam**',
            0: '''Le message introduit n'est pas un Spam'''}
    st.write(mydict[prediction[0]])

    st.subheader('Prediction Probability')
    st.write(prediction_proba)
    print(prediction)

import streamlit as st
import pickle
import time

st.title('Text Emotion Analysis')


# load the model
model = pickle.load(open('ed.pkl', 'rb'))

text = st.text_input('Enter your text here:')
st.write('You entered:', text)

submit = st.button('Predict')

if submit:
    start = time.time()
    prediction = model.predict([text])
    end = time.time()
    st.write('Prediction time taken: ', round(end-start, 2), 'seconds')
    
    print(prediction)
    st.write(prediction)

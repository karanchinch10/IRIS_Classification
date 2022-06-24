import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image


# html_f = """
# <div style="background-color: skyblue;padding: 10px">
#     <h2 style="color: black;text-align:center;font-weight: bold">IRIS Flowers Prediction App</h2>
# </div>
# """
#st.markdown(html_f,unsafe_allow_html=True)



model = pickle.load(open('model.pkl','rb'))
st.title("IRIS Flowers Prediction App")
st.image("flower.jpg")

def user_report():
    sl = st.slider('Sepal Length',1.0,10.0,0.1)
    sw = st.slider('Sepal Width',1.0,6.0,0.1)
    pl = st.slider('Petal Length',1.0,10.0,0.1)
    pw = st.slider('Petal Width',1.0,4.0,0.1)

    user_report_data = {
        'SepalLengthCm' : sl,
        'SepalWidthCm' : sw,
        'PetalLengthCm' : pl,
        'PetalWidthCm' : pw,
    }
    res = pd.DataFrame(user_report_data,index=[0])
    return res



d = user_report()
st.subheader('Input Data')
st.write(d)


y_perd = model.predict(d)

result = st.button("Predict")
if result:
    st.success('Predicted Flower  :  {} '.format(y_perd[0]))



# ht = """
# <style> 
# #MainMenu {visibility: hidden;
# footer {visibility: hidden;}
# </style>
# """
# st.markdown(ht,unsafe_allow_html=True)


st.write("made by karan chinchpure")


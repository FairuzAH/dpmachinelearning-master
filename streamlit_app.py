import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title('Machine Learning App')

st.write('This is a Machine Learning Application')

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
  df

  st.write('**X**')
  x_raw = df.drop('species', axis=1)
  x_raw

  st.write('**Y**')
  y_raw = df.species
  y_raw

with st.expander('Data visualization'):
  st.scatter_chart(data=df, x= 'bill_length_mm', y='body_mass_g', color='species')  
#input features
with st.sidebar:
  st.header('Input Features')
  #"species","island","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g","sex"
  island = st.selectbox('Island',{'Biscoe','Dream', 'Togerseon' })
  bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
  bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
  flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
  body_mass_g = st.slider('Body mass in (g)', 2700,6300, 4200)
  sex = st.selectbox('Ses',{'Male','Female' })

  #create a datafreame for the input features
  data = {'island': island,
          'bill_length_mm': bill_length_mm,
          'bill_depth_mm': bill_depth_mm,
          'flipper_length_mm': flipper_length_mm,
          'body_mass_g': body_mass_g,
          'sex': sex}
  input_df = pd.DataFrame(data, index=[0])
  input_penguins = pd.concat([input_df, x_raw ], axis=0)
  
with st.expander('Input features'):
  st.write('**Input Penguin**')
  input_df
  st.write('**Combined penguins data**')
  input_penguins
  
#data prap
# encode x(s)
encode = ['island', ' gender']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)

X = df_penguins[1:]
input_row= df_penguins[:1]

# encode y
target_mapper = {'Adelie': 0,
                'Chinstrap':1,
                "Gentoo":2}
def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('Data preparation'):
  st.write('**Encoded x**')
  input_row
  st.write('**Encoded y**')
  y

#model training
## train ml model
clf = RandomForestClassifier()
clf.fit(X, y) 

## apply model to make prediction
prediction = clf.predict(input_row)
prediction_prob = clf.predict_proba(input_row)

df_prediction_prob = pd.DataFrame(prediction_prob)
df_prediction_prob.columns = ('Adelie', 'Chinstrap', 'Gentoo')
df_prediction_prob.rename(columns= {0: 'Adelie', 
                                1: 'Chinstrap', 
                                2: 'Gentoo'})

# display predictied species
# st.subheader('Predicted Species')
# penguin_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
# st.success(str(penguins_species(prediction[0])))


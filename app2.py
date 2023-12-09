#!/usr/bin/env python
# coding: utf-8

# In[1]:



import streamlit as st   # parameters not required.
from tensorflow.keras.models import load_model
import pickle
from keras_preprocessing.sequence import pad_sequences


# In[2]:


tokenizer = None

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


# In[59]:


model = load_model('trained_reviews_model2.h5')


# In[60]:


#st.title ("Amazon Review Neural Network Predictor", '')
#st.header("Please write your review")


# In[61]:


product = st.text_input('Product Name')
#st.write('Your product is', product)


# In[62]:


review = st.text_input('Your Review')
#st.write('Your Review is', review)


# In[63]:


#tokenizer.fit_on_texts(review)
sequences = tokenizer.texts_to_sequences([review])


# In[64]:


max_seq_length = max(len(seq) for seq in sequences)


# In[69]:


padded_sequences = pad_sequences(sequences, maxlen=476)


# In[70]:


print(padded_sequences)
X = padded_sequences
prediction = model.predict(X)


# In[71]:


st.write(f'Your rating Review Value According to Neural Network is: {prediction}')

#st.write(f'The Best Price RandomForest Will Be: {round(pred_forest[0])} Eur')


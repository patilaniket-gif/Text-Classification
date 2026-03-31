#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv("spam.csv")
df.head(5)


# In[3]:


#Describe spam and not spam
df.groupby("Category").describe()


# In[4]:


#Convert in number
df["spam"]=df["Category"].apply(lambda x: 1 if x=="spam" else 0)
df.head(5)


# In[6]:


#convert in train and test
x_train, x_test, y_train,  y_test=train_test_split(df["Message"],df["spam"],test_size=0.2, stratify=df["spam"])


# In[7]:


len(x_train)


# In[8]:


len(x_test)


# In[9]:


#URl for preprocessor bert
bert_preprocess=hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")

#URl for encoder bert
bert_encoder=hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")


# In[10]:


#embeding the sentence 
def get_sentence_embeding(sentences):
    preprocessed_text=bert_preprocess(sentences)
    return bert_encoder(preprocessed_text)["pooled_output"]


# In[11]:


get_sentence_embeding([
  "Hello Lonnie,\n\nJust wanted to touch base reg..." ,
    "Congratulations, you've won a prize! Call us n...	"
])


# In[12]:


#cosine learning
e=get_sentence_embeding([
    "banana","grapes",
    "mango","jeff bezos",
    "elon musk"
])


# In[13]:


from sklearn.metrics.pairwise import cosine_similarity


# In[14]:


cosine_similarity([e[3]],[e[4]])


# In[15]:


#model Building
text_input=tf.keras.layers.Input(shape=(),dtype=tf.string, name="text")
preprocessed_text=bert_preprocess(text_input)
outputs=bert_encoder(preprocessed_text)

# layer adding
#1st-Dropout
l1=tf.keras.layers.Dropout(0.3, name="dropout")(outputs["pooled_output"])
#2nd=Dense
l2=tf.keras.layers.Dense(1, activation="sigmoid", name="output")(l1)

model = tf.keras.Model(inputs=[text_input],outputs=[l2])
model.summary()


# In[16]:


#compiling the model
model.compile(optimizer="adam",
             loss="binary_crossentropy",
             metrics=["accuracy"])


# In[19]:


model.fit(x_train,y_train, epochs=5)


# In[20]:


model.evaluate(x_test,y_test)


# In[42]:


Reviews = [
    "Just forced myself to eat a slice. I'm really not hungry tho. This sucks. Mark is getting worried. He knows I'm sick when I turn down pizza. Lol",
    "I'm back &amp; we're packing the car now, I'll let you know if there's room",
    "As a valued customer, I am pleased to advise you that following recent review of your Mob No. you are awarded with a £1500 Bonus Prize, call 09066364589",
    "Please call our customer service representative on 0800 169 6031 between 10am-9pm as you have WON a guaranteed £1000 cash or £5000 prize!",
    "Hi. Wk been ok - on hols now! Yes on for a bit of a run. Forgot that i have hairdressers appointment at four so need to get home n shower beforehand. Does that cause prob for u?"
]


# In[51]:


for i in range(len(Reviews)):
    Prediction = model.predict([Reviews[i]])
    output=Prediction[0][0]
    print(f"Review:  {Reviews[i]}")
    
    if output < 0.5:
        print(f"Result: Not Spam (Score: {output})/n")
    else:
        print(f"Result: Spam (Score: {output})/n")


# In[ ]:





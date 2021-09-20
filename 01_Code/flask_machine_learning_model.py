#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from flask import Flask, render_template,request
from joblib import load
from flask import Flask


# In[2]:


app = Flask(__name__)

@app.route("/",methods=["GET","POST"])
def hello_world():
    request_type_str=request.method
    if request_type_str=="GET":
        return  render_template('index.html')
    else:
        text=request.form["text"]
        model_in=load("model.joblib")
        prediction=model_in.predict([text])[0]
        return str(prediction)
    
if __name__ == '__main__':
    app.run(host="localhost", port=8000, debug=True)


# In[ ]:





# In[ ]:





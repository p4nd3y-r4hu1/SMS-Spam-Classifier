#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd


# In[8]:


import numpy as np


# In[9]:


from matplotlib import  pyplot as plt


# In[10]:


df=pd.read_csv("spam.csv",encoding="ISO-8859-1")


# In[11]:


df.head()


# In[12]:


df.shape


# In[13]:


v1=df["v1"]


# In[14]:


def match(s):
    if (s=="ham"):
        return 0;
    if(s=="spam"):
        return 1;
after_match=map(match,v1)    


# In[15]:


after_match


# In[16]:


df["v1"]=list(after_match)


# df
# 

# In[17]:


df


# # # Data Cleaning 
# 

# In[18]:


df.info()


# In[19]:


df1=df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'])


# In[20]:


df1


# In[21]:


#Renaming the columns
df1.rename(columns={'v1':'target','v2':'text'},inplace=True)


# In[22]:


df1


# In[23]:


import sklearn.preprocessing


# In[24]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df1['target']=encoder.fit_transform(df1['target'])


# In[25]:


df1


# In[26]:


df1.isnull().sum()


# In[27]:


df1.duplicated().sum()


# In[28]:


# remove duplicacy
df1=df1.drop_duplicates(keep='first')


# In[29]:


df1.duplicated().sum()


# In[30]:


df1.info()


# In[31]:


df1.isnull()


# In[32]:


df1


# Eda
# 

# In[33]:


df1['target'].value_counts()


# In[34]:


plt.pie(df1['target'].value_counts(),labels=['ham','spam'],autopct='%0.2f')
plt.show()


# In[35]:


import nltk


# In[36]:


nltk.download('punkt')


# In[37]:


df1['num_characters']=df1['text'].apply(len)


# In[38]:


df1.head()


# In[39]:


df1['word_length']=df1['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[40]:


df1['word_length']


# In[41]:


df1.head()


# In[42]:


df1.rename(columns={'word_length':'num_words'})


# In[43]:


df1.head()


# In[44]:


df1['num_sentences']=df1['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[45]:


df1


# In[46]:


df1.rename(columns={'word_length':'num_sentences'})


# In[47]:


df1


# In[48]:


df1[df1['target'] == 0][['num_sentences','num_characters','word_length']].describe()


# In[49]:


df1[df1['target'] == 1][['num_sentences','num_characters','word_length']].describe()


# In[50]:


import seaborn as sns


# In[51]:


df1


# In[52]:


sns.histplot(df1[df1['target']==0]['num_characters'])
sns.histplot(df1[df1['target']==1]['num_characters'],color='green')


# In[53]:


sns.pairplot(df1,hue='target')


# In[54]:


sns.heatmap(df1.corr(),annot=True)


# In[82]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# In[83]:


import string  


# In[86]:


def text_transform(text):
    text=text.lower()
    
    text=nltk.word_tokenize(text)
    
    y=[]
    
    for i in text:
        if i.isalnum():
            y.append(i)
    
    
    text=y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text=y[:]
    y.clear()
    
    
    for i in text:
        y.append(ps.stem(i));
    
    
    return " ".join(y)
    
    


# In[88]:


text_transform("HI How are running You 34")


# In[81]:


from nltk.stem.porter import PorterStemmer 
ps=PorterStemmer()
ps.stem('running')


# In[90]:


df1['text_transformed']=df1['text'].apply(text_transform)


# In[91]:


df1.head()


# In[94]:


pip install wordcloud


# from wordcloud import WordCloud
# wc= WordCloud(width=500,height=500,min_font=15,background='white')

# In[104]:


from wordcloud import WordCloud
wc= WordCloud(width=500,height=500,min_font_size=15,background_color='white')


# In[108]:


wc_spam=wc.generate(df1[df1['target']==1]['text_transformed'].str.cat(sep=""))


# In[109]:


plt.figure(figsize=(15,6))
plt.imshow(wc_spam)


# In[110]:


wc_spam=wc.generate(df1[df1['target']==0]['text_transformed'].str.cat(sep=""))
plt.figure(figsize=(15,6))
plt.imshow(wc_spam)


# # Model Training
# 

# In[112]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x=cv.fit_transform(df1['text_transformed']).toarray()


# In[113]:


x


# In[116]:


y=df1['target'].values


# In[117]:


y


# In[118]:


from sklearn.model_selection import train_test_split


# In[119]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[125]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix
gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()


# In[126]:


gnb.fit(x_train,y_train)
predict1=gnb.predict(x_test)
print(accuracy_score(y_test,predict1))
print(precision_score(y_test,predict1))
print(confusion_matrix(y_test,predict1))


# In[127]:


mnb.fit(x_train,y_train)
predict2=mnb.predict(x_test)
print(accuracy_score(y_test,predict2))
print(precision_score(y_test,predict2))
print(confusion_matrix(y_test,predict2))


# In[128]:


bnb.fit(x_train,y_train)
predict3=bnb.predict(x_test)
print(accuracy_score(y_test,predict3))
print(precision_score(y_test,predict3))
print(confusion_matrix(y_test,predict3))


# In[ ]:





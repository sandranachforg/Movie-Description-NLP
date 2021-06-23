#!/usr/bin/env python
# coding: utf-8

# ### Movie Description Analysis with NLP

# #### 1. Step: Import libraries and dataset

# In[201]:


import matplotlib.pyplot as plt # data visualization
import itertools #to connect a list of lists to one 

import pandas as pd #data manipulation

import nltk  #natural language processing
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords #and, #the,  #by


# In[202]:


data = pd.read_csv("IMDb movies.csv")


# In[203]:


data.head()


# In[204]:


df= data[["original_title", "genre", "duration", "description", "avg_vote"]]

df.head()


# In[205]:


df.dropna(inplace = True)


# In[206]:


df.isnull().sum()


# #### 2. Preprocessing of textual data 

# In[212]:



stop = stopwords.words('english') #set stopwords to english

wordnet_lemmatizer = WordNetLemmatizer()  #initialize the WordNetLemmatizer

def create_tokens(df):
    
    # create a column where we make all words lowercase
    df["lower_case"]= df["description"].str.lower()
    
    #create a column where we remove the stopwords from descriptions
    df["no_stopwords"] = df["lower_case"].apply(lambda x:' '.join([word for word in x.split() if word not in (stop)]))
    
    # create a column where we tokenize the description column
    df['tokenized'] = df.apply(lambda row: nltk.word_tokenize(row['no_stopwords']), axis=1)
    
    # create a column where we only keep tokens that are actual english words
    df['only_alphas'] = df['tokenized'].apply(lambda x: [word for word in x if word.isalpha()])
    
    # create a column where we lemmatize (reduce to its roots) the words present in 'only_alphas'
    df['lemmatized'] = df['only_alphas'].apply(lambda x: [wordnet_lemmatizer.lemmatize(word) for word in x])
    
    
    #return the dataframe
    
    return df


# In[215]:


df_clean.head()


# In[213]:


# apply function
df_clean = create_tokens(df)


# In[216]:


#create a function that splits a string into words and then returns the length of words

def word_count(string):
    words = string.split()
    
    return len(words)


# In[217]:


#test the function before applying it to the whole column

word_count("hello how are you?")


# In[218]:


# create a function that splits a word into seperate letters and then counts the average word length 

def avg_word_length (x):
    
    words = x.split()
    
    word_lenghts= [len(word) for word in words]
    
    avg_word_length = sum(word_lenghts)/len(words)
    
    return (avg_word_length)


# In[219]:


#test the function before applying it to the whole column

avg_word_length('hello')


# In[220]:


# use the apply function to apply the function you created to count words in each description

df_clean ["num_words"] = df_clean['description'].apply(word_count)

# use apply function
df_clean ['avg_word_length'] = df_clean['description'].apply(avg_word_length).round(0)

#calculate the number of stopwords by subtracting the column with no stopwords from the column containing stopwords
df_clean ['number_stopwords'] = df_clean['description'].apply(word_count) - df_clean['no_stopwords'].apply(word_count)


# In[221]:


df_clean.head()


# #### 3. Which words are most common in the description of dramas that have a higher rating than 7.0 ?

# In[222]:


#select only data where genre is drama

drama = df_clean[(df_clean["genre"]== 'Drama') & (df_clean["avg_vote"]> 7.0)]

drama.head()


# In[225]:


a = drama.lemmatized.to_list()  #transfrom the lemmatized column to a list 

b = (list(itertools.chain.from_iterable(a)))      #connect the list of lists 

bow = Counter(b)    #apply the Counter function on the long list


print(dict(bow.most_common(10)))      #apply the most_common function on the bag of words

drama_words = dict(bow.most_common(10))       #transform this list to a dictionary


# In[226]:


keys = drama_words.keys()   #specify the x - axis values
values = drama_words.values()  #specify the y- axis values


fig, ax = plt.subplots()

ax = plt.bar(keys, values)

plt.xticks(rotation=45)

plt.show()


# #### 4. Which words are most common in the description of horror movies where the rating is > 6.0 ?

# In[227]:


#select only data where genre is horror

horror= df_clean[(df_clean["genre"]== "Horror") & (df_clean["avg_vote"] > 6.0)]


# In[228]:


horror.head()


# In[231]:


a = horror.lemmatized.to_list()

b = (list(itertools.chain.from_iterable(a)))

bow = Counter(b)

print(dict(bow.most_common(10)))

horror_words = dict(bow.most_common(10))


# In[232]:


keys = horror_words.keys()   #specify the x - axis values
values = horror_words.values()  #specify the y- axis values


fig, ax = plt.subplots()

ax = plt.bar(keys, values)

plt.xticks(rotation=45)

plt.show()


# #### 4. Which genres tend to have the longest descriptions? Does duration of movies impact the length of the description?

# In[233]:


df_clean.head()


# In[238]:


#use a groupby statement to find the average number of words in descriptions of each genre

df_clean.num_words.groupby(df_clean['genre']).mean().round(0).sort_values(ascending = False).head(5)


# In[236]:


# oddly Mystery, Romance, have a 2.0 rating. 
df_clean[df_clean['genre'] == 'Mystery, Romance']

#df_clean[df_clean['genre'] == 'Drama, Musical, Thriller']


# Answer: Based on our findings, Family, Romance, Sci-Fi have on average the longest descriptions. But are they on average also longer in duration?

# In[237]:


df_clean.duration.groupby(df_clean['genre']).mean().round(0).sort_values(ascending = False).head(5)


# Answer: No, Family, Romance, Sci-Fi are not in the top 5 with the longest duration

# In[ ]:





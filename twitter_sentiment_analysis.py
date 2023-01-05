
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re 
import string
import wordcloud

import nltk
from nltk.stem.porter import PorterStemmer

import warnings

# %matplotlib inline
warnings.filterwarnings('ignore')

"""
## Loading the Dataset"""

# importing dataset 

df = pd.read_csv('dataset.csv')
df = df.drop('id', axis=1)

# shape of the dataset 

df.shape

df.head()

df.info()

df.describe()

"""## Preprocessing the Dataset


"""

# checking for any NaN Value

df.isnull().any()

# checking for any Duplicate Value

df.duplicated().any()

# Removing duplicates of Dataset.

df.drop_duplicates(subset=['tweet'], keep='last', inplace=True)
print("Shape of dataset after removing duplicates:", df.shape)

# resetting Index after deletion of duplicates

df = df.reset_index()
df = df.drop('index', axis=1)
df.head()

# removes 'pattern' in the input text

def remove_pattern(input_txt, pattern):
  r = re.findall(pattern, input_txt)         # finding all same patterns 
  for word in r:
    input_txt = re.sub(word, "", input_txt)  # substituting pattern with ""
  return input_txt

print("Before Cleaning the data: ")
df.head()

# remove twitter handles (@user)

df['tweet'] = np.vectorize(remove_pattern)(df['tweet'], "@[\w]*")
df.head()

# remove special characters, numbers and punctuations.

df['tweet'] = df['tweet'].str.replace("[^a-zA-Z#]", " ")
df.head()

# removing short words (less than 3 alphabets)

df['tweet'] = df['tweet'].apply(lambda x: " ".join([w for w in x.split() if len(w) > 3]))
df.head()

# Function to remove Hash from HashTags

    # def remove_hash(tweet):
    #     tweet = re.sub(r'#','',tweet)
    #     return tweet 

    # df['tweet'] = df['tweet'].apply(remove_hash)

# Tokenization (Individual words are considered as tokens)

tokenized_tweet = df['tweet'].apply(lambda x: x.split())
tokenized_tweet.head()

# Stem the words
# converting forms of verb (play/playing/played) as a common word

stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda sentence: [stemmer.stem(word) for word in sentence])
tokenized_tweet.head()

# combine words into "Single Sentence"

for i in range(len(tokenized_tweet)):
  tokenized_tweet[i] = " ".join(tokenized_tweet[i])

df['tweet'] = tokenized_tweet
df.head()

"""## Exploratory Data Analysis"""

# checking out the NEGATIVE comments from the train set

df[df['label'] == 0].head(10)

# checking out the POSITIVE comments from the train set 

df[df['label'] == 1].head(10)

# visualize the frequent words

all_words = " ".join([sentence for sentence in df['tweet']])

from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

# plot the graph
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('All Frequent Words')
plt.axis('off')
plt.show()

# frequent words visualization for +ve ( labelled = 0 )

all_words = " ".join([ sentence for sentence in df['tweet'][df['label']==0] ])

wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Positive Words')
plt.axis('off')
plt.show()

# frequent words visualization for -ve (labelled = 1)

all_words = " ".join([sentence for sentence in df['tweet'][df['label']==1 ]])

wordcloud = WordCloud(width=800, height=500, max_font_size=100, random_state=42).generate(all_words)

plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Negative Words')
plt.axis('off')
plt.show()

# Extract the hashtag

def hashtag_extract(tweets):
  hashtags = []
  for tweet in tweets:
    ht = re.findall(r"#(\w+)", tweet)
    hashtags.append(ht)
  return hashtags

# extract hashtags from non-racist/sexist tweets
ht_positive = hashtag_extract(df['tweet'][df['label'] == 0])

# extract hashtags from racist/sexist tweets
ht_negative = hashtag_extract(df['tweet'][df['label'] == 1])

ht_positive[:5]

ht_negative[:5]

# un-nest list

ht_positive = sum(ht_positive, [])
ht_negative = sum(ht_negative, [])

ht_positive[:5]

ht_negative[:5]

# checking Frequencies of 'Positive' Hashtags

freq = nltk.FreqDist(ht_positive)
d = pd.DataFrame({'Hashtag': list(freq.keys()),
                    'Count': list(freq.values())})
d.head()

# select top 10 +ve hashtags

d = d.nlargest(columns = 'Count', n=10)
plt.figure(figsize=(15,9))
sns.barplot(data = d, x = 'Hashtag', y = 'Count')
plt.show()

# checking Frequencies of 'Negative' Hashtags


freq = nltk.FreqDist(ht_negative)
d = pd.DataFrame({'Hashtag': list(freq.keys()),
                  'Count': list(freq.values())})
d.head()

# select top 10 -ve hashtags

d = d.nlargest(columns='Count', n=10)
plt.figure(figsize=(15,9))
sns.barplot(data=d, x='Hashtag', y='Count')
plt.show()

# To Store cleaned dataframe in csv file

# df.to_csv(r'dataset_cleaned.csv', index=False)

"""## Input Split

"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score

"""### Feature Extraction & Train-Test Split of Dataset """

df = pd.read_csv('Twitter Sentiment Cleaned.csv')

X = df['tweet'].astype(str) 
y = df['label'].astype(str) 
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.2, random_state = 42)

vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(X_train)
print('No. of feature_words: ', len(vectoriser.get_feature_names()))

X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)

"""## Machine Learning Models"""

models = {
    
    'Logistics_Regression' :{
        'model' : LogisticRegression(multi_class = 'auto'),
        'parameters' : {
            'C' : [90], 'solver' : ['lbfgs']
        }
    },
    
    'SVC' :{
        'model' : SVC(),
        'parameters' : {
            'C': [10], 'gamma': [1], 'kernel': ['linear']
        }
    },

    'MultinomialNB' :{
        'model' : MultinomialNB(),
        'parameters' : {
            'alpha' : np.linspace(0.5,1), 'fit_prior' : [True]
        }
    },
    
     'Decision_Tree' :{
        'model' : DecisionTreeClassifier(),
         'parameters' : {
            
        }

    },
    
    'XGBClassifier' :{
        'model' : XGBClassifier(),
        'parameters' : {
            
        }
           
    },
    
    'Random_Forest' :{
        'model' : RandomForestClassifier(),
        'parameters' : {
            'n_estimators' : [95], 
            'max_depth':[None], 'criterion':['gini']
        }
    }
}

score = []

for model_name, mp in models.items():
    clf = GridSearchCV(mp['model'], mp['parameters'], cv=5, n_jobs=-1) # Using Cross Validation of 5 and n_jobs=-1 for fast training by using all the processors
    print(mp['model'])
    print('\nFitting...')
    best_model = clf.fit(X_train, y_train)                      # Training the model
    clf_pred = best_model.predict(X_test)                       # Predicting the results
    print(confusion_matrix(y_test,clf_pred))                    # Printing Confusion Matrix
    print(metrics.classification_report(y_test, clf_pred))      # Printing Classification Report
    score.append({                                              # Appending results to 'scores' list
        'model' : model_name,
        'best_accuracy' : best_model.score(X_test, y_test),
        'best_parameters' : clf.best_params_
    })
    print('\nThe score is appended to the list...\n')

# Creating DataFrame with model, best_accuracy and best_parameters:
res = pd.DataFrame(score, columns=['model', 'best_accuracy', 'best_parameters'])
res

"""#### Best Accuracy - SVC ( 0.963596 )"""


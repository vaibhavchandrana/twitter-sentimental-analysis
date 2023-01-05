# Twitter Sentiment Analysis - Project

Twitter is a popular microblogging service where users create status messages known as tweets. These tweets sometimes express opinions on various topics.

## Objective:

The goal of this project is to create an algorithm that can correctly classify Twitter messages as positive or negative based on a query term. Our hypothesis is that using machine learning techniques, we can achieve high accuracy in classifying sentiment in Twitter messages.

## What is the need of sentiment analysis?

In general, this type of sentiment analysis is useful for consumers researching a product or service, as well as marketers researching public opinion of their company.

## Requirments:

1. Notebook( Colab/ Jupyter ) 
2. Libraries reqired
    1. numpy
    2. pandas
    3. matplotlib
    4. seaborn
    5. re ( for regular expression operations )
    6. string
    7. wordcloud
    8. nltk ( natural language tool-kit)
    9. scikit-learn ( for machine learning )
    

## Procedure:

### Data Collection

I gathered the labbeled dataset, which is available on kaggle. It contains approximately 30k data entries with labels 0 and 1.

> Not Hate-Speech( Postive ) → 0
> 

> Hate-Speech( Negative ) → 1
> 

### Data Cleaning

1. Checking for any NULL or DUPLICATE  values & delete it. Reset index.
2. Remove PATTERNS from each tweet like following:
    1. remove twitter handles (@user)
    2. remove special characters, numbers and punctuations
    3. removing short words (less than 3 alphabets)
    4. removing hashtags(#) , although it doesn't impact accuracy.
3. Tokenization - Tokenization is used in natural language processing **to split paragraphs and sentences into smaller units that can be more easily assigned meaning**. (Individual words are considered as tokens)
4. Stemming - Stemming is **the process of reducing a word to its stem that affixes to suffixes and prefixes or to the roots of words known as "lemmas". In simple words, converting forms of verb (play/playing/played) as a common word**
5. Combine words into a "Single Sentence”.

### Exploratory Data Analysis

1. Analysis of frequency of  all words.
2. An analysis of the frequency of positive and negative words, as well as the most frequently occurring word in each.
3. Visualizing the data by using wordcloud and bar charts.
4. Extracting hashtags and analysing them, as the hashtags describe the emotion/sentiment best in one word.

### Machine Learning Approaches

1. Feature Scaling - Feature Scaling is **a technique to standardize the independent features present in the data in a fixed range.**
    
    For this project i have used TfidfVectorizer - Transforms text to feature vectors that can be used as input to estimator. vocabulary_ Is a dictionary that converts each token (word) to feature index in the matrix, each unique token gets a feature index.
    
    <img src="https://assets.datacamp.com/production/repositories/3752/datasets/e5ef37fe9dfe7e5877c92b51486aa2d3f5ed8449/TFIDF.png" style="width:500px; heigth:300px"></img>
    
    
    And spliting dataset into ‘training dataset’ and ‘testing dataset’.
    
2. Logistic Regression Classification - 
    
    **Logistic Regression** is a statistical approach and a Machine Learning algorithm that is used for classification problems and is based on the concept of probability. It is used when the dependent variable (target) is categorical. It is widely used when the classification problem at hand is binary; true or false, yes or no, etc.
    
    Logistics regression uses the sigmoid function to return the probability of a label.
    
      <img src="https://miro.medium.com/max/1400/1*KZQYpR-aWsSF2Zl7JFRI5A.png" style="width:500px; heigth:300px"></img>
    
3. Support Vector Classification (SVC) - 
    
    Support Vector Machine or SVM is one of the most popular Supervised Learning algorithms, which is used for Classification as well as Regression problems. However, primarily, it is used for Classification problems in Machine Learning.
    
    The goal of the SVM algorithm is to create the best line or decision boundary that can segregate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a hyperplane.
    
      <img src="https://ars.els-cdn.com/content/image/3-s2.0-B978032385214200001X-f06-02-9780323852142.jpg" style="width:500px; heigth:300px"></img>
    
4. MultinomialNB (Navie Bayes) - 
    
    Multinomial Naive Bayes algorithm is **a Bayesian learning approach popular in Natural Language Processing (NLP)**. 
    
    The program guesses the tag of a text, such as an email or a newspaper story, using the Bayes theorem. It calculates each tag's likelihood for a given sample and outputs the tag with the greatest chance.
    
5. Decision Tree Classification - 
    
    Decision Tree is a **Supervised learning technique** that can be used for both classification and Regression problems, but mostly it is preferred for solving Classification problems. 
    
    It is a tree-structured classifier, where **internal nodes represent the features of a dataset, branches represent the decision rules** and **each leaf node represents the outcome.**
    
      <img src="https://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1545934190/1_r5ikdb.png" style="width:500px; heigth:300px"></img>
    
6. XGB Classifier - 
    
    The XGBoost or Extreme Gradient Boosting algorithm is **a decision tree based machine learning algorithm which uses a process called boosting to help improve performance.**
    
      <img src="https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/Ensemble-algorithms-boosting.png?ssl=1" style="width:500px; heigth:300px"></img>
    
7. Random Forest Classification - 
    
    The random forest is **a classification algorithm consisting of many decisions trees**. It uses bagging and feature randomness when building each individual tree to try to create an uncorrelated forest of trees whose prediction by committee is more accurate than that of any individual tree.
    
      <img src="https://miro.medium.com/max/1052/1*VHDtVaDPNepRglIAv72BFg.jpeg" style="width:500px; heigth:300px"></img>
    
    
<br/>

## Results:

| model | best_accuracy | best_parameters |
| --- | --- | --- |
| Logistics_Regression | 0.962749746 | {'C': 90, 'solver': 'lbfgs'} |
| SVC | 0.963596343 | {'C': 10, 'gamma': 1, 'kernel': 'linear'} |
| MultinomialNB | 0.937521165 | {'alpha': 0.5, 'fit_prior': True} |
| Decision_Tree | 0.944801896 | {} |
| XGBClassifier | 0.947680325 | {} |
| Random_Forest | 0.958686082 | {'criterion': 'gini', 'max_depth': None, 'n_estimators': 95} |

<br/>

## Alternatives:

**For Datasets we can also use**

1. **[tweepy](https://docs.tweepy.org/en/stable/)** - An easy-to-use Python library for accessing the Twitter API.
2. **[nltk.corpus package](https://www.nltk.org/api/nltk.corpus.reader.twitter.html) -** nltk.corpus.reader.twitter module

**For Feature Scaling**

1. CountVectorizer - CountVectorizer means **breaking down a sentence or any text into words by performing preprocessing tasks like converting all words to lowercase, thus removing special characters**. In NLP models can't understand textual data they only accept numbers, so this textual data needs to be vectorized.

```python
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(df['tweet'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(bow, df['label'], test_size=0.3, random_state= 42)
```
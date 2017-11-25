
# coding: utf-8

# In[399]:

print("CKME 136: Capstone Course: Data Analytics, Big Data & Predictive Modeling.")  
print("Sentiment Analysis of Tweets for the US Airline Industry: A classification approach")
print("Donald T. Lane")


# In[400]:

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import Counter


# In[401]:

raw_data = pd.read_csv('D:/COURSE_TEMP\capstone/airline_sentiment/Airline_twitter_data/tweets.csv')


# In[402]:

raw_data.shape


# In[403]:

raw_data.head()


# In[404]:

raw_data.describe(include='all').T


# In[405]:

raw_data[0:3]


# In[406]:

raw_data.drop(['tweet_id','negativereason', 'airline_sentiment_confidence', 'negativereason_confidence', 'airline', 'airline_sentiment_gold','negativereason_gold', 'name', 
               'retweet_count', 'tweet_coord', 'tweet_created', 'tweet_location', 'user_timezone'], 
              axis = 1, inplace = True)


# In[407]:

raw_data.shape


# In[408]:

raw_data[0:5]


# In[409]:

#Convert text to lowercase in text field
raw_data['text'] = raw_data.text.str.lower()


# In[410]:

raw_data[0:4]


# In[411]:

raw_data['norm_text'] = raw_data['text']


# In[412]:

raw_data[0:4]


# In[413]:

#Clean up the dataset and prepare it for classifcation


# In[414]:

#Remove the @User
raw_data['norm_text']= raw_data.norm_text.str.replace('\@[a-z0-9]+', ' ')


# In[415]:

raw_data[0:4]


# In[416]:

# Remove Stopwords
from nltk.corpus import stopwords
stop = stopwords.words("english")


# In[417]:

raw_data['norm_text'] = raw_data['norm_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# In[418]:

raw_data[0:4]


# In[419]:

#Remove URLs
raw_data['norm_text'] = raw_data.norm_text.str.replace('https?:\/\/.*[\r\n]*',' ')
raw_data['norm_text'] = raw_data.norm_text.str.replace('http:\/\/.*[\r\n]*',' ')
raw_data['norm_text'] = raw_data.norm_text.str.replace('http',' ')


# In[420]:

raw_data[0:4]


# In[421]:

#Remove the Hashtags
raw_data['norm_text'] = raw_data.norm_text.str.replace('\#',' ')


# In[422]:

raw_data[0:10]


# In[423]:

#Remove the comma
raw_data['norm_text'] = raw_data.norm_text.str.replace('\,',' ')


# In[424]:

raw_data[0:10]


# In[425]:

#Remove the semi colon
raw_data['norm_text'] = raw_data.norm_text.str.replace('\;[a-z0-9]+',' ')
raw_data['norm_text'] = raw_data.norm_text.str.replace(';',' ')                                                    


# In[426]:

#Remove special characters without removing apostraphes
pattern=re.compile("[^\w']")
raw_data['norm_text'] = raw_data.norm_text.str.replace(pattern,' ')


# In[427]:

raw_data[0:10]


# In[428]:

#Remove the colon
raw_data['norm_text'] = raw_data.norm_text.str.replace('\:[a-z0-9]+',' ')
raw_data['norm_text'] = raw_data.norm_text.str.replace(':+',' ')


# In[429]:

#Remove the amphersand
raw_data['norm_text'] = raw_data.norm_text.str.replace('&',' ')


# In[430]:

raw_data[0:10]


# In[431]:

#Remove words that have 3 or less letters
shortword = re.compile(r'\b\w{1,3}\b') #r'\W*\b\w{1,3}\b'
raw_data['norm_text'] = raw_data.norm_text.str.replace(shortword,' ')


# In[432]:

raw_data['norm_text'] = raw_data.norm_text.str.replace('\'','')


# In[433]:

raw_data['norm_text'] = raw_data.norm_text.str.replace('didn',"didn't")


# In[434]:

raw_data[0:10]


# In[435]:

#Remove periods
raw_data['norm_text'] = raw_data.norm_text.str.replace('\.+',' ')
raw_data['norm_text'] = raw_data.norm_text.str.replace('.',' ')


# In[436]:

#Remove dashes
raw_data['norm_text'] = raw_data.norm_text.str.replace('-',' ')


# In[437]:

#Remove question marks, exlamation points, dollar signs
raw_data['norm_text'] = raw_data.norm_text.str.replace('!+','')
raw_data['norm_text'] = raw_data.norm_text.str.replace('\?+','')
raw_data['norm_text'] = raw_data.norm_text.str.replace('?','')
raw_data['norm_text'] = raw_data.norm_text.str.replace('$','')
raw_data['norm_text'] = raw_data.norm_text.str.replace('(','')
raw_data['norm_text'] = raw_data.norm_text.str.replace(')','')
raw_data['norm_text'] = raw_data.norm_text.str.replace('"\w+','')


# In[438]:

raw_data['norm_text'] = raw_data.norm_text.str.replace('[0-9]+',' ')


# In[439]:

myre = re.compile(u'('
    u'\ud83c[\udf00-\udfff]|'
    u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
    u'[\u2600-\u26FF\u2700-\u27BF])+', 
    re.UNICODE)
raw_data['norm_text'] = raw_data.norm_text.str.replace(myre,' ')


# In[440]:

myre2 = re.compile(u'['
    u'\U0001F300-\U0001F64F'
    u'\U0001F680-\U0001F6FF'
    u'\u2600-\u26FF\u2700-\u27BF]+', 
    re.UNICODE)
raw_data['norm_text'] = raw_data.norm_text.str.replace(myre2,' ')


# In[441]:

#Remove extra whitespaces
raw_data['norm_text'] = raw_data.norm_text.str.replace(' +',' ')


# In[443]:

raw_data[0:20]


# In[444]:

raw_data['norm_text'] = raw_data.norm_text.str.replace('\b[^\W\d][^\W\d]+\b','')


# In[445]:

raw_data[0:4]


# In[446]:

#BACKUP OF THE RAWDATA SET AT THIS STEP
raw_data.to_csv('D:/COURSE_TEMP/capstone/airline_sentiment/phase3/int_files/airline_cleaned.csv', sep=',')


# In[447]:

import nltk
nltk.download('stopwords')


# In[448]:

# Remove Stopwords - a second time after cleaning.  Only applicable to unigrams and tf-idf
from nltk.corpus import stopwords
stop = stopwords.words("english")


# In[449]:

raw_data['norm_text'] = raw_data['norm_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# In[450]:

raw_data[0:4]


# In[451]:

#BACKUP OF THE RAWDATA SET AT THIS STEP
raw_data.to_csv('D:/COURSE_TEMP/capstone/airline_sentiment/phase3/int_files/airline_prep2.csv', sep=',')
#BACKUP OF THE TRAIN DATA SET AT TTHIS STEP
#train.to_csv('D:/COURSE_TEMP/capstone/airline_sentiment/phase3/int_files/train.csv', sep=',')
#BACKUP OF THE TEST DATA SET AT TTHIS STEP
#test.to_csv('D:/COURSE_TEMP/capstone/airline_sentiment/phase3/int_files/test.csv', sep=',')


# In[452]:

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier


# In[454]:

#Get the top 20 words for the full data set.
all_words = pd.Series([y for x in raw_data.norm_text.values.flatten() for y in x.split()]).value_counts()


# In[455]:

all_words[0:20]


# In[456]:

#Create positive, nuetral, and negative files
positive_words = raw_data[(raw_data.airline_sentiment=='positive')]
negative_words = raw_data[(raw_data.airline_sentiment=='negative')]
neutral_words = raw_data[(raw_data.airline_sentiment=='neutral')]


# In[457]:

positive_words[0:3]


# In[458]:

#Get the top 20 words for the positive tweets.
top_pos = pd.Series([y for x in positive_words.norm_text.values.flatten() for y in x.split()]).value_counts()


# In[459]:

#Get the top 20 words for the negative tweets.
top_neg = pd.Series([y for x in negative_words.norm_text.values.flatten() for y in x.split()]).value_counts()


# In[460]:

#Get the top 20 words for the neutral tweets.
top_neut = pd.Series([y for x in neutral_words.norm_text.values.flatten() for y in x.split()]).value_counts()


# In[461]:

top_pos[0:20]


# In[462]:

top_neg[0:20]


# In[463]:

top_neut[0:20]


# In[464]:

#Create word clouds for each class


# In[465]:

import pip
pip.main(['install', 'wordcloud'])
from os import path
from scipy.misc import imread
import matplotlib.pyplot as plt
import random
from wordcloud import WordCloud, STOPWORDS


# In[466]:

wordcloud = WordCloud(width = 1000, height = 500).generate(' '.join(positive_words['norm_text']))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[467]:

wordcloud = WordCloud(width = 1000, height = 500).generate(' '.join(negative_words['norm_text']))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[468]:

wordcloud = WordCloud(width = 1000, height = 500).generate(' '.join(neutral_words['norm_text']))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[469]:

#-----------Begin to perform Classification----------------#


# In[470]:

#rename dataset for classificcation using multiclass (3 classes - negative, neutral, positive)
raw_data_negneupos = raw_data
raw_data_negneupos.shape


# In[471]:

def convert_word(row):
    sentiment = []
    text = row['airline_sentiment']
    if "positive" in text:
        sentiment.append('1')
    elif "negative" in text:
        sentiment.append('-1')
    elif "neutral" in text:
        sentiment.append('0')
    return (sentiment)
raw_data_negneupos['sentiment'] = raw_data_negneupos.apply(convert_word, axis = 1)


# In[472]:

raw_data_negneupos[0:4]


# In[473]:

#Convert object datatype to integer
raw_data_negneupos['sentiment'] = raw_data_negneupos['sentiment'].apply(lambda x: ', '.join(x))


# In[474]:

raw_data_negneupos['sentiment'] = raw_data_negneupos['sentiment'].astype(str).astype(int)


# In[475]:

raw_data_negneupos['sentiment'].dtype.kind 


# In[476]:

raw_data_negneupos.shape


# In[477]:

raw_data_negneupos[0:4]


# In[ ]:




# In[478]:

#Create a dataset with only 2 classes (negative and positive)
raw_data_negpos = raw_data[(raw_data.airline_sentiment=='positive') | (raw_data.airline_sentiment=='negative')]
raw_data_negpos.shape


# In[479]:

raw_data_negpos[0:4]


# In[480]:

def convert_word(row):
    sentiment = []
    text = row['airline_sentiment']
    if "positive" in text:
        sentiment.append('1')
    elif "negative" in text:
        sentiment.append('-1')
    return (sentiment)
raw_data_negpos['sentiment'] = raw_data_negpos.apply(convert_word, axis = 1)


# In[481]:

raw_data_negpos[0:4]


# In[482]:

#Convert object datatype to integer
raw_data_negpos['sentiment'] = raw_data_negpos['sentiment'].apply(lambda x: ', '.join(x))


# In[483]:

raw_data_negpos['sentiment'] = raw_data_negpos['sentiment'].astype(str).astype(int)


# In[484]:

raw_data_negpos['sentiment'].dtype.kind 


# In[485]:

raw_data_negpos.shape


# In[486]:

raw_data_negpos[0:4]


# In[487]:

##-----Multiclass - Multinomial Naive Bayes classification - unigrams - 3000 word limit.----------##
##________________________________________________________________________________________________##


# In[488]:

#Assign applicable columns to new variable to prepare for classifier
X = raw_data_negneupos.norm_text
y = raw_data_negneupos.sentiment


# In[489]:

# split the data into training and test data sets (75:25)
from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

#split data (another technique) - tried and works - maintaining in here for reference
y_train=y.sample(frac=0.75,random_state=200)
y_test=y.drop(y_train.index)
X_train=X.sample(frac=0.75, random_state=200)
X_test=X.drop(X_train.index)


# In[490]:

X_train.shape


# In[491]:

X_test.shape


# In[492]:

y_train.shape


# In[493]:

y_test.shape


# In[494]:

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import naive_bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn import metrics


# In[495]:

# instantiate CountVectorizer
#using unigram model - most frequent 3000 words
#baseline classifier will be based on this approach - Notes from meeting with Kanchana)
stop = set(stopwords.words('english'))
vect = CountVectorizer(stop_words=stop, max_features = 3000)


# In[496]:

# fit and transform X_train into X_train_fit
X_train_fit = vect.fit_transform(X_train)
X_train_fit.shape


# In[497]:

# transform X_test into X_test_fit
X_test_fit = vect.transform(X_test)
X_test_fit.shape


# In[498]:

# import and instantiate Multinomial NB classifier
nb = MultinomialNB()


# In[499]:

#Accuracy - 10-fold cross validation on training data set.  y score for each fold. 
from sklearn.model_selection import cross_val_score
scores = cross_val_score(nb, X_train_fit, y_train, cv=10)
scores


# In[500]:

nb.fit(X_train_fit, y_train)


# In[501]:

# make class predictions for X train set - training accuracy, as per notes with Kanchana
y_pred_class_train = nb.predict(X_train_fit)


# In[502]:

# calculate accuracy of training data set (note: not 10-fold cross validation)
from sklearn import metrics
metrics.accuracy_score(y_train, y_pred_class_train)


# In[503]:

# make class predictions for test data set
y_pred_class = nb.predict(X_test_fit)


# In[504]:

# calculate accuracy of class predictions
metrics.accuracy_score(y_test, y_pred_class)


# In[505]:

# print the confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)


# In[506]:

# print the classification report
print(metrics.classification_report(y_test, y_pred_class))


# In[507]:

# calculate null accuracy
y_test.value_counts().head() / len(y_test)


# In[255]:

#ROC_AUC cannot be applied to multi-class dataset
#roc_auc_score(y_test, nb.predict_proba(X_test_fit)[:,1])


# In[ ]:




# In[ ]:




# In[256]:

##-----Bi-class- Multinomial Naive Bayes classification - unigrams - 3000 word limit.----------##
##_____________________________________________________________________________________________##


# In[508]:

#Assign applicable columns to new variable to prepare for classifier
X1 = raw_data_negpos.norm_text
y1 = raw_data_negpos.sentiment


# In[509]:

# split the data into training and test data sets (75:25)
from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

#split data (another technique) - tried and works - maintaining in here for reference
y1_train=y1.sample(frac=0.75,random_state=200)
y1_test=y1.drop(y1_train.index)
X1_train=X1.sample(frac=0.75, random_state=200)
X1_test=X1.drop(X1_train.index)


# In[510]:

X1_train.shape


# In[511]:

X1_test.shape


# In[512]:

y1_train.shape


# In[513]:

y1_test.shape


# In[514]:

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score
from sklearn import metrics


# In[515]:

# instantiate CountVectorizer
#using unigram model 
#baseline classifier will be based on this approach - Notes from meeting with Kanchana)
stop = set(stopwords.words('english'))
vect1 = CountVectorizer(stop_words=stop, max_features = 3000)

#stop = set(stopwords.words('english'))
#vect2 = TfidfVectorizer(use_idf=True, min_df=5, max_df = 0.8, max_features=2000)#, stop_words=stop)
#use_idf – weight factor must use inverse document frequency
#min_df – remove the words from the vocabulary which have occurred in less than ‘min_df’ number of tweets.
#max_df – remove the words from the vocabulary which have occurred in more than ‘max_df’ * total number of tweets in dataset
#max_features – choose maximum number of words to be kept in vocabulary ordered by term frequency.


# In[516]:

# fit and transform X_train into X_train_fit
X1_train_fit = vect1.fit_transform(X1_train)
X1_train_fit.shape


# In[517]:

# transform X_test into X_test_fit
X1_test_fit = vect1.transform(X1_test)
X1_test_fit.shape


# In[518]:

# import and instantiate MultinomialNB
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[519]:

#Accurac10-fold cross validation on trainig data set.  y score for each fold. 
from sklearn.model_selection import cross_val_score
scores = cross_val_score(nb, X1_train_fit, y1_train, cv=10)
scores


# In[520]:

nb.fit(X1_train_fit, y1_train)


# In[521]:

# make class predictions for X train set - training accuracy, as per notes with Kanchana
y1_pred_class_train = nb.predict(X1_train_fit)


# In[522]:

# calculate accuracy of training data set (note: not 10-fold cross validation)
from sklearn import metrics
metrics.accuracy_score(y1_train, y1_pred_class_train)


# In[523]:

# make class predictions for test data fit
y1_pred_class = nb.predict(X1_test_fit)


# In[524]:

# calculate accuracy of class predictions
metrics.accuracy_score(y1_test, y1_pred_class)


# In[525]:

# print the confusion matrix
metrics.confusion_matrix(y1_test, y1_pred_class)


# In[526]:

# print the classification report
print(metrics.classification_report(y1_test, y1_pred_class))


# In[527]:

# calculate null accuracy
y1_test.value_counts().head() / len(y1_test)


# In[277]:

roc_auc_score(y1_test, nb.predict_proba(X1_test_fit)[:,1])


# In[ ]:




# In[ ]:

##-----------Multiclass  -  Naive Bayes, unigrams 500 --------------------##
##________________________________________________________________________##


# In[528]:

# instantiate CountVectorizer
#using unigram model - most frequent 500 words
#baseline classifier will be based on this approach - Notes from meeting with Kanchana)
stop = set(stopwords.words('english'))
vect = CountVectorizer(stop_words=stop, max_features = 500)


# In[529]:

# fit and transform X_train into X_train_fit
X_train_fit = vect.fit_transform(X_train)
X_train_fit.shape


# In[530]:

# transform X_test into X_test_fit
X_test_fit = vect.transform(X_test)
X_test_fit.shape


# In[531]:

# transform X_test into X_test_fit
X_test_fit = vect.transform(X_test)
X_test_fit.shape


# In[532]:

# import and instantiate Multinomial NB classifier
nb = MultinomialNB()


# In[535]:

#Accuracy - 10-fold cross validation on training data set.  y score for each fold. 
from sklearn.model_selection import cross_val_score
scores = cross_val_score(nb, X_train_fit, y_train, cv=10)
scores


# In[536]:

nb.fit(X_train_fit, y_train)


# In[537]:

# make class predictions for X train set - training accuracy, as per notes with Kanchana
y_pred_class_train = nb.predict(X_train_fit)


# In[538]:

# calculate accuracy of training data set (note: not 10-fold cross validation)
from sklearn import metrics
metrics.accuracy_score(y_train, y_pred_class_train)


# In[539]:

# make class predictions for test data set
y_pred_class = nb.predict(X_test_fit)


# In[540]:

# calculate accuracy of class predictions
metrics.accuracy_score(y_test, y_pred_class)


# In[541]:

# print the confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)


# In[542]:

# print the classification report
print(metrics.classification_report(y_test, y_pred_class))


# In[543]:

# calculate null accuracy
y_test.value_counts().head() / len(y_test)


# In[544]:

#ROC_AUC cannot be applied to multi-class dataset
#roc_auc_score(y_test, nb.predict_proba(X_test_fit)[:,1])


# In[ ]:




# In[545]:

##-----Bi-class- Multinomial Naive Bayes classification - unigrams - 500 word limit.----------##
##____________________________________________________________________________________________##


# In[546]:

# instantiate CountVectorizer
#using unigram model 
stop = set(stopwords.words('english'))
vect1 = CountVectorizer(stop_words=stop, max_features = 500)

#stop = set(stopwords.words('english'))
#vect2 = TfidfVectorizer(use_idf=True, min_df=5, max_df = 0.8, max_features=2000)#, stop_words=stop)
#use_idf – weight factor must use inverse document frequency
#min_df – remove the words from the vocabulary which have occurred in less than ‘min_df’ number of tweets.
#max_df – remove the words from the vocabulary which have occurred in more than ‘max_df’ * total number of tweets in dataset
#max_features – choose maximum number of words to be kept in vocabulary ordered by term frequency.


# In[547]:

# fit and transform X_train into X_train_fit
X1_train_fit = vect1.fit_transform(X1_train)
X1_train_fit.shape


# In[548]:

# transform X_test into X_test_fit
X1_test_fit = vect1.transform(X1_test)
X1_test_fit.shape


# In[549]:

# transform X_test into X_test_fit
X1_test_fit = vect1.transform(X1_test)
X1_test_fit.shape


# In[550]:

# import and instantiate MultinomialNB
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[551]:

#Accurac10-fold cross validation on trainig data set.  y score for each fold. 
from sklearn.model_selection import cross_val_score
scores = cross_val_score(nb, X1_train_fit, y1_train, cv=10)
scores


# In[552]:

nb.fit(X1_train_fit, y1_train)


# In[553]:

# make class predictions for X train set - training accuracy, as per notes with Kanchana
y1_pred_class_train = nb.predict(X1_train_fit)


# In[554]:

# calculate accuracy of training model using train, test split
from sklearn import metrics
metrics.accuracy_score(y1_train, y1_pred_class_train)


# In[555]:

# make class predictions for X_test_fit
y1_pred_class = nb.predict(X1_test_fit)


# In[556]:

# calculate accuracy of class predictions
from sklearn import metrics
metrics.accuracy_score(y1_test, y1_pred_class)


# In[557]:

# print the confusion matrix
metrics.confusion_matrix(y1_test, y1_pred_class)


# In[558]:

# print the classification report
print(metrics.classification_report(y1_test, y1_pred_class))


# In[559]:

# calculate null accuracy
y1_test.value_counts().head(2) / y1_test.shape


# In[560]:

roc_auc_score(y1_test, nb.predict_proba(X1_test_fit)[:,1])


# In[ ]:




# In[ ]:

##-----Multiclass - Multinomial Naive Bayes classification - TF-IDF --------##
#____________________________________________________________________________#


# In[561]:

# instantiate CountVectorizer
#Solution 2 using TF-IDF, using all features
#vect = CountVectorizer()

vect2 = TfidfVectorizer(use_idf=True, min_df=5, max_df = 0.8)
#use_idf – weight factor must use inverse document frequency
#min_df – remove the words from the vocabulary which have occurred in less than ‘min_df’ number of tweets.
#max_df – remove the words from the vocabulary which have occurred in more than ‘max_df’ * total number of tweets in dataset
#max_features – choose maximum number of words to be kept in vocabulary ordered by term frequency.


# In[562]:

# fit and transform X_train into X_train_fit
X_train_fit = vect2.fit_transform(X_train)
X_train_fit.shape


# In[563]:

# transform X_test into X_test_fit
X_test_fit = vect2.transform(X_test)
X_test_fit.shape


# In[564]:

#Top features in the TF-IDF vector
indices = np.argsort(vect2.idf_)[::-1]
features = vect2.get_feature_names()
top_n = 25
top_features = [features[i] for i in indices[:top_n]]
print (top_features)


# In[565]:

# import and instantiate MultinomialNB
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[566]:

#Accuracy score for each fold, using 10-fold cross validation on trainig data set
from sklearn.model_selection import cross_val_score
scores = cross_val_score(nb, X_train_fit, y_train, cv=10)
scores


# In[567]:

nb.fit(X_train_fit, y_train)


# In[568]:

# make class predictions for X train set - training accuracy, as per notes with Kanchana
y_pred_class_train = nb.predict(X_train_fit)


# In[569]:

#calculate accuracy of training model using train, test split
from sklearn import metrics
metrics.accuracy_score(y_train, y_pred_class_train)


# In[570]:

# make class predictions for X_test_fit
y_pred_class = nb.predict(X_test_fit)


# In[571]:

# calculate accuracy of class predictions
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)


# In[572]:

# print the confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)


# In[247]:

# print the classification report
print(metrics.classification_report(y_test, y_pred_class))


# In[248]:

#ROC_AUC cannot be applied to multi-class dataset
#roc_auc_score(y_test, nb.predict_proba(X_test_fit)[:,1])


# In[ ]:




# In[ ]:

##---------Multinomial Naive Bayes, TF-IDF, 2 class ----------------##
##__________________________________________________________________##


# In[750]:

# instantiate TFIDFVectorizer
#Solution 2 using TF-IDF with limit of fearues set to 2000
#vect = CountVectorizer()
vect2 = TfidfVectorizer(use_idf=True, min_df=5, max_df = 0.8)
#use_idf – weight factor must use inverse document frequency
#min_df – remove the words from the vocabulary which have occurred in less than ‘min_df’ number of tweets.
#max_df – remove the words from the vocabulary which have occurred in more than ‘max_df’ * total number of tweets in dataset
#max_features – choose maximum number of words to be kept in vocabulary ordered by term frequency.


# In[751]:

# fit and transform X_train into X_train_fit
X1_train_fit = vect2.fit_transform(X1_train)
X1_train_fit.shape


# In[752]:

# transform X_test into X_test_fit
X1_test_fit = vect2.transform(X1_test)
X1_test_fit.shape


# In[753]:

# transform X_test into X_test_fit
X5_test_fit = vect2.transform(X5_test)
X5_test_fit.shape


# In[754]:

#Top features in the TF-IDF vector
indices = np.argsort(vect2.idf_)[::-1]
features = vect2.get_feature_names()
top_n = 25
top_features = [features[i] for i in indices[:top_n]]
print (top_features)


# In[755]:

# import and instantiate MultinomialNB
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[756]:

#Accuracy score for each fold, using 10-fold cross validation on trainig data set
from sklearn.model_selection import cross_val_score
scores = cross_val_score(nb, X1_train_fit, y1_train, cv=10)
scores


# In[757]:

nb.fit(X1_train_fit, y1_train)


# In[758]:

# make class predictions for X train set - training accuracy, as per notes with Kanchana
y1_pred_class_train = nb.predict(X1_train_fit)


# In[759]:

# calculate accuracy of training model using train, test split
from sklearn import metrics
metrics.accuracy_score(y1_train, y1_pred_class_train)


# In[760]:

# make class predictions for X_test_fit
y1_pred_class = nb.predict(X1_test_fit)


# In[588]:

# calculate accuracy of class predictions
from sklearn import metrics
metrics.accuracy_score(y1_test, y1_pred_class)


# In[590]:

# print the confusion matrix
metrics.confusion_matrix(y1_test, y1_pred_class)


# In[592]:

# print the classification report
print(metrics.classification_report(y1_test, y1_pred_class))


# In[594]:

roc_auc_score(y1_test, nb.predict_proba(X1_test_fit)[:,1])


# In[596]:

# calculate null accuracy
y1_test.value_counts().head() / y1_test.shape


# In[ ]:

#######Calculating stats manuallly  + other information extraction#######


# In[600]:

# examine the class distribution of the testing set
y1_test.value_counts()


# In[601]:

# calculate null accuracy
y1_test.value_counts().head(1) / y1_test.shape


# In[603]:

# first 10 false positives (negative incorrectly classified as positive sentiment)
X1_test[y1_test < y1_pred_class].head(10)


# In[604]:

# first 10 false negatives (positive incorrectly classified as negative)
X1_test[y1_test > y1_pred_class].head(10)


# In[606]:

# store the words/features of X_train
X1_train_tokens = vect2.get_feature_names()
len(X1_train_tokens)


# In[607]:

# first row is negative sentiment, second row is positive sentiment
nb.feature_count_.shape


# In[608]:

# store the number of times each token appears across each class
negative_words_count = nb.feature_count_[0, :]
positive_words_count = nb.feature_count_[1, :]


# In[612]:

# create a DataFrame of tokens with their separate negative and positive counts
#tokens = pd.DataFrame({'token':X_train_tokens, 'negative':negative_words_count, 'positive':positive_words_count}).set_index('token')
tokens = pd.DataFrame({'token':X1_train_tokens, 'negative':negative_words_count, 'positive':positive_words_count}).set_index('token')


# In[613]:

# add 1 to negative and positive counts to avoid dividing by 0
tokens['negative'] = tokens.negative + 1
tokens['positive'] = tokens.positive + 1


# In[614]:

# first number is negative tweets, second number is positive tweets
nb.class_count_


# In[615]:

# convert the negative and positive counts into frequencies
tokens['negative'] = tokens.negative / nb.class_count_[0]
tokens['positive'] = tokens.positive / nb.class_count_[1]


# In[616]:

# calculate the ratio of positive to negative tweets for each token
tokens['positive_ratio'] = tokens.positive / tokens.negative


# In[617]:

# sort the DataFrame by positive_ratio (descending order), and examine the first 10 rows
tokens.sort_values('positive_ratio', ascending=False).head(10)


# In[618]:

# sort the DataFrame by positive_words_count_ratio (ascending order), and examine the first 10 rows
tokens.sort_values('positive_ratio', ascending=True).head(10)


# In[254]:




# In[ ]:




# In[255]:




# In[619]:

##-----Multiclass - Random Forest classification - TFIDF, ----------##
##__________________________________________________________________##


# In[620]:

#Random Forest
from sklearn.ensemble import RandomForestClassifier


# In[621]:

clf = RandomForestClassifier(max_features=2000)
clf.fit(X_train_fit, y_train)


# In[622]:

X_train.shape


# In[623]:

X_test.shape


# In[624]:

y_train.shape


# In[625]:

y_test.shape


# In[626]:

X_train_fit.shape


# In[627]:

#Accuracy score for each fold, using 10-fold cross validation on trainig data set
from sklearn.model_selection import cross_val_score
scores2 = cross_val_score(clf, X_train_fit, y_train, cv=10)
scores2


# In[628]:

# Apply the Classifier we trained to the test data 
y_pred_class2 = clf.predict(X_test_fit)


# In[630]:

# calculate accuracy of class predictions
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class2)


# In[631]:

# print the confusion matrix
metrics.confusion_matrix(y_test, y_pred_class2)


# In[632]:

# print the classification report
print(metrics.classification_report(y_test, y_pred_class2))


# In[633]:

# calculate null accuracy
y_test.value_counts().head() / y_test.shape


# In[ ]:

#ROC_AUC cannot be applied to multi-class dataset
#roc_auc_score(y_test, nb.predict_proba(X_test_fit)[:,1])


# In[ ]:




# In[151]:

##-----2 classes - Random Forest classification ----------##
##________________________________________________________##


# In[634]:

#instantiate TFIDFVectorizer
#Solution 2 using TF-IDF with limit of fearues set to 2000

vect3 = TfidfVectorizer(use_idf=True, min_df=5, max_df = 0.8)
#use_idf – weight factor must use inverse document frequency
#min_df – remove the words from the vocabulary which have occurred in less than ‘min_df’ number of tweets.
#max_df – remove the words from the vocabulary which have occurred in more than ‘max_df’ * total number of tweets in dataset
#max_features – choose maximum number of words to be kept in vocabulary ordered by term frequency.


# In[635]:

# fit and transform X_train into X_train_fit
X1_train_fit = vect3.fit_transform(X1_train)
X1_train_fit.shape


# In[636]:

# transform X_test into X_test_fit
X1_test_fit = vect3.transform(X1_test)
X1_test_fit.shape


# In[638]:

#Random Forest
from sklearn.ensemble import RandomForestClassifier


# In[639]:

X1_train_fit.shape


# In[640]:

#Instantiate the classifier
clf = RandomForestClassifier()
clf.fit(X1_train_fit, y1_train)


# In[641]:

#Accuracy score for each fold, using 10-fold cross validation on trainig data set
from sklearn.model_selection import cross_val_score
scores2 = cross_val_score(clf, X1_train_fit, y1_train, cv=10)
scores2


# In[642]:

# Apply the Classifier we trained to the test data 
y1_pred_class2 = clf.predict(X1_test_fit)


# In[644]:

# calculate accuracy of class predictions
from sklearn import metrics
metrics.accuracy_score(y1_test, y1_pred_class2)


# In[645]:

# print the confusion matrix
metrics.confusion_matrix(y1_test, y1_pred_class2)


# In[646]:

# print the classification report
print(metrics.classification_report(y1_test, y1_pred_class2))


# In[647]:

roc_auc_score(y1_test, clf.predict_proba(X1_test_fit)[:,1])


# In[173]:

# calculate null accuracy
y1_test.value_counts().head() / y1_test.shape


# In[ ]:




# In[ ]:




# In[ ]:




# In[256]:

#BACKUP OF THE RAWDATA SET AT THIS STEP
X_train.to_csv('D:/COURSE_TEMP/capstone/airline_sentiment/phase3/int_files/X_train.csv', sep=',')
y_train.to_csv('D:/COURSE_TEMP/capstone/airline_sentiment/phase3/int_files/y_train.csv', sep=',')
#BACKUP OF THE TRAIN DATA SET AT TTHIS STEP
#train.to_csv('D:/COURSE_TEMP/capstone/airline_sentiment/phase3/int_files/train.csv', sep=',')
#BACKUP OF THE TEST DATA SET AT TTHIS STEP
#test.to_csv('D:/COURSE_TEMP/capstone/airline_sentiment/phase3/int_files/test.csv', sep=',')


# In[ ]:




# In[ ]:

##------------------Test of new dataset created by thrid party individuals---------------------##
##            Bi-class data set - Multinomial Naive Bayes, TF-IDF classifier trained above     ##
##_____________________________________________________________________________________________##


# In[800]:

test_data = pd.read_csv('D:/COURSE_TEMP\capstone/airline_sentiment/Airline_twitter_data/test_tweets.csv')


# In[801]:

test_data.shape


# In[802]:

test_data.head()


# In[803]:

#Convert text to lowercase in text field
test_data['norm_text'] = test_data.tweet.str.lower()


# In[804]:


test_data[0:4]


# In[805]:

test_data[0:4]


# In[806]:

test_data['norm_text']= test_data.norm_text.str.replace('\@[a-z0-9]+', ' ')


# In[807]:

# Remove Stopwords
from nltk.corpus import stopwords
stop = stopwords.words("english")
test_data['norm_text'] = test_data['norm_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# In[808]:

#Remove special characters without removing apostraphes
pattern=re.compile("[^\w']")
test_data['norm_text'] = test_data.norm_text.str.replace(pattern,' ')


# In[809]:

test_data['norm_text'] = test_data.norm_text.str.replace('https?:\/\/.*[\r\n]*',' ')
test_data['norm_text'] = test_data.norm_text.str.replace('http:\/\/.*[\r\n]*',' ')
test_data['norm_text'] = test_data.norm_text.str.replace('http',' ')
test_data['norm_text'] = test_data.norm_text.str.replace('\#',' ')
test_data['norm_text'] = test_data.norm_text.str.replace('\,',' ')
test_data['norm_text'] = test_data.norm_text.str.replace('\;[a-z0-9]+',' ')
test_data['norm_text'] = test_data.norm_text.str.replace(';',' ')   
test_data['norm_text'] = test_data.norm_text.str.replace('\:[a-z0-9]+',' ')
test_data['norm_text'] = test_data.norm_text.str.replace(':+',' ')
test_data['norm_text'] = test_data.norm_text.str.replace('&',' ')
test_data['norm_text'] = test_data.norm_text.str.replace('\.+',' ')
test_data['norm_text'] = test_data.norm_text.str.replace('.',' ')
test_data['norm_text'] = test_data.norm_text.str.replace('-',' ')
test_data['norm_text'] = test_data.norm_text.str.replace('!+','')
test_data['norm_text'] = test_data.norm_text.str.replace('\?+','')
test_data['norm_text'] = test_data.norm_text.str.replace('?','')
test_data['norm_text'] = test_data.norm_text.str.replace('$','')
test_data['norm_text'] = test_data.norm_text.str.replace('(','')
test_data['norm_text'] = test_data.norm_text.str.replace(')','')
test_data['norm_text'] = test_data.norm_text.str.replace('"\w+','')
test_data['norm_text'] = test_data.norm_text.str.replace('[0-9]+',' ')
test_data['norm_text'] = test_data.norm_text.str.replace('\b[^\W\d][^\W\d]+\b','')


# In[810]:

shortword = re.compile(r'\b\w{1,3}\b') #r'\W*\b\w{1,3}\b'
test_data['norm_text'] = test_data.norm_text.str.replace(shortword,' ')
test_data['norm_text'] = test_data.norm_text.str.replace('\'','')
test_data['norm_text'] = test_data.norm_text.str.replace('didn',"didn't")
test_data['norm_text'] = test_data.norm_text.str.replace(' +',' ')


# In[811]:

test_data[0:4]


# In[812]:

list(test_data)


# In[813]:

test_data['sentiment'].dtype.kind 


# In[814]:

test_data.shape


# In[815]:

# instantiate TFIDFVectorizer
#Solution 2 using TF-IDF with limit of fearues set to 2000
#vect = CountVectorizer()
vect2 = TfidfVectorizer(use_idf=True, min_df=5, max_df = 0.8)
#use_idf – weight factor must use inverse document frequency
#min_df – remove the words from the vocabulary which have occurred in less than ‘min_df’ number of tweets.
#max_df – remove the words from the vocabulary which have occurred in more than ‘max_df’ * total number of tweets in dataset
#max_features – choose maximum number of words to be kept in vocabulary ordered by term frequency.


# In[816]:

# fit and transform X_train into X_train_fit
X1_train_fit = vect2.fit_transform(X1_train)
X1_train_fit.shape


# In[817]:

# transform X_test into X_test_fit
X1_test_fit = vect2.transform(X1_test)
X1_test_fit.shape


# In[ ]:




# In[ ]:




# In[818]:

X5 = test_data['norm_text']
y5 = test_data['sentiment']


# In[819]:

y5_train=y5.sample(frac=0.0,random_state=200)
y5_test=y5.drop(y5_train.index)
X5_train=X5.sample(frac=0.0, random_state=200)
X5_test=X5.drop(X5_train.index)


# In[820]:

X5_test.shape


# In[ ]:




# In[821]:

y5_test.shape


# In[822]:

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score
from sklearn import metrics


# In[823]:

X5_test_fit = vect2.transform(X5_test)
X5_test_fit.shape


# In[ ]:




# In[824]:

# import and instantiate MultinomialNB
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[825]:

#Accurac10-fold cross validation on trainig data set.  y score for each fold. 
from sklearn.model_selection import cross_val_score
scores = cross_val_score(nb, X1_train_fit, y1_train, cv=10)
scores


# In[826]:

nb.fit(X1_train_fit, y1_train)


# In[827]:

# make class predictions for X_test_fit
y5_pred_class = nb.predict(X5_test_fit)


# In[ ]:




# In[828]:

# calculate accuracy of class predictions
metrics.accuracy_score(y5_test, y5_pred_class)


# In[829]:

# print the confusion matrix
metrics.confusion_matrix(y5_test, y5_pred_class)


# In[830]:

# print the classification report
print(metrics.classification_report(y5_test, y5_pred_class))


# In[831]:

# calculate null accuracy
y1_test.value_counts().head() / len(y1_test)


# In[832]:

roc_auc_score(y5_test, nb.predict_proba(X5_test_fit)[:,1])


# 

# In[ ]:




# In[ ]:

###Other code used throughotu the project###


# In[ ]:

#The code below was used to attempt to label polarity.  It is relevant if i have ttime to do that work.  
#Leaving it here for reference/


# In[ ]:

#train2.rename(columns={'\s':'rowID', 'tweet_id':'tweet_id', 'airline_sentiment':'airline_sent', 
#                      'airline_sentiment_confidence':'airline_sent_conf','airline':'airline',
#                      'text':'text', 'tweet_created':'created', 'airline2':'airline2'}, inplace=True)


# In[ ]:

#def clean_data(token_text):
#    '''
#    Utility function to clean the text in a tweet by removing 
#    links and special characters using regex.
#    '''
#    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())


# In[ ]:

#Assign polarity to train data set using TextBlob
#from textblob import TextBlob

#def analize_sentiment(text):
#    analysis = TextBlob(train2(token_text))
#    if analysis.sentiment.polarity > 0:
#        return 1
#    elif analysis.sentiment.polarity == 0:
#        return 0
#    else:
#        return -1


# In[ ]:

#train2['SA'] = np.array([ analize_sentiment(text) for text in train2['token_text'] ])


# In[ ]:




# In[ ]:

get airline from the tweet


def get_airline2(row):
    airline2 = []
    text = row["text"]
    if "@virginamerica" in text or "virgin" in text:
         airline2.append("Virgin America ")
    elif "@united" in text or "united" in text:
        airline2.append("United ")
    elif "@southwestair" in text or "southwestair" in text:
        airline2.append("Southwest ")
    elif "@delta" in text or "delta" in text or "@jetblue" in text or "jetblue" in text:
        airline2.append("Delta ")
    elif "@usairways" in text or "usairways" in text:
        airline2.append("US Airways ")
    elif "@americanair" in text or "americanair" in text:
        airline2.append("American Airlines ")
    return ",".join(airline2)
raw_data['airline2'] = raw_data.apply(get_airline2,axis=1)


# In[30]:

import plotly
from plotly import graph_objs
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# In[31]:

get_ipython().magic('matplotlib inline')

df = raw_data.airline_sentiment
neg = len(df[raw_data["airline_sentiment"] == "negative"])
pos = len(df[raw_data["airline_sentiment"] == "positive"])
neu = len(df[raw_data["airline_sentiment"] == "neutral"])
dist = [
    graph_objs.Bar(
        x=["negative","neutral","positive"],
        y=[neg, neu, pos],
)]
plotly.offline.iplot({"data":dist, "layout":graph_objs.Layout(title="Sentiment type distribution in Airline dataset")})


# In[33]:

get_ipython().magic('matplotlib inline')
# Plot histogram of tweets per airline


airline_count = Counter(raw_data['airline2'])
ac_df = pd.DataFrame.from_dict(airline_count, orient = 'index')
ac_df.plot(kind='bar')


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




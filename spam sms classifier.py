import pandas as pd
messages=pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names=['label', 'message'])
print(messages.head())
#print(messages.describe())
#print(messages.groupby('label').describe())
messages['length'] = messages['message'].apply(len)

import string
from nltk.corpus import stopwords

def text_process(mess):
    nopunc=[char for char in mess if char not in string.punctuation]
    nopunc=''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC

pipeline=Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)), 
    ('tfidf', TfidfTransformer()),
    ('classifier', SVC(kernel='sigmoid', gamma=1.0))
])

from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test=train_test_split(messages['message'], messages['label'], test_size=0.3, random_state=101)

pipeline.fit(msg_train, label_train)
pred=pipeline.predict(msg_test)

from sklearn.metrics import classification_report, confusion_matrix

print("\nConfusion Matrix:")
print(confusion_matrix(label_test, pred))
print("\nClassification Report")
print(classification_report(label_test, pred))

from sklearn.metrics import accuracy_score

print("\nAccuracy:")
print(accuracy_score(label_test, pred))

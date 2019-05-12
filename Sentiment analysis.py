from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import fileinput


df = pd.read_csv('reviews_data.csv')
df = df.dropna()

# pip install afinn   !!required
from afinn import Afinn
afinn = Afinn()
# compute sentiment scores (polarity)
text_all = df['text'].tolist()   # save text to a list
sentiment_score = []
for text in text_all:
    sentiment_score.append(afinn.score(text))

print(sentiment_score)
df[['sentiment_score']].plot(kind='hist',rwidth=0.8, figsize=(10,5), bins=30)
plt.show()
# categorize the data
sentiment_category = ['positive' if score > 0 else 'negative' if score < 0 else 'neutral' for score in sentiment_score]

df['sentiment_category'] = sentiment_category
df['sentiment_score'] = sentiment_score

print(df)

#############################################
#derive ratings based on review
# here I used "reviewer_average_stars" as ratings, and build a regression model with text as features. 
vectorizer = CountVectorizer(analyzer = 'word',lowercase = False,)
features = vectorizer.fit_transform(text_all)
features_nd = features.toarray()

from sklearn.cross_validation import train_test_split

ratings = df['reviewer_average_stars'].tolist()
X_train, X_test, y_train, y_test  = train_test_split(features_nd, ratings, train_size=0.75, random_state=123)

from sklearn.linear_model import LinearRegression  # I picked Linear Regression model for this project, however, more models should be tested if time is allowed
reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)

from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, y_pred))

# Another way is to use sentiment score, reviewer_review_count, and reviewer_useful as variables:
feature_cols = ['sentiment_score', 'reviewer_review_count', 'reviewer_useful']
X = df.loc[:, feature_cols]
y = df.reviewer_average_stars

model_2 = LinearRegression().fit(X, y)
y_pred_2 = model_2.predict(X)
print(mean_absolute_error(y, y_pred_2))
# it turns out the second method can predict the ratings better!!

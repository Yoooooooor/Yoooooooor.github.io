# -*- coding: utf-8 -*-
"""7380_task5_over.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sgk5tyf72DoU2ybKtFgo8ec5eP9aF4zL

##CODE
"""

# Install libraries using pip package in the current Jupyter kernel
import sys
!{sys.executable} -m pip install matplotlib
!{sys.executable} -m pip install pandas
!{sys.executable} -m pip install numpy

import pandas as pd 
import numpy as np

# from google.colab import drive
# drivePath = '/content/drive' #please do not change
# drive.mount(drivePath)

URL = 'https://raw.githubusercontent.com/Cloudy97/7380Data/main/news_dataset1.csv'

newsData = pd.read_csv(URL)
newsData = newsData.dropna()

newsData.head(5)

newsData['ContentID'].isnull().sum()

newsData.shape

#Create a user-item binary matrix

users = newsData.UserID.unique()
content = newsData.ContentID.unique()

newsMatrix = pd.DataFrame(columns=content, index=users)
newsMatrix.head(2)

# Type of events recorded in the logs
eventTypes = newsData.Event.unique()
print(eventTypes)

eventWeights = {
    'read': 15,
    'read_more': 50,
    'author': 5,
    'category': 5,
    'read_comments': 30}

# Iterate the evidence
for index, row in newsData.iterrows():
    # Select the user and items involved
    currentUser = row['UserID']
    currentContent = row['ContentID']
    
    # Extract the appropriate weight for the event
    w = eventWeights[row['Event']]
    
    # Find the value eventually stored for the current user-item combination
    currentValue = newsMatrix.at[currentUser, currentContent]
    if np.isnan(currentValue):
        currentValue = 0
        
    # Compute the new value and update the user-item matrix
    updatedValue = currentValue + w #+ (1 * w)
    newsMatrix.at[currentUser, currentContent] = updatedValue

newsMatrix.head(5)
#newsMatrix.dropna(inplace=True)

#Create a user-item matrix but with specific threshold 
newsMatrixWithThreshold  = pd.DataFrame(columns=content, index=users)

#built up some Matrix for counting how many times the event has been counted
readMatrix = pd.DataFrame(columns=content, index=users)
read_moreMatrix = pd.DataFrame(columns=content, index=users)
read_commentsMatrix = pd.DataFrame(columns=content, index=users)


eventWeights = {
    'read': 15,
    'read_more': 50,
    'author': 5,
    'category': 5,
    'read_comments': 30}

for index, row in newsData.iterrows():
    currentUser = row['UserID']
    currentContent = row['ContentID']
    w = eventWeights[row['Event']]
    
    # We set the read event threshold to 3
    if(row['Event']=="read"):
        currentTimes = readMatrix.at[currentUser,currentContent]
        if (currentTimes > 3):
            continue
        if np.isnan(currentTimes):
            currentTimes = 0
        updatedTimes = currentTimes + 1
        readMatrix.at[currentUser, currentContent] = updatedTimes

    # We set the read_more event threshold to 3
    if(row['Event']=="read_more"):
        currentTimes = read_moreMatrix.at[currentUser,currentContent]
        if (currentTimes > 3):
            continue
        if np.isnan(currentTimes):
            currentTimes = 0
        updatedTimes = currentTimes + 1
        read_moreMatrix.at[currentUser, currentContent] = updatedTimes

    # We set the buy event threshold to 3
    if(row['Event']=="read_comments"):
        currentTimes = read_commentsMatrix.at[currentUser,currentContent]
        if (currentTimes > 3):
            continue
        if np.isnan(currentTimes):
            currentTimes = 0
        updatedTimes = currentTimes + 1
        read_commentsMatrix.at[currentUser, currentContent] = updatedTimes

    currentValue = newsMatrixWithThreshold.at[currentUser, currentContent]
    if np.isnan(currentValue):
        currentValue = 0
        
    updatedValue = currentValue + w #+ (1 * w)
    newsMatrixWithThreshold.at[currentUser, currentContent] = updatedValue

newsMatrixWithThreshold

import datetime
from datetime import date, timedelta, datetime


def compute_decay(eventDate, decayDays):
    age = (date.today() - datetime.strptime(eventDate, '%Y-%m-%d %H:%M:%S').date()) // timedelta(days=decayDays)
    #print("Age of event:", age)
    decay = 1/age #simple decay
    #print("Decay factor:", decay)
    
    return decay

createdEvent = newsData.at[0,'Date']
thresholdDays = 2 # Number of days 
decayFactor = compute_decay(createdEvent, thresholdDays)

print(decayFactor)

newsMatrixWithDecay = pd.DataFrame(columns=content, index=users)

for index, row in newsData.iterrows():

    currentUser = row['UserID']
    currentContent = row['ContentID']
    

    w = eventWeights[row['Event']]

    currentValue = newsMatrixWithDecay.at[currentUser, currentContent]
    if np.isnan(currentValue):
        currentValue = 0
    
    #Introduce the decay
    createdEvent = row['Date']
    thresholdDays = 30 # Number of days 
    decayFactor = compute_decay(createdEvent, thresholdDays)
    
    # Compute the new value with decay and update the user-item matrix
    updatedValue = currentValue + w*decayFactor #+ (1 * w)

    newsMatrixWithDecay.at[currentUser, currentContent] = updatedValue

newsMatrixWithDecay
#newsMatrixWithDecay.dropna(inplace=True)

"""## Check the sparsity of the matrix
Quite high
"""

# Compute sparsity value of the matrix

# Number of possible ratings
matrixSize = newsMatrixWithDecay.shape[0]*newsMatrixWithDecay.shape[1]

# Number of na elements
matrixNA = newsMatrixWithDecay.isna().sum().sum()

# Sparsity value
uiMatrixSparsity = matrixNA / matrixSize
uiMatrixSparsity

"""## Similarity Precomputation

### Convert to boolean matrix
To count the presence of co-ratings between movies
"""

newsMatrixWithDecayBool = newsMatrixWithDecay.mask(newsMatrixWithDecay>0,1)
newsMatrixWithDecayBool = newsMatrixWithDecayBool.fillna(0)
newsMatrixWithDecayBool.head(3)

"""### Multiply item * item by using 'dot' function
To find the number of co-ratings
"""

Overlapping = newsMatrixWithDecayBool.T.dot(newsMatrixWithDecayBool)
Overlapping.shape

Overlapping.head(3)

"""Create the item to item similarity matrix"""

iiSimMatrix = pd.DataFrame().reindex_like(Overlapping)
iiSimMatrix

"""### Calculate each item's pair similarity, then save the result in the similarity matrix
Only the items with overlapping ratings! Using the cosine similarity
"""

!{sys.executable} -m pip install scipy

from scipy.spatial.distance import cosine

def cosine_sim(df1, df2):
    # check for na in dataframes
    df1na = df1.isna()
    df1clean = df1[~df1na]
    df2clean = df2[~df1na]

    df2na = df2clean.isna()
    df1clean = df1clean[~df2na]
    df2clean = df2clean[~df2na]

    
    # Compute cosine similarity
    distance = cosine(df1clean, df2clean)
    sim = 1 - distance
    
    return sim

# Extract the movies' list
moviesToPrecompute = iiSimMatrix.columns
#threshold = 0

#Extract the item's pair ratings
for item1 in moviesToPrecompute:
    item1Ratings = newsMatrixWithDecay[item1]
    for item2 in moviesToPrecompute:
        item2Ratings = newsMatrixWithDecay[item2]
        threshold = 3
        #threshold = Overlapping.sum(axis=0).mean()
        if Overlapping.at[item1,item2] > threshold:
            iiSimMatrix.at[item1, item2] = cosine_sim(item1Ratings, item2Ratings)

iiSimMatrix.head(5)

"""### Let's check how many similarity we precompute"""

iiSimMatrix.count().sum()

"""# Item-based Collaborative Filtering"""

def itemCF_precomputed(similarityMatrix, currentItem, numItems):
    #Select current item from the similarity matrix, remove not rated items, sort the values and select the top-k items
    recommendationList = similarityMatrix[currentItem].dropna().sort_values(ascending=False).head(numItems)
    
    return recommendationList.index.to_list()

"""The userID is not really needed for recommendation, we will use it to choose an item that the user rated and we want to find similar items."""

cuRatedNews = newsMatrixWithDecay.loc[666].dropna().sort_values(ascending=False)
cuRatedNews.head()

"""let's predict the news similar to his top choice: the news `756`"""

# choose one News to find the similar movies to reccommend
itemCF_precomputed(iiSimMatrix, 756, 5)

# check the number of all catalogue
len(content)

"""Let's check the catalogue coverage"""

#Catalogue coverage
allrecs = [] # a list to save all items in recs

# get rec-items for each item
for item in content:
  recItems = itemCF_precomputed(iiSimMatrix, item, 5)
  allrecs += recItems

# remove duplication
allrecs = list(set(allrecs))

# use the formula to calculate the coverage
Catalogue_Coverage = len(allrecs)/len(content)
print('Catalogue Coverage Value:',Catalogue_Coverage)

"""We can see that our algorithm can almost navigate the full catalogue.

# User-based Collaborative Filtering
"""

import warnings
warnings.filterwarnings('ignore')

#Overlapping_user = newsMatrixWithDecayBool.dot(newsMatrixWithDecayBool.T) #the number of common interactions between pairs of users in the dataset
#from scipy.sparse import csr_matrix

def preprocess_user_CF(df):
    # Normalize the data
    min_val = np.nanmin(df.values)
    max_val = np.nanmax(df.values)
    df_norm = ((df - min_val) / (max_val - min_val)) * 10
    df_norm = df_norm.astype(float)
    # Calculate overlapping users
    overlapping_user = newsMatrixWithDecayBool.dot(newsMatrixWithDecayBool.T)
    
    return df_norm, overlapping_user

def userCF_prediction_optimized(df_norm, overlapping_user, currentUser, numUsers, numItems):
    # Initialize candidate_user with currentUser
    candidate_user = [currentUser]
    threshold = 3
    
    # Find users with overlapping interactions above the threshold
    for user in df_norm.index:
        if overlapping_user.at[currentUser, user] > threshold:
            candidate_user.append(user)
            
    # Select currentUser's data
    cuDf = df_norm.loc[currentUser]
    # Calculate Pearson correlation between currentUser and other users
    corrDf = df_norm.corrwith(cuDf, axis=1, method='pearson')
    # Sort correlations in descending order
    corrDf.sort_values(ascending=False, inplace=True)
    # Remove currentUser's correlation
    corrDf.drop(labels=[currentUser], inplace=True)
    # Select top numUsers correlated users
    corrDf = corrDf.head(numUsers)
    # Select items that currentUser hasn't interacted with
    toPredict = cuDf[cuDf.isna()]
    # Select ratings from the top numUsers correlated users
    ratings = df_norm.loc[corrDf.index]
    # Select index for the items to predict
    ratingsToPredict = ratings[toPredict.index]
    # Calculate the mean rating for each item
    predictedRatings = ratingsToPredict.mean()
    predictedRatings.sort_values(ascending=False, inplace=True)

    return predictedRatings.head(numItems)
# Preprocess the data and calculate overlapping users
df_norm, overlapping_user = preprocess_user_CF(newsMatrix)

# Print the top 5 recommendations for user 1359
print(userCF_prediction_optimized(df_norm, overlapping_user, 1359, 20, 5))

r = userCF_prediction_optimized(df_norm, overlapping_user, 1359, 20, 5)
len(r)

"""Let's check the coverage"""

# user coverage PseudoCode
'''
  Pu = 0
  for user in all_users:
    if num_of_recset > 0:
      Pu += 1
    else:
      Pu += 0
  coverage = Pu/num_of_all_users
''';

# check the number of all users
len(users)

# user coverage 

Pu = 0
for user in users:
  recset = userCF_prediction_optimized(df_norm, overlapping_user, user, 20, 5)
  if len(recset) > 0:
    Pu +=1
UserCoverage = Pu/len(users)

UserCoverage

"""We can see that our algorithm has very good user coverage! It shows that we can make recommendation for every existing user."""
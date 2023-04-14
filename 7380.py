import pandas as pd
import numpy as np
import sys
import pandas as pd 
import numpy as np

newsData = pd.read_csv('C:\\Users\\Administrator\\Desktop\\ccc\\news_dataset1.csv   ')
newsData = newsData.dropna()

newsData.head(5)

newsData['ContentID'].isnull().sum()

newsData.shape

users = newsData.UserID.unique()
content = newsData.ContentID.unique()

newsMatrix = pd.DataFrame(columns=content, index=users)
newsMatrix.head(2)

eventTypes = newsData.Event.unique()
print(eventTypes)

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
    
    currentValue = newsMatrix.at[currentUser, currentContent]
    if np.isnan(currentValue):
        currentValue = 0
    updatedValue = currentValue + w #+ (1 * w)
    newsMatrix.at[currentUser, currentContent] = updatedValue

newsMatrix.head(5)

newsMatrixWithThreshold  = pd.DataFrame(columns=content, index=users)
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
 
    if(row['Event']=="read"):
        currentTimes = readMatrix.at[currentUser,currentContent]
        if (currentTimes > 3):
            continue
        if np.isnan(currentTimes):
            currentTimes = 0
        updatedTimes = currentTimes + 1
        readMatrix.at[currentUser, currentContent] = updatedTimes

    if(row['Event']=="read_more"):
        currentTimes = read_moreMatrix.at[currentUser,currentContent]
        if (currentTimes > 3):
            continue
        if np.isnan(currentTimes):
            currentTimes = 0
        updatedTimes = currentTimes + 1
        read_moreMatrix.at[currentUser, currentContent] = updatedTimes

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
    age = (date.today() - datetime.strptime(eventDate, '%Y/%m/%d %H:%M').date()) // timedelta(days=decayDays)
    decay = 1/age   
    return decay


createdEvent = newsData.at[0,'Date']
thresholdDays = 2 
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
 
    createdEvent = row['Date']
    thresholdDays = 30
    decayFactor = compute_decay(createdEvent, thresholdDays)
    
    updatedValue = currentValue + w*decayFactor #+ (1 * w)

    newsMatrixWithDecay.at[currentUser, currentContent] = updatedValue

newsMatrixWithDecay

matrixSize = newsMatrixWithDecay.shape[0]*newsMatrixWithDecay.shape[1]

matrixNA = newsMatrixWithDecay.isna().sum().sum()

uiMatrixSparsity = matrixNA / matrixSize
uiMatrixSparsity

"""## 相似度预计算

###将矩阵转换为布尔类型矩阵，以计算电影之间的共同评分情况
"""

newsMatrixWithDecayBool = newsMatrixWithDecay.mask(newsMatrixWithDecay>0,1)
newsMatrixWithDecayBool = newsMatrixWithDecayBool.fillna(0)
newsMatrixWithDecayBool.head(3)

"""### 使用“点乘”函数将物品*物品相乘，以查找共同评分的数量"""

Overlapping = newsMatrixWithDecayBool.T.dot(newsMatrixWithDecayBool)
Overlapping.shape

Overlapping.head(3)

"""Create the item to item similarity matrix"""

iiSimMatrix = pd.DataFrame().reindex_like(Overlapping)
iiSimMatrix

"""### 计算每个物品对之间的相似度，并将结果保存在相似度矩阵中。仅考虑有重叠评分的物品，使用余弦相似度计算"""

from scipy.spatial.distance import cosine

def cosine_sim(df1, df2):

    df1na = df1.isna()
    df1clean = df1[~df1na]
    df2clean = df2[~df1na]

    df2na = df2clean.isna()
    df1clean = df1clean[~df2na]
    df2clean = df2clean[~df2na]

    distance = cosine(df1clean, df2clean)
    sim = 1 - distance
    
    return sim

moviesToPrecompute = iiSimMatrix.columns

for item1 in moviesToPrecompute:
    item1Ratings = newsMatrixWithDecay[item1]
    for item2 in moviesToPrecompute:
        item2Ratings = newsMatrixWithDecay[item2]
        threshold = 20
        #threshold = Overlapping.sum(axis=0).mean()
        if Overlapping.at[item1,item2] > threshold:
            iiSimMatrix.at[item1, item2] = cosine_sim(item1Ratings, item2Ratings)

iiSimMatrix.head(5)

iiSimMatrix.count().sum()

"""# 基于物品的协同过滤算法"""

def itemCF_precomputed(similarityMatrix, currentItem, numItems):
    recommendationList = similarityMatrix[currentItem].dropna().sort_values(ascending=False).head(numItems)
    
    return recommendationList.index.to_list()

cuRatedNews = newsMatrixWithDecay.loc[666].dropna().sort_values(ascending=False)
cuRatedNews.head()

"""预测与首选项新闻相似的新闻：756"""

itemCF_precomputed(iiSimMatrix, 756, 5)

"""# 基于用户的协同过滤算法"""

from sklearn.metrics.pairwise import cosine_similarity

def custom_cosine_similarity(a, b):
    sim = cosine_similarity(a.values.reshape(1, -1), b.values.reshape(1, -1))
    return sim[0][0]

def userCF_prediction(df, currentUser, numUsers, numItems):
    df_clean = df.dropna(axis=1, how='any')
    cuDf = df_clean.loc[[currentUser]]
    
    corrDf = df_clean.apply(lambda x: custom_cosine_similarity(cuDf.iloc[0], x), axis=1)
    corrDf.sort_values(ascending=False, inplace=True)
    corrDf.drop(labels=[currentUser], inplace=True)

    top_users = corrDf.head(numUsers)
    toPredict = cuDf[cuDf.isna()]

    ratings = df.loc[top_users.index]
    ratingsToPredict = ratings[toPredict.columns]

    predictedRatings = ratingsToPredict.mean()
    predictedRatings.sort_values(ascending=False, inplace=True)

    return predictedRatings.head(numItems)

newsMatrixWithDecay_filled = newsMatrixWithDecay.fillna(0)
newsMatrixWithDecay_filled[newsMatrixWithDecay_filled < 0] = 0

min_val = np.nanmin(newsMatrixWithDecay.values)
max_val = np.nanmax(newsMatrixWithDecay.values)
newsMatrixWithDecayNorm = ((newsMatrixWithDecay_filled - min_val) / (max_val - min_val)) * 10
recommendations = userCF_prediction(newsMatrixWithDecayNorm, 4471, 2, 5)  
print(recommendations)

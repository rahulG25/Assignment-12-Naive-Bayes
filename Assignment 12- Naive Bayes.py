#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import warnings
warnings.filterwarnings('ignore')

from pandas.plotting import scatter_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import statsmodels.api as sm

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, plot_confusion_matrix


# # Import dataset

# In[2]:


salarydata_train = pd.read_csv('SalaryData_Train.csv')
salarydata_train.head()


# In[3]:


salarydata_test = pd.read_csv('SalaryData_Test.csv')
salarydata_test.head()


# # Exploratory data analysis

# In[4]:


salarydata_train.shape


# We can see that there are 30161 instances and 14 attributes in the training data set.

# In[5]:


salarydata_test.shape


# We can see that there are 15060 instances and 14 attributes in the test data set.

# # View top 5 rows of dataset

# In[6]:


# preview the Training dataset

salarydata_train.head()


# In[7]:


# preview the Test dataset

salarydata_test.head()


# # View summary of Training dataset

# In[8]:


salarydata_train.info()


# In[9]:


salarydata_train.describe()


# In[10]:


salarydata_test.info()


# In[11]:


salarydata_test.describe()


# In[12]:


#Finding the special characters in the data frame 
salarydata_train.isin(['?']).sum(axis=0)


# In[16]:


#Finding the special characters in the data frame 
salarydata_test.isin(['?']).sum(axis=0)


# In[17]:


print(salarydata_train[0:10])


# # Explore categorical variables

# In[18]:


# find categorical variables

categorical = [var for var in salarydata_train.columns if salarydata_train[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :\n\n', categorical)


# In[19]:


# view the categorical variables

salarydata_train[categorical].head()


# There are 9 categorical variables are given by workclass, education, maritalstatus, occupation, relationship, race, sex, native and Salary.
# 
# Salary is the target variable.

# # Explore problems within categorical variables

# In[20]:


# check missing values in categorical variables
salarydata_train[categorical].isnull().sum()


# There are no missing values in the categorical variables. I will confirm this further.

# In[21]:


# view frequency counts of values in categorical variables

for var in categorical: 
    
    print(salarydata_train[var].value_counts())


# In[22]:


# view frequency distribution of categorical variables

for var in categorical: 
    
    print(salarydata_train[var].value_counts()/np.float(len(salarydata_train)))


# In[23]:


# check labels in workclass variable

salarydata_train.workclass.unique()


# In[24]:


# check frequency distribution of values in workclass variable

salarydata_train.workclass.value_counts()


# Explore occupation variable

# In[25]:


# check labels in occupation variable

salarydata_train.occupation.unique()


# In[26]:


# check frequency distribution of values in occupation variable

salarydata_train.occupation.value_counts()


# # Explore native_country variable

# In[27]:


# check labels in native_country variable

salarydata_train.native.unique()


# In[28]:


# check frequency distribution of values in native_country variable

salarydata_train.native.value_counts()


# # Number of labels: cardinality

# In[29]:


# check for cardinality in categorical variables

for var in categorical:
    
    print(var, ' contains ', len(salarydata_train[var].unique()), ' labels')


# # Explore Numerical Variables

# In[30]:


# find numerical variables

numerical = [var for var in salarydata_train.columns if salarydata_train[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)


# There are 5 numerical variables
# 
# There are : ['age', 'educationno', 'capitalgain', 'capitalloss', 'hoursperweek']

# In[33]:


# view the numerical variables

salarydata_train[numerical].head()


# There are 5 numerical variables.
# 
# These are given by age, educationno, capitalgain, capitalloss and hoursperweek. All of the numerical variables are of discrete data type.

# # Explore problems within numerical variables

# In[34]:


# check missing values in numerical variables

salarydata_train[numerical].isnull().sum()


# # Declare feature vector and target variable

# In[35]:


X = salarydata_train.drop(['Salary'], axis=1)

y = salarydata_train['Salary']


# # Split data into separate training and test set

# In[36]:


# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[37]:


# check the shape of X_train and X_test

X_train.shape, X_test.shape


# # Feature Engineering

# In[38]:


X_train.dtypes


# In[39]:


X_test.dtypes


# In[40]:


# display categorical variables

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

categorical


# In[41]:


# display numerical variables

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

numerical


# In[42]:


# print percentage of missing values in the categorical variables in training set

X_train[categorical].isnull().mean()


# In[43]:


# print categorical variables with missing data

for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, (X_train[col].isnull().mean()))


# In[44]:


# impute missing categorical variables with most frequent value

for df2 in [X_train, X_test]:
    df2['workclass'].fillna(X_train['workclass'].mode()[0], inplace=True)
    df2['occupation'].fillna(X_train['occupation'].mode()[0], inplace=True)
    df2['native'].fillna(X_train['native'].mode()[0], inplace=True)  


# In[45]:


# check missing values in categorical variables in X_train

X_train[categorical].isnull().sum()


# In[46]:


# check missing values in categorical variables in X_test

X_test[categorical].isnull().sum()


# In[47]:


# check missing values in X_train

X_train.isnull().sum()


# In[48]:


# check missing values in X_test

X_test.isnull().sum()


# # Encode categorical variables

# In[49]:


# print categorical variables

categorical


# In[50]:


X_train[categorical].head()


# In[51]:


get_ipython().system('pip install category_encoders')


# In[52]:


# import category encoders

import category_encoders as ce


# In[53]:


# encode remaining variables with one-hot encoding

encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 
                                 'race', 'sex', 'native'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)


# In[54]:


X_train.head()


# In[55]:


X_train.shape


# We can see that from the initial 14 columns, we now have 102 columns.

# In[56]:


X_test.head()


# In[57]:


X_test.shape


# # Feature Scaling

# In[58]:


cols = X_train.columns


# In[59]:


from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# In[60]:


X_train = pd.DataFrame(X_train, columns=[cols])


# In[61]:


X_test = pd.DataFrame(X_test, columns=[cols])


# In[62]:


X_train.head()


# We now have X_train dataset ready to be fed into the Gaussian Naive Bayes classifier.

# # Model training

# In[64]:


# train a Gaussian Naive Bayes classifier on the training set
from sklearn.naive_bayes import GaussianNB


# instantiate the model
gnb = GaussianNB()


# fit the model
gnb.fit(X_train, y_train)


# # Predict the results

# In[65]:


y_pred = gnb.predict(X_test)

y_pred


# # Check accuracy score

# In[66]:


from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# Model accuracy score: 0.7995
# Here, y_test are the true class labels and y_pred are the predicted class labels in the test-set.

# # Compare the train-set and test-set accuracy

# In[67]:


y_pred_train = gnb.predict(X_train)

y_pred_train


# In[68]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


# # Check for overfitting and underfitting

# In[69]:


# print the scores on training and test set

print('Training set score: {:.4f}'.format(gnb.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(gnb.score(X_test, y_test)))


# Training set score: 0.8023
# Test set score: 0.7995
# The training-set accuracy score is 0.8023 while the test-set accuracy to be 0.7995. These two values are quite comparable. So, there is no sign of overfitting

# # Compare model accuracy with null accuracy

# In[70]:


# check class distribution in test set

y_test.value_counts()


# In[71]:


# check null accuracy score

null_accuracy = (7407/(7407+2362))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))


# We can see that our model accuracy score is 0.8023 but null accuracy score is 0.7582. So, we can conclude that our Gaussian Naive Bayes Classification model is doing a very good job in predicting the class labels.

# # Confusion matrix

# In[73]:


# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


# In[96]:


# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# # Classification metrices

# In[97]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# # Classification accuracy

# In[98]:


TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]


# In[99]:


# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))


# # Classification error

# In[100]:


# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))


# # Precision

# In[101]:


# print precision score

precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))


# # Recall

# In[102]:


recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))


# True Positive Rate is synonymous with Recall.

# In[103]:


true_positive_rate = TP / float(TP + FN)


print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))


# # False Positive Rate

# In[104]:


false_positive_rate = FP / float(FP + TN)


print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))


# # Specificity

# In[105]:


specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))


# # Calculate class probabilities

# In[106]:


# print the first 10 predicted probabilities of two classes- 0 and 1

y_pred_prob = gnb.predict_proba(X_test)[0:10]

y_pred_prob


# Observations

# In each row, the numbers sum to 1.
# 
# There are 2 columns which correspond to 2 classes - <=50K and >50K.
# 
#   * Class 0 => <=50K - Class that a person makes less than equal to 50K.
# 
#   * Class 1 => >50K - Class that a person makes more than 50K.
# Importance of predicted probabilities
# 
# We can rank the observations by probability of whether a person makes less than or equal to 50K or more than 50K.
# predict_proba process
# 
# Predicts the probabilities
# 
# Choose the class with the highest probability
# 
# Classification threshold level
# 
# There is a classification threshold level of 0.5.
# 
# Class 0 => <=50K - probability of salary less than or equal to 50K is predicted if probability < 0.5.
# 
# Class 1 => >50K - probability of salary more than 50K is predicted if probability > 0.5.

# In[107]:


# store the probabilities in dataframe

y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - <=50K', 'Prob of - >50K'])

y_pred_prob_df


# In[108]:


# print the first 10 predicted probabilities for class 1 - Probability of >50K

gnb.predict_proba(X_test)[0:10, 1]


# In[109]:


# store the predicted probabilities for class 1 - Probability of >50K

y_pred1 = gnb.predict_proba(X_test)[:, 1]


# In[110]:


# plot histogram of predicted probabilities


# adjust the font size 
plt.rcParams['font.size'] = 12


# plot histogram with 10 bins
plt.hist(y_pred1, bins = 10)


# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of salaries >50K')


# set the x-axis limit
plt.xlim(0,1)


# set the title
plt.xlabel('Predicted probabilities of salaries >50K')
plt.ylabel('Frequency')


# # ROC - AUC
# 

# In[111]:


# plot ROC Curve

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred1, pos_label = '>50K')

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for Gaussian Naive Bayes Classifier for Predicting Salaries')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()


# In[112]:


# compute ROC AUC

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred1)

print('ROC AUC : {:.4f}'.format(ROC_AUC))


# # Interpretation

# In[113]:


# calculate cross-validated ROC AUC 

from sklearn.model_selection import cross_val_score

Cross_validated_ROC_AUC = cross_val_score(gnb, X_train, y_train, cv=5, scoring='roc_auc').mean()

print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))


# # k-Fold Cross Validation
# 

# In[114]:


# Applying 10-Fold Cross Validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(gnb, X_train, y_train, cv = 10, scoring='accuracy')

print('Cross-validation scores:{}'.format(scores))


# In[115]:


# compute Average cross-validation score

print('Average cross-validation score: {:.4f}'.format(scores.mean()))


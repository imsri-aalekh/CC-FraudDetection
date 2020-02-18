
# coding: utf-8

# In[ ]:


#import the necessary packages 


# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from matplotlib import gridspec


# In[ ]:


# Load the dataset from the csv file using pandas 


# In[3]:


data=pd.read_csv(r"C:\Users\Sachin kumar Singh\Downloads\creditcardfraud\creditcard.csv")


# In[4]:


data.head()


# In[6]:


data.shape


# In[7]:


data.describe()


# In[8]:


# Determine number of fraud cases in dataset


# In[9]:


fraud = data[data['Class'] == 1] 
valid = data[data['Class'] == 0] 
outlierFraction = len(fraud)/float(len(valid)) 
print(outlierFraction) 
print('Fraud Cases: {}'.format(len(data[data['Class'] == 1]))) 
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))


# Most of the transactions are non-fraud. If we use this dataframe as the base for our predictive models and analysis we might get a lot of errors and our algorithms will probably overfit since it will “assume” that most transactions are not fraud. But we don’t want our model to assume, we want our model to detect patterns that give signs of fraud!
# The data set is highly skewed, consisting of 492 frauds in a total of 284,807 observations. This resulted in only 0.172% fraud cases. This skewed set is justified by the low number of fraudulent transactions.
# 
# Now that we have the data, we are using only 3 parameters for now in training the model (Time, Amount, and Class)

# In[11]:


fraud.describe()


# In[12]:


valid.describe()


# # Plotting

# In[13]:


# Correlation matrix(Heat Map)


# In[14]:


corrmat = data.corr() 
fig = plt.figure(figsize = (12, 9)) 
sns.heatmap(corrmat, vmax = .8, square = True) 
plt.show()


# In the HeatMap we can clearly see that most of the features do not correlate to other features but there are some features that either has a positive or a negative correlation with each other. For example, V2 and V5 are highly negatively correlated with the feature called Amount. We also see some correlation with V20 and Amount. This gives us a deeper understanding of the Data available to us

# In[15]:


# dividing the X and the Y from the dataset


# In[16]:


X = data.drop(['Class'], axis = 1) 
Y = data["Class"]


# In[17]:


#Printing shape


# In[21]:


print(X.shape)
print()
print(Y.shape)


# In[22]:


#Dividing the dataset for train_test_split


# In[23]:


xData = X.values 
yData = Y.values


# In[ ]:


#Splitting the dataset into Training and Testing using scikit-learn


# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


xTrain, xTest, yTrain, yTest = train_test_split( 
        xData, yData, test_size = 0.2, random_state = 42)


# In[26]:


#Building a Random Forest Classifier using scikit-learn


# In[27]:


from sklearn.ensemble import RandomForestClassifier 


# In[28]:


#model creation


# In[29]:


rfc = RandomForestClassifier()


# In[30]:


rfc.fit(xTrain, yTrain) 


# In[31]:


#Storing the predictions


# In[32]:


yPred = rfc.predict(xTest)


# In[ ]:


# Different Evaluation Metrics


# In[34]:


from sklearn.metrics import classification_report, accuracy_score  
from sklearn.metrics import precision_score, recall_score 
from sklearn.metrics import f1_score, matthews_corrcoef 
from sklearn.metrics import confusion_matrix


# In[35]:


n_outliers = len(fraud) 
n_errors = (yPred != yTest).sum() 
print("************Random Forest classifier****************") 
  
acc = accuracy_score(yTest, yPred) 
print("The accuracy is {}".format(acc)) 
  
prec = precision_score(yTest, yPred) 
print("The precision is {}".format(prec)) 
  
rec = recall_score(yTest, yPred) 
print("The recall is {}".format(rec)) 
  
f1 = f1_score(yTest, yPred) 
print("The F1-Score is {}".format(f1)) 
  
MCC = matthews_corrcoef(yTest, yPred) 
print("The Matthews correlation coefficient is{}".format(MCC))


# In[36]:


#Confusion Matrix


# In[37]:


LABELS = ['Normal', 'Fraud'] 
conf_matrix = confusion_matrix(yTest, yPred) 
plt.figure(figsize =(12, 12)) 
sns.heatmap(conf_matrix, xticklabels = LABELS,  
            yticklabels = LABELS, annot = True, fmt ="d"); 
plt.title("Confusion matrix") 
plt.ylabel('True class') 
plt.xlabel('Predicted class') 
plt.show()


# The Accuracy of our model using Random Forest is 0.999 because of the large imbalance in data set

# Since over 99% of our transactions are non-fraudulent, an algorithm that always predicts that the transaction is non-fraudulent would achieve an accuracy higher than 99%. Owing to such imbalance in data, an algorithm that does not do any feature analysis and predicts all the transactions as non-frauds will also achieve an accuracy of 99.94% (Random Forest). Therefore, accuracy is not a correct measure of efficiency in our case.

# In[40]:


#############################################################


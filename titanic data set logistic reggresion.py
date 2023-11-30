#!/usr/bin/env python
# coding: utf-8

# In[1]:


#libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[2]:


#funtions
def calculate_survival_stats(raw_data: pd.core.frame.DataFrame):
    survival_stats = data.groupby(['Sex', 'Pclass', 'Survived']).size().reset_index(name='Count')

    # Pivot the DataFrame to get the desired output format
    survival_stats = survival_stats.pivot_table(index='Sex', columns=['Survived', 'Pclass'], values='Count',
                                                fill_value=0).reset_index()

    # Rename columns as per the specified format
    survival_stats.columns = ['Gender', 'First class 0', 'First class 1', 'Second class 0', 'Second class 1',
                              'Third class 0', 'Third class 1']

    return survival_stats


# In[3]:


data = pd.read_csv(r"C:\Users\ADMIN\Desktop\dataset\titanic train set.csv")
data_test = pd.read_csv(r"C:\Users\ADMIN\Desktop\dataset\test.csv")
colm = data.columns.values
# Handle categorical variables (e.g., 'Sex')
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'C': 1, 'Q': 2 ,'S':3})
data_test['Sex'] = data_test['Sex'].map({'male': 0, 'female': 1})
data_test['Embarked'] = data_test['Embarked'].map({'C': 1, 'Q': 2 ,'S':3})


# In[4]:


calculate_survival_stats(data)


# In[5]:


data_test.head()


# In[6]:


colm


# In[7]:


data.isnull().sum()


# In[8]:


#funtion for ratio of null values to original
def ratio_null_total(data,null_colm):
    for column in null_colm:
        percent = (data[column].isnull().sum()/data.shape[0])*100
        print(f'percentage for null values for {column} feature is {percent}')


# In[9]:


null_colm = ['Age','Cabin','Embarked']
ratio_null_total(data,null_colm)


# In[10]:


#Replacing null values on Age feature by mean value
#Cabin feature is not gonna use so we negalate that feature
#Embarked feature has low null values it can either be drop or replaces by stat values
data['Age']=data['Age'].fillna(data['Age'].mean())


# In[11]:


data.isnull().sum()


# In[12]:


# Select relevant features and target variable
features = ['Pclass', 'Sex','Age', 'SibSp', 'Parch','Embarked']
target = 'Survived'


# In[13]:


#df_tst['Age']=df_tst['Age'].fillna(data['Age'].mean())
titanic_data = data.dropna(subset=features + [target])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(titanic_data[features], titanic_data[target], test_size=0.2, random_state=42)


# In[14]:


titanic_data.shape


# In[15]:


# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# In[16]:


# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
classification_rep = classification_report(y_test, predictions)

# Display the evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)


# In[17]:


data_test.count()


# In[18]:


df_tst = data_test.drop(["PassengerId",'Name', 'Ticket', 'Fare' ,'Cabin'], axis='columns')


# In[19]:


df_tst.info()


# In[20]:


df_tst.isnull().sum()


# In[21]:


df_tst['Age']=df_tst['Age'].fillna(data['Age'].mean())


# In[22]:


predictions = model.predict(df_tst)
output = pd.DataFrame({'PassengerId': data_test.PassengerId, 'Survived': predictions})


# In[23]:


output


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
train=pd.read_csv("C:\Windows\Lenovo\\train.csv")
gender=pd.read_csv("C:\Windows\Lenovo\\gender_submission.csv")
test=pd.read_csv("C:\Windows\Lenovo\\test.csv")


# In[2]:


import matplotlib.pyplot


# In[3]:


def bar_chart(feature):
    Survived=train[train["Survived"]==1][feature].value_counts()
    dead=train[train["Survived"]==0][feature].value_counts()
    df=pd.DataFrame([Survived,dead])
    df.index=["Survived","dead"]
    df.plot(kind="bar",stacked=True,figsize=(10,5))


# In[4]:


train.head()


# In[5]:


test.head()


# In[6]:


gender.head()


# In[7]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[9]:


train_test_data=[train,test]
for dataset in train_test_data:
    dataset["Title"]=dataset["Name"].str.extract('([A-Za-z]+)\.',expand=False)


# In[10]:


train['Title'].value_counts()


# In[11]:


title_mapping={"Mr":0,"Miss":1,"Mrs":2,"Master":3,"Dr":3,"Rev":3,"Col":3,"Major":3,
               "Mlle":3,"Countess":3,"Dona":3,"Mme":3,"Capt":3,"Sir":3,"Don":3,
               "Jonkheer":3,"Lady":3,"Ms":3}
for dataset in train_test_data:
    dataset["Title"]=dataset["Title"].map(title_mapping)


# In[12]:


train["Age"].fillna(train.groupby("Title")["Age"].transform("median"),inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"),inplace=True)


# In[13]:


train.head()


# In[14]:


train.drop("Name",axis=1,inplace=True)
test.drop("Name",axis=1,inplace=True)


# In[15]:


facet=sns.FacetGrid(train,hue="Survived",aspect=4)
facet.map(sns.kdeplot,"Age",shade=True)
facet.set(xlim=(0,train["Age"].max()))
facet.add_legend()
plt.show()


# In[16]:


facet=sns.FacetGrid(train,hue="Survived",aspect=4)
facet.map(sns.kdeplot,"Age",shade=True)
facet.set(xlim=(0,train["Age"].max()))
facet.add_legend()
plt.xlim(20,30)


# In[17]:


facet=sns.FacetGrid(train,hue="Survived",aspect=4)
facet.map(sns.kdeplot,"Age",shade=True)
facet.set(xlim=(0,train["Age"].max()))
facet.add_legend()
plt.xlim(0,20)


# In[18]:


facet=sns.FacetGrid(train,hue="Survived",aspect=4)
facet.map(sns.kdeplot,"Age",shade=True)
facet.set(xlim=(0,train["Age"].max()))
facet.add_legend()
plt.xlim(40,60)


# In[19]:


facet=sns.FacetGrid(train,hue="Survived",aspect=4)
facet.map(sns.kdeplot,"Age",shade=True)
facet.set(xlim=(0,train["Age"].max()))
facet.add_legend()
plt.xlim(60,80)


# In[20]:


train.head()


# In[21]:


for dataset in train_test_data:
    dataset.loc[dataset["Age"] <=16, "Age"]=0,
    dataset.loc[(dataset["Age"] >16)&(dataset["Age"] <=26), "Age"]=1,
    dataset.loc[(dataset["Age"] >26)&(dataset["Age"] <=36), "Age"]=2,
    dataset.loc[(dataset["Age"] >36)&(dataset["Age"] <=62), "Age"]=3,
    dataset.loc[dataset["Age"] >62, "Age"]=4


# In[22]:


train.head()


# In[23]:


Pclass1=train[train["Pclass"]==1]["Embarked"].value_counts()
Pclass2=train[train["Pclass"]==2]["Embarked"].value_counts()
Pclass3=train[train["Pclass"]==3]["Embarked"].value_counts()
df=pd.DataFrame([Pclass1,Pclass2,Pclass3])
df.index=["1st class","2nd class","3rd class"]
df.plot(kind="bar",stacked=True,figsize=(10,5))


# In[24]:


for dataset in train_test_data:
    dataset["Embarked"]=dataset["Embarked"].fillna("S")


# In[25]:


train.head()


# In[26]:


embarked_mapping = {"S":0,"C":1,"Q":2}
for dataset in train_test_data:
    dataset["Embarked"]=dataset["Embarked"].map(embarked_mapping)


# In[27]:


test.head()


# In[28]:


train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"),inplace=True)
test["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"),inplace=True)


# In[29]:


facet=sns.FacetGrid(train,hue="Survived",aspect=4)
facet.map(sns.kdeplot,"Fare",shade=True)
facet.set(xlim=(0,train["Fare"].max()))
facet.add_legend()
plt.show()


# In[30]:


for dataset in train_test_data:
    dataset.loc[dataset["Fare"] <=17, "Fare"]=0,
    dataset.loc[(dataset["Fare"] >17)&(dataset["Fare"] <=30), "Fare"]=1,
    dataset.loc[(dataset["Fare"] >30)&(dataset["Fare"] <=100), "Fare"]=2,
    dataset.loc[dataset["Fare"] >100, "Fare"]=3


# In[31]:


train.head()


# In[32]:


for dataset in train_test_data:
    dataset["Cabin"]=dataset["Cabin"].str[:1]


# In[33]:


Pclass1=train[train["Pclass"]==1]["Cabin"].value_counts()
Pclass2=train[train["Pclass"]==2]["Cabin"].value_counts()
Pclass3=train[train["Pclass"]==3]["Cabin"].value_counts()
df=pd.DataFrame([Pclass1,Pclass2,Pclass3])
df.index=["1st class","2nd class","3rd class"]
df.plot(kind="bar",stacked=True,figsize=(10,5))


# In[34]:


cabin_mapping = {"A":0,"B":0.4,"C":0.8,"D":1.2,"E":1.6,"F":2,"G":2.4,"T":2.8}
for dataset in train_test_data:
    dataset["Cabin"]=dataset["Cabin"].map(cabin_mapping)


# In[35]:


train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"),inplace=True)
test["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"),inplace=True)


# In[36]:


test.head()


# In[37]:


train['FamilySize']=train["SibSp"]+train["Parch"]+1
test['FamilySize']=test["SibSp"]+test["Parch"]+1


# In[38]:


facet=sns.FacetGrid(train,hue="Survived",aspect=4)
facet.map(sns.kdeplot,"FamilySize",shade=True)
facet.set(xlim=(0,train["FamilySize"].max()))
facet.add_legend()
plt.show()


# In[39]:


Family_mapping = {1: 0,2 :0.4,3: 0.8,
                  4: 1.2,5: 1.6,6: 2,7: 2.4,
                  8: 2.8,9: 3.2,10: 3.6,11: 4.0}
for dataset in train_test_data:
    dataset["FamilySize"]=dataset["FamilySize"].map(Family_mapping)


# In[40]:


train.head()


# In[41]:


sex_mapping={"male":0,"female":1}
for dataset in train_test_data:
    dataset["Sex"]=dataset["Sex"].map(sex_mapping)


# In[42]:


train.head()


# In[43]:


features_drop=["Ticket","SibSp","Parch"]
train=train.drop(features_drop,axis=1)
test=test.drop(features_drop,axis=1)
train=train.drop(["PassengerId"],axis=1)


# In[44]:


train_data=train.drop("Survived",axis=1)
target=train["Survived"]
train_data.shape,target.shape


# In[45]:


train_data.head(10)


# In[46]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_Fold=KFold(n_splits=10,shuffle=True,random_state=0)


# In[47]:


clf=KNeighborsClassifier(n_neighbors=13)
scoring='accuracy'
score=cross_val_score(clf,train_data,target,cv=k_Fold,n_jobs=1,scoring=scoring)
print(score)


# In[48]:


round(np.mean(score)*100,2)


# In[49]:


clf=DecisionTreeClassifier()
scoring='accuracy'
score=cross_val_score(clf,train_data,target,cv=k_Fold,n_jobs=1,scoring=scoring)
print(score)


# In[50]:


round(np.mean(score)*100,2)


# In[51]:


clf=RandomForestClassifier(n_estimators=13)
scoring='accuracy'
score=cross_val_score(clf,train_data,target,cv=k_Fold,n_jobs=1,scoring=scoring)
print(score)


# In[52]:


round(np.mean(score)*100,2)


# In[53]:


clf= GaussianNB()
scoring='accuracy'
score=cross_val_score(clf,train_data,target,cv=k_Fold,n_jobs=1,scoring=scoring)
print(score)


# In[54]:


round(np.mean(score)*100,2)


# In[55]:


clf= SVC()
scoring='accuracy'
score=cross_val_score(clf,train_data,target,cv=k_Fold,n_jobs=1,scoring=scoring)
print(score)


# In[56]:


round(np.mean(score)*100,2)


# In[57]:


clf=SVC()
clf.fit(train_data,target)
test_data=test.drop("PassengerId",axis=1).copy()
prediction=clf.predict(test_data)


# In[65]:


submission =pd.DataFrame({"PassengerId":test["PassengerId"],"Survived":prediction}).to_csv('kaggle0.csv',index=False)


# In[67]:


submission_1=pd.read_csv("kaggle0.csv")


# In[68]:


submission_1.head()


# In[ ]:





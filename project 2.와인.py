#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[4]:


wine = load_wine()
wine_data = wine.data
wine_label = wine.target
print(wine_data)
print(wine_label)


# In[5]:


print(wine_data.shape)
print(wine_label.shape)


# ### 1. Decision Tree

# In[6]:


from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(wine_data,
                                                    wine_label,
                                                    test_size=0.2,
                                                    random_state=10)

decision_tree = DecisionTreeClassifier(random_state=10)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
print(classification_report(y_test, y_pred))


# ### 2. RandomForest

# In[7]:


from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(random_state=10)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
print(classification_report(y_test, y_pred))


# ### 3. Support Vector Machine

# In[8]:


from sklearn import svm

svm_model = svm.SVC()
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred))


# ### 4. SGD Classifier

# In[9]:


from sklearn.linear_model import SGDClassifier

sgd_model = SGDClassifier()
sgd_model.fit(X_train, y_train)
y_pred = sgd_model.predict(X_test)
print(classification_report(y_test, y_pred))


# ### 5. Logistic Regression

# In[10]:


from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)
print(classification_report(y_test, y_pred))


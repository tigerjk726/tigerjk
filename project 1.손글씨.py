#!/usr/bin/env python
# coding: utf-8

# # 프로젝트 (1) load_digits : 손글씨를 분류해 봅시다

# In[14]:


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[7]:


digits = load_digits()
digits_data = digits.data
digits_label = digits.target
print(digits_data)
print(digits_label)


# In[13]:


print(digits_data.shape)
print(digits_label.shape)


# ### 1. Decision Tree

# In[16]:


from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(digits_data,
                                                    digits_label,
                                                    test_size=0.2,
                                                    random_state=10)

decision_tree = DecisionTreeClassifier(random_state=10)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
print(classification_report(y_test, y_pred))


# ### 2. RandomForest

# In[17]:


from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(random_state=10)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
print(classification_report(y_test, y_pred))


# ### 3. Support Vector Machine

# In[18]:


from sklearn import svm

svm_model = svm.SVC()
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred))


# ### 4. SGD Classifier

# In[20]:


from sklearn.linear_model import SGDClassifier

sgd_model = SGDClassifier()
sgd_model.fit(X_train, y_train)
y_pred = sgd_model.predict(X_test)
print(classification_report(y_test, y_pred))


# ### 5. Logistic Regression

# In[24]:


from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)
print(classification_report(y_test, y_pred))


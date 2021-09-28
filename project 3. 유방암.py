#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[5]:


breast_cancer = load_breast_cancer()
breast_cancer_data = breast_cancer.data
breast_cancer_label = breast_cancer.target
print(breast_cancer_data)
print(breast_cancer_label)


# In[6]:


print(breast_cancer_data.shape)
print(breast_cancer_label.shape)


# ### 1. Decision Tree

# In[8]:


from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(breast_cancer_data,
                                                    breast_cancer_label,
                                                    test_size=0.2,
                                                    random_state=10)

decision_tree = DecisionTreeClassifier(random_state=10)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
print(classification_report(y_test, y_pred))


# ### 2. RandomForest

# In[10]:


from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(random_state=10)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
print(classification_report(y_test, y_pred))


# ### 3. Support Vector Machine

# In[12]:


from sklearn import svm

svm_model = svm.SVC()
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred))


# ### 4. SGD Classifier

# In[15]:


from sklearn.linear_model import SGDClassifier

sgd_model = SGDClassifier()
sgd_model.fit(X_train, y_train)
y_pred = sgd_model.predict(X_test)
print(classification_report(y_test, y_pred))


# ### 5. Logistic Regression

# In[18]:


from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)
print(classification_report(y_test, y_pred))


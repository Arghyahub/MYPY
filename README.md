
# Linear Regression
```python
from sklearn.linear_model import LinearRegression
```
`y = wx + b`
Where `w` is the weight and `b` is the bias

# Train test split 
It is a test splitting procedure 
```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2)
``` 

# Logistic regression
```python
from sklearn.linear_model import LogisticRegression
```
It is classification of either yes or no, we use something like a sigmoid function

`z = wx + b`
`g(z) = 1/(1 + e^[-z])`

# Panda - Numpy - Seaborn - LabelEncoder
**Numpy - Panda** 
```python
pd.crosstab(df.salary,df.left) # Creates total salary per left
pd.crosstab(df.salary,df.left).plot(kind='bar') # plots bar graph
```

**Label Encoder** 
 ```python
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()
inputs['company_n'] = le_company.fit_transform(inputs['company']) 
inputs['job_n'] = le_job.fit_transform(inputs['job']) 
inputs['degree_n'] = le_degree.fit_transform(inputs['degree']) 
print(le_company.classes_)
print(le_company.transform(le_company.classes_))
``` 
Output
`['abc pharma' 'facebook' 'google']`
`[0 1 2]`



# Multiclass Regression
```python
from sklearn.linear_model import LogisticRegression
```
**Logistic regression with multiple categories**

# Decision Tree
```python
from sklearn.preprocessing import LabelEncoder
```

# Decision Tree
It is a type of classifier that splits from nodes
```python
from sklearn.tree import DecisionTreeClassifier
```

# Support vector machine
Another Classifier
```python
from sklearn.svm import SVC
model = SVC()
```

















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

# K fold cross validation
 ```python
from sklearn.model_selection import cross_val_score
cross_val_score(LogisticRegression(solver='liblinear',multi_class='ovr'), digits.data, digits.target,cv=3)
cross_val_score(SVC(gamma='auto'), digits.data, digits.target,cv=3)
cross_val_score(RandomForestClassifier(n_estimators=40),digits.data, digits.target,cv=3)
``` 

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


# Confusion matrix in and seaborn
 ```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
import seaborn as sn
import matplotlib.pyplot as plt
plt.figure(figsize = (10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
``` 

# Min max Scaler
 ```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df['Income($)'] = scaler.transform(df[['Income($)']])
df.head(4)
``` 

# Standard Scaler
```python
from sklearn.preprocessing import StandardScaler
```

# Count vectorizer
#### Message is not numeric and computer understands numbers

So we will convert the words using CountVectorizer()
which will store the occurence of each word
 ```python
from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
X_train_count.toarray()
``` 

# Pipeline
 ```python
from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('Vectorizer', CountVectorizer()),
    ('nb',MultinomialNB())  
])
clf.fit(X_train,Y_train)
clf.score(X_test,Y_test)
``` 

# Principal component Analysis for reduction of data
 ```python
from sklearn.decomposition import PCA
pca = PCA(0.95) #retain 95% info
``` 


# Grid search CV or replate GridSearchCV to randomizedSearchCV
 ```python
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(svm.SVC(gamma='auto'), {
    'C': [1,10,20],
    'kernel': ['rbf','linear']
},cv=5,return_train_score=False)

clf.fit(iris.data,iris.target)
clf.cv_results_
df = pd.DataFrame(clf.cv_results_)
df[['param_C','param_kernel','mean_test_score']]

``` 

# Hyper Parameter Tuning 

```python
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    }
}

scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(iris.data, iris.target)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df

```

# Logistic regression
```python
from sklearn.linear_model import LogisticRegression
```
It is classification of either yes or no, we use something like a sigmoid function

`z = wx + b`
`g(z) = 1/(1 + e^[-z])`


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

# Random forest classifier
 ```python
from sklearn.ensemble import RandomForestClassifier
``` 


# K means clustering
 ```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3)
km.get_params()

y_predicted = km.fit_predict(df[['Age','Income($)']])
y_predicted

df['cluster'] = y_predicted
df.head(3)

# Now we will plot the elbow plot and find out which is the best number of clusters to form

k_rng = range(1,10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df)
    sse.append(km.inertia_)

plt.plot(k_rng,sse)
``` 
# Naive Byes

 ```python
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train,Y_train)
model.get_params()
model.score(X_test,Y_test)
``` 

# Multinomial Naive Byes
 ```python
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.get_params()
model.fit(X_train_count,Y_train)
``` 

# Gaussian Naive Byes
 ```python
from sklearn.naive_bayes import GaussianNB
``` 

# Lasso and Ridge regression
 ```python
from sklearn import linear_model
lasso_reg = linear_model.Lasso(alpha=100, max_iter=300, tol=0.1)
from sklearn.linear_model import Ridge
ridge_reg= Ridge(alpha=5, max_iter=100, tol=0.1)
``` 


# KNN
 ```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
``` 

# Bagging
```python
from sklearn.ensemble import BaggingClassifier

bag_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(), 
    n_estimators=100, 
    max_samples=0.8, 
    oob_score=True,
    random_state=0
)
bag_model.fit(X_train, Y_train)
bag_model.oob_score_
```
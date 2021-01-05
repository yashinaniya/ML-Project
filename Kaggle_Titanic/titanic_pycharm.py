import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

""" importing dataset """

# dataset = pd.read_csv(r'C:\Users\Yash\Desktop\datasets\train_new.csv')
dataset = pd.read_csv('train.csv')

test_data = pd.read_csv('test.csv')
combine = [dataset, test_data]
#dis = dataset.head()
#print(dis)
test_data.head()

for dat in combine:
    dat['Title'] = dat.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

for dat in combine:
    dat['Title'] = dat['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dat['Title'] = dat['Title'].replace('Mlle', 'Miss')
    dat['Title'] = dat['Title'].replace('Ms', 'Miss')
    dat['Title'] = dat['Title'].replace('Mme', 'Mrs')

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dat in combine:
    dat['Title'] = dat['Title'].map(title_mapping)
    dat['Title'] = dat['Title'].fillna(0)

dataset = dataset.drop(['Name', 'PassengerId', 'Cabin', 'Embarked', 'Ticket','Fare'], axis=1)

test_data = test_data.drop(['Name', 'PassengerId', 'Cabin', 'Embarked', 'Ticket', 'Fare'], axis=1)
combine = [dataset, test_data]
new = test_data.head()


X = dataset.iloc[:, 1:].values
X_test = test_data.iloc[:, :].values
y = dataset.iloc[:, 0].values

#print(X)
#print(X_test)
#print(y)
test_for_nan = test_data.info()
print(test_for_nan)

# test_data.info()

""" Data Visualization """
"""
ax = sns.countplot(x='Sex', hue='Survived', data=dataset)
#plt.show()

col = ['Survived', 'Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked']
no_of_rows = 2
no_of_col = 3
fig, axs = plt.subplots(no_of_rows, no_of_col, figsize=(no_of_col * 3.5, no_of_rows * 3))

for r in range(0, no_of_rows):
    for c in range(0, no_of_col):
        i = r * no_of_col + c
        ax = axs[r][c]
        sns.countplot(dataset[col[i]], hue=dataset["Survived"], ax=ax)
        ax.set_title(col[i], fontsize=14, fontweight='bold')
        ax.legend(title="survived", loc='upper center')

plt.tight_layout()
plt.show()
"""

# Taking care of missing data
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(dataset['Age'])
dataset['Age'] = imputer.transform(dataset['Age'])
imputer.fit(test_data['Age'])
test_data['Age'] = imputer.transform(test_data['Age'])
# print(X)
# print(X_test)

"""

"""
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
# Encoding the test set
c_t = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X_test = np.array(c_t.fit_transform(X_test))
# print(X)

"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)
# print(X)
# print(X_test)



# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X, y)


# Training the SVM model on the Training set
from sklearn.svm import SVC

classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X, y)

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X, y)

# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X, y)
"""

# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
classifier.fit(X, y)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(y_pred.reshape(len(y_pred), 1))

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#Importing the dataset
df1 = pd.read_csv('csv_result-Autism-Child-Data.csv')
df2 = pd.read_csv('csv_result-Autism-Adult-Data.csv')
df3 = pd.read_csv('csv_result-Autism-Adolescent-Data.csv')
df = pd.concat([df1,df2,df3], ignore_index = True)

#Data cleaning/EDA
df.describe()
df.info()

print(df.isnull().sum()) 
df.replace(to_replace = '?',value = np.nan ,inplace = True)
df['age'] = pd.to_numeric(df['age'], errors = 'coerce')
df = df.fillna(method = 'ffill')
df.drop(df[df.age > 100].index, inplace = True, axis = 0)
print(df.isnull().sum())

df.drop(['ethnicity','relation','age_desc','used_app_before','id'], axis = 1, inplace = True)
df.rename(columns = {'austim':'family member with PDD','jundice':'jaundice'}, inplace = True)

#Relation between child with autism and family member having PDD
plt.figure(figsize = (10,7))
sns.countplot(x = 'Class/ASD', data = df, hue = 'family member with PDD')
plt.show()

#Gender-wise classifcation of children with PDD 
plt.figure(figsize = (10,7))
sns.countplot(x = 'Class/ASD', data = df, hue = 'gender')
plt.show()

#Top 3 countries with most number of autistic children
plt.figure(figsize = (16,8))
g = sns.countplot(x = 'contry_of_res', data = df, hue = 'Class/ASD')
plt.setp(g.get_xticklabels(), rotation=90)
plt.legend(loc = 'upper center')
plt.show()

#Relation between children born with jaundice and autism
plt.figure(figsize = (10,7))
sns.countplot(x = 'jaundice', data = df, hue = 'Class/ASD')
plt.show()

#Correlation heatmap
plt.figure(figsize = (10,10))
sns.heatmap(df.corr())
plt.show()

df.drop(['contry_of_res'], axis = 1, inplace = True)

#Encoding the categorical variables
from sklearn.preprocessing import LabelEncoder
df = df.apply(LabelEncoder().fit_transform)

#Separating the Dependent and Independent variables
X = df.iloc[:, :15].values
y = df.iloc[:, 15].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Checking other metrics
from sklearn.metrics import accuracy_score,classification_report
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(accuracies.mean())
print(accuracies.std())

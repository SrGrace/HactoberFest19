
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
np.set_printoptions(precision=4)
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


##using wine dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Categorical.from_codes(wine.target, wine.target_names)

y = pd.Categorical.from_codes(wine.target, wine.target_names)

##We create a DataFrame containing both the features and classes.
df = X.join(pd.Series(y, name='class'))

##For every class, we create a vector with the means of each feature.
class_feature_means = pd.DataFrame(columns=wine.target_names)
for c, rows in df.groupby('class'):
    class_feature_means[c] = rows.mean()
class_feature_means


###Then, we plug the mean vectors (mi) into the equation from before in order to obtain the within class scatter matrix.


within_class_scatter_matrix = np.zeros((13,13))
for c, rows in df.groupby('class'):
    rows = rows.drop(['class'], axis=1)
    
    s = np.zeros((13,13))
for index, row in rows.iterrows():
        x, mc = row.values.reshape(13,1), class_feature_means[c].values.reshape(13,1)
        
        s += (x - mc).dot((x - mc).T)
    
        within_class_scatter_matrix += s
        
##we calculate the between class scatter matrix using the following formula.

feature_means = df.mean()
between_class_scatter_matrix = np.zeros((13,13))
for c in class_feature_means:    
    n = len(df.loc[df['class'] == c].index)
    
    mc, m = class_feature_means[c].values.reshape(13,1), feature_means.values.reshape(13,1)
    
    between_class_scatter_matrix += n * (mc - m).dot((mc - m).T)
    
    
##obtain linear discriminants
eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(within_class_scatter_matrix).dot(between_class_scatter_matrix))


'''
In order to ensure that the eigenvalue maps to the same eigenvector after sorting, we place them in a temporary array.

'''
pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]
pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
for pair in pairs:
    print(pair[0])
    
    
'''
Just looking at the values, it’s difficult to determine how much of the variance is explained by each component. 
Thus, we express it as a percentage.

'''

eigen_value_sums = sum(eigen_values)
print('Explained Variance')
for i, pair in enumerate(pairs):
    print('Eigenvector {}: {}'.format(i, (pair[0]/eigen_value_sums).real))
    
##we create a matrix W with the first two eigenvectors.
w_matrix = np.hstack((pairs[0][1].reshape(13,1), pairs[1][1].reshape(13,1))).real

X_lda = np.array(X.dot(w_matrix))

'''
creating and fitting an instance of the PCA class.

'''

le = LabelEncoder()
y = le.fit_transform(df['class'])


##we encode every class as a number so that we can incorporate the class labels into our plot.
##Then, we plot the data as a function of the two LDA components and use a different color for each class.
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.scatter(
    X_lda[:,0],
    X_lda[:,1],
    c=y,
    cmap='rainbow',
    alpha=0.7,
    edgecolors='b'
)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X, y)


'''
We can access the explained_variance_ratio_ property to view the percentage of the variance explained by each component.
'''
lda.explained_variance_ratio_

plt.xlabel('LD1')
plt.ylabel('LD2')
plt.scatter(
    X_lda[:,0],
    X_lda[:,1],
    c=y,
    cmap='rainbow',
    alpha=0.7,
    edgecolors='b'
)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X, y)

pca.explained_variance_ratio_

'''
PCA selected the components which would result in the highest spread (retain the most information) 
and not necessarily the ones which maximize the separation between classes.

'''

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=y,
    cmap='rainbow',
    alpha=0.7,
    edgecolors='b'
)

## split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_lda, y, random_state=1)

##As we can see, the Decision Tree classifier correctly classified everything in the test set.
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
confusion_matrix(y_test, y_pred)





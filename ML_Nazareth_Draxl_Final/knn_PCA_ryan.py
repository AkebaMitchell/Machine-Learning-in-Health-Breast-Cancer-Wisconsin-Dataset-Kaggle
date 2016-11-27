# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 04:58:20 2016

@author: ryank
"""

################ Running PCA anf kNN on Breast Cancer Wisconsin Data set 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier as knn
#from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


X = pd.read_csv('data.csv')
y = X['diagnosis']    # diagnosis labels 
lb = preprocessing.LabelBinarizer()
lb = lb.fit(y)
lb.classes_
y = lb.transform(y)

df = X.drop(['id','Unnamed: 32','diagnosis'], axis =1)
FeatureNames = np.asarray(df.columns.values)

X = df.values 
pca = PCA(n_components=30)
projectedAxes = pca.fit_transform(scale(X))
#   or do pca.fit(X) and the projectedAxes = pca.transform(scale(X))

plt.figure(1)
plt.suptitle('First two components')
plt.xlabel('PC_1')
plt.ylabel('PC_2')
plt.scatter(projectedAxes[:,0], projectedAxes[:,1], c = "#D06B36", s = 50,\
     alpha = 0.4, linewidth='0')

#print pca.components_
#print pca.explained_variance_
print ('')
print (100*pca.explained_variance_ratio_)
print('')


## looking at columns with loadings for first principal component - interesting
## to see that concavity, compactness have highest loadings and correlate with 
##radius, area, perimeter as you would expect 

comp1Loadings = np.asarray(pca.components_[0])[\
np.argsort( np.abs(pca.components_[0]))[::-1]][0:30]
comp1Names = np.asarray(FeatureNames)[np.argsort( \
np.abs(pca.components_[0]))[::-1]][0:30]
for i in range(0, 30):
    print ( "Column \"" , comp1Names[i] , "\" has a loading of: ", \
    comp1Loadings[i])
print('')

## looking at columns with loadings for second principal component - fractal 
#dimensionshave highest loadings 
print('')
comp1Loadings = np.asarray(pca.components_[1])[np.argsort( \
     np.abs(pca.components_[1]))[::-1]][0:30]
comp1Names = np.asarray(FeatureNames)[np.argsort( \
np.abs(pca.components_[1]))[::-1]][0:30]
for i in range(0, 30):
    print ( "Column \"" , comp1Names[i] , "\" has a loading of: ", \
    comp1Loadings[i])
print('')
          
    
# We will try 20 components as they account for approximately 97% variance 
  
X_train, X_test, y_train, y_test = tts(X, y, test_size =0.3, random_state=10)

n_neighbors = 10
weights = 'uniform'
clf = knn(n_neighbors, weights, algorithm='auto', metric='minkowski')

knn_PCA= Pipeline([('pca', PCA(n_components=20)),\
    ('knn', clf)])
knn_PCA.fit(X_train, y_train)

y_predicted = knn_PCA.predict(X_test)
print(classification_report(y_test, y_predicted))

'''# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].

x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h)
z = knn_PCA.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
z = z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights))

plt.show()
''' 
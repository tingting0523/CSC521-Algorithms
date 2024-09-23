# Customers are clustered based on their annual income and spending score, 
# and the distribution of each cluster is visualized in a scatter plot using different colored dots. 
# This helps identify different types of customers and behavior patterns.
# 将顾客根据其年度收入和消费评分进行聚类，并通过不同颜色的点在散点图中可视化每个聚类的分布。这有助于识别顾客的不同类型和行为模式
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.display import Image
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

dataset = pd.read_csv('/Users/jolie/Desktop/school/98-SCSU/3-course-521/Research/Mall_Customers.csv')

# We take only two Features (Annual Income and Spending Score) to classify customer type. 选择数据框的第 4 列及其后的所有列作为特征 X
X = dataset.iloc[:, 3:]

import scipy.cluster.hierarchy as hc
import matplotlib.pyplot as plt
from pylab import rcParams

#设置图表大小
rcParams['figure.figsize'] = 10, 5

# Using Dendogram to find the optimal number of clusters
dendogram  = hc.dendrogram(hc.linkage(X, method = 'ward'))  # use ward method to calculate the Euclidean distance between points
# create Dendogram tree 
plt.title('Dendrogram') 
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

dendogram  = hc.dendrogram(hc.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.axhline(200, c='r', linestyle='--')   # Add horizontal lines to help determine the number of clusters  How to evaluate the value of Euclidean Distances???
plt.show()


# Using Agglomerative hierarchical clustering Approch
from sklearn.cluster import AgglomerativeClustering
hc_Agg = AgglomerativeClustering(n_clusters = 5, linkage = 'ward')  # create model  # use ward method to calculate the distance between points
y_hc = hc_Agg.fit_predict(X)  # Fit and predict clusters

# Visualizing the clusters
# X.iloc[y_hc == 0, 0] and X.iloc[y_hc == 0, 1] represents the first and second features of all samples of the ith cluster
# s:point size 
plt.scatter(X.iloc[y_hc == 0, 0], X.iloc[y_hc == 0, 1], s = 100, c = 'red', label = 'Careful')  
plt.scatter(X.iloc[y_hc == 1, 0], X.iloc[y_hc == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X.iloc[y_hc == 2, 0], X.iloc[y_hc == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(X.iloc[y_hc == 3, 0], X.iloc[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(X.iloc[y_hc == 4, 0], X.iloc[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Sensible')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
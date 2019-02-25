import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("College.csv",index_col=0)

# Create a scatterplot of Grad.Rate versus Room.Board where the points are colored by the Private column.
sns.set_style('whitegrid')
sns.lmplot('Room.Board','Grad.Rate',data=df, hue='Private',
           palette='coolwarm',height=6,aspect=1,fit_reg=False)

# Create a scatterplot of F.Undergrad versus Outstate where the points are colored by the Private column.
sns.set_style('whitegrid')
sns.lmplot('Outstate','F.Undergrad',data=df, hue='Private',
           palette='coolwarm',height=6,aspect=1,fit_reg=False)

# Create a stacked histogram showing Out of State Tuition based on the Private column.
sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',height=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)

# Create a similar histogram for the Grad.Rate column.
sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',height=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)

df[df['Grad.Rate'] > 100]
df['Grad.Rate']['Cazenovia College'] = 100
df[df['Grad.Rate'] > 100]

# re-do the histogram visualization to make sure it actually went through.
sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',height=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)

from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=2)
kmeans.fit(df.drop('Private',axis=1))
print(kmeans.cluster_centers_)

plt.show()
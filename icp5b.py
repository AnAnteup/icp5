from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import silhouette_score

data= pd.read_csv('College.csv')

data=data.drop(columns=['Name','Private'],axis=1)

for i in range(2,18):
    kmean = KMeans(n_clusters=i, max_iter=300)
    x= kmean.fit_predict(data)

    sil=silhouette_score(data,x)
    print( " Silhouette analysis for KMeans clustering",i," on sample data is",sil)

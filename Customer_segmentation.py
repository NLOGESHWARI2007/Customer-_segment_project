import pandas as pd
from sklearn.cluster import KMeans

data = {
    'Income': [15, 16, 17, 60, 62, 65, 90, 95, 100],
    'SpendingScore': [39, 81, 6, 55, 52, 59, 90, 88, 95]
}

df = pd.DataFrame(data)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Income', 'SpendingScore']])

print(df)

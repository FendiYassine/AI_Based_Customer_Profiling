
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date
import ydata_profiling as yd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

file_path = r'C:\Users\torjm\OneDrive\Bureau\coding_folder\Axe finance stage\CC GENERAL.csv'
df = pd.read_csv(file_path)

dt=df[['CUST_ID','BALANCE','BALANCE_FREQUENCY','PURCHASES','ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES','CASH_ADVANCE','PURCHASES_FREQUENCY','ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY','CASH_ADVANCE_FREQUENCY','CASH_ADVANCE_TRX','PURCHASES_TRX','CREDIT_LIMIT','TENURE']]

dt['Customer_Type'] = ''
dt.loc[(dt['ONEOFF_PURCHASES_FREQUENCY'] > 0) & (dt['PURCHASES_INSTALLMENTS_FREQUENCY'] == 0), 'Customer_Type'] = 'One-off Purchases'
dt.loc[(dt['ONEOFF_PURCHASES_FREQUENCY'] == 0) & (dt['PURCHASES_INSTALLMENTS_FREQUENCY'] >0), 'Customer_Type'] = 'Installments purchases'
dt.loc[(dt['ONEOFF_PURCHASES_FREQUENCY'] > 0) & (dt['PURCHASES_INSTALLMENTS_FREQUENCY'] > 0), 'Customer_Type'] = 'Both'
dt.loc[(dt['ONEOFF_PURCHASES_FREQUENCY'] == 0) & (dt['PURCHASES_INSTALLMENTS_FREQUENCY'] == 0), 'Customer_Type'] = 'None'

dt_drop=dt[(abs(dt['ONEOFF_PURCHASES']+dt['INSTALLMENTS_PURCHASES']-dt['PURCHASES'])//1)!=0]

dt=dt.drop(dt_drop.index)

y=dt[(dt['PURCHASES_TRX']==0)&(dt['PURCHASES']!=0)]
dt=dt.drop(y.index)
dt=dt.dropna()
max_allowed_frequency = 1
dt['CASH_ADVANCE_FREQUENCY'] = dt['CASH_ADVANCE_FREQUENCY'].apply(lambda x: min(x, max_allowed_frequency))


# In[612]:


df_copy=dt.copy()
df_copy = df_copy.drop(columns=['Customer_Type'])
#df_copy.to_csv(r'C:\Users\torjm\OneDrive\Bureau\project\reactproject\Customer.csv', index=False)
"""

def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))
    return outliers

# Check outliers for each column
for column in df_copy.columns:
    if df_copy[column].dtype in ['float64', 'int64']:  # Check if the column is numerical
        outliers = detect_outliers_iqr(df_copy[column])
        print(f"Number of outliers in {column}: {outliers.sum()}")


from sklearn.ensemble import IsolationForest

# Initialize Isolation Forest
clf = IsolationForest(contamination=0.1)  # contamination is the proportion of outliers in the dataset
outliers = clf.fit_predict(df_copy.select_dtypes(include=['float64', 'int64']))

# Convert -1 labels (outliers) to boolean for filtering
outliers = outliers == -1

# Handling outliers
#Replace outliers with median of the column (you can adjust this strategy as needed)
for col in df_copy.select_dtypes(include=['float64', 'int64']).columns:
    median_value = df_copy[col].median()
    df_copy.loc[outliers, col] = median_value

"""
# In[615]:

# **Dealing with Skewed data**

# In[617]:


from sklearn.preprocessing import QuantileTransformer
cols_right_skewed = df_copy.skew(numeric_only=True)[df_copy.skew(numeric_only=True) > 1].index.tolist()
cols_left_skewed = df_copy.skew(numeric_only=True)[df_copy.skew(numeric_only=True) < -1].index.tolist()
print('Right skewed columns: ', cols_right_skewed)
print('Left skewed columns: ', cols_left_skewed)

columns_to_transform = cols_right_skewed + cols_left_skewed

pt = QuantileTransformer(n_quantiles=100, output_distribution='normal')
transformed_data = pd.DataFrame(pt.fit_transform(df_copy[columns_to_transform]),
                                columns= columns_to_transform)

df_copy[columns_to_transform] = transformed_data


# In[618]:


# test
dt_skew = dt.skew(numeric_only=True).to_frame(name='skewness before')
df_copy_skew = df_copy.skew(numeric_only=True).to_frame(name='skewness after Power Transformation')
combined_df = pd.concat([dt_skew, df_copy_skew], axis=1)


# In[619]:





# In[620]:





# In[621]:


df_copy.isnull().sum()


# In[622]:


df_copy=df_copy.dropna() 


# In[623]:




# **MutiCollinearity Problem**

# In[624]:


dtf=df_copy.select_dtypes(include=[np.number])


# In[625]:


scaler = StandardScaler()
X = scaler.fit_transform(dtf)


# In[626]:


#PCA
seed = 42
pca = PCA(n_components=2, random_state=seed)

X_pca = pca.fit_transform(dtf)


# # K-means

# In[627]:


k_range = range(2, 10)
kmeans_per_k = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=seed).fit(X_pca)
    kmeans_per_k.append(kmeans)


# In[628]:

"""
#Elbow method
def plot_optimal_k(kmeans_per_k: list):

    # Inertia
    inertias = [model.inertia_ for model in kmeans_per_k]

    # Silhouette score
    silhouette_scores = [silhouette_score(X_pca, model.labels_) for model in kmeans_per_k]

    fig, ax = plt.subplots(figsize=(16, 6))
    # Plot elbow score
    plt.subplot(1, 2, 1)
    sns.lineplot(x=k_range, y=inertias, marker="o")
    plt.ylabel("Inertia")
    plt.xlabel("K values")
    
    # Plot silhouette score
    plt.subplot(1, 2, 2)
    sns.lineplot(x=k_range, y=silhouette_scores, marker="o")
    plt.ylabel("Silhouette score")
    plt.xlabel("K values")

    plt.show()

# Calling the function
plot_optimal_k(kmeans_per_k)

"""
# In[629]:


best_model = kmeans_per_k[2]
y_kmeans = best_model.predict(X_pca)
print(y_kmeans)


# In[630]:


def plot_kmeans_clusters(clusterer, y_kmeans):
    cluster_colors = ["#e9c46a", "#a2d2ff", "#9d4edd", "#81b29a"]
    labels = ["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3", "Centroids"]

    # Percentage labels
    unique, counts = np.unique(y_kmeans, return_counts=True)
    count_dict = dict(zip(labels, counts))
    total = sum(count_dict.values())
    label_dict = {key: round(value/total*100, 2) for key, value in count_dict.items()}

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10), layout="constrained")

    # Silhouette plot
    s_viz = SilhouetteVisualizer(clusterer, colors=cluster_colors, ax=ax1)
    s_viz.fit(X_pca)
    s_viz.finalize()
    s_viz.ax.set_title("Silhouette diagram", **title)
    s_viz.ax.set(xlabel="Silhouette coefficient", ylabel="Cluster label")
    s_viz.ax.legend(loc="lower right", frameon=True, fancybox=True, facecolor="w")

    # Scatter plot
    for i in unique:
        ax2.scatter(X_pca[y_kmeans==i, 0], X_pca[y_kmeans==i, 1], c=cluster_colors[i], linewidth=0.4, edgecolor="k", s=10)
    ax2.scatter(clusterer.cluster_centers_[:, 0], clusterer.cluster_centers_[:, 1], c="k", s=10)
    ax2.set_title("Clusters distribution", **title)
    ax2.set(xticks=[], yticks=[])
    ax2.legend(labels, ncols=len(labels), loc="lower center", bbox_to_anchor=(0.5, -0.1), markerscale=3)
    # Bar plot
    ax3 = plt.subplot(2, 2, (3, 4))
    bars = ax3.bar(x=unique, height=counts, color=cluster_colors)
    ax3.set_title("Clusters distribution (count)", **title)
    ax3.set_ylabel("Count")
    ax3.set_xticks(ticks=unique, labels=labels[:-1])
    for bar in bars:
        ax3.annotate("{:.2%}".format(bar.get_height() / total), xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
                xytext=(0, 0), textcoords="offset points", ha="center", va="bottom")

    fig.tight_layout()
    plt.show()

# Calling the function
#plot_kmeans_clusters(best_model, y_kmeans)


# In[631]:


def clustering_evaluation(X, y_kmeans):
    s_score = np.round(silhouette_score(X, y_kmeans), 3)
    db_score = np.round(davies_bouldin_score(X, y_kmeans), 3)
    ch_score = np.round(calinski_harabasz_score(X, y_kmeans), 3)
    
    # Printing the metrics
    print("Clustering Evaluation")
    print("-"*25)
    print("Silhouette score:", s_score)
    print("Davies-Bouldin index:", db_score)
    print("Calinski-Harabasz index:", ch_score)
    
    # Creating a dataframe with the metrics
    metrics_df = pd.DataFrame({
        'Silhouette Score': [s_score],
        'Davies-Bouldin Index': [db_score],
        'Calinski-Harabasz Index': [ch_score]
    })
    
    return metrics_df
#clustering_evaluation(X_pca, y_kmeans)


# In[632]:


dx={'cluster':y_kmeans}
res1=pd.DataFrame(dx)
res2=pd.DataFrame(df_copy['CUST_ID'])
result = pd.concat([res2, res1], axis=1)

def search_old_cust_id(cust_id):
    x = result[result['CUST_ID'].astype(str) == str(cust_id)]
    return x['cluster'].iloc[0]


def predict_cluster(d):
    global result
    #PCA
    a=d['CUST_ID']
    d= d.drop(columns=['CUST_ID'])
    print("Shape of d:", d.shape)
    X = scaler.fit_transform(d)
    print("Shape of X:", X.shape)
    #seed = 42
    #pca = PCA(n_components=2, random_state=seed)
    #print("Shape of X:", X.shape)
    d_pca = pca.fit_transform(X)
    y = best_model.predict(d_pca)
    dx={'cluster':y}
    res1=pd.DataFrame(dx)
    res2=pd.DataFrame(a)
    res = pd.concat([res2, res1], axis=1)
    result = pd.concat([result, res], ignore_index=True)
    return search_old_cust_id(a)

i=clustering_evaluation(X_pca,y_kmeans)
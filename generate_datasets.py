# %%
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
# %%
def generate_elipse_data(N=500, random_state=42):
    np.random.seed(random_state)
    X = 2*(np.random.rand(N, 2)-0.5)
    y = 1*(2*(X[:,0]**2 + 2*X[:,1]**2 - 2*X[:,1]*X[:,0])<0.8)
    return X, y

def generate_gaussians_distributions(N=500, random_state=42):
    np.random.seed(random_state)
    X1 = np.random.multivariate_normal([0.5, 0.5], [[0.1,-0.085],[-0.085,0.1]], N//2)
    X2 = np.random.multivariate_normal([-0.25, -0.25], [[0.1,0],[0,0.1]], N//2)
    X = np.append(X1, X2, axis=0)
    y = np.append(np.zeros(N//2), np.ones(N//2))
    return X, y
# %%

# %%
X, y = generate_gaussians_distributions()
# X, y = generate_elipse_data()
# %%
df_gaus = pd.DataFrame([X[:, 0], X[:, 1], y]).T.rename(
    columns={
        0: 'var1',
        1: 'var2',
        2: 'labels'
    }
)
df_gaus['labels'] = df_gaus['labels'].astype(int)

df_gaus['labels'] = df_gaus['labels'].apply(lambda x: ['class_1', 'class_2'][x])
# %%
f, ax = plt.subplots()
df_gaus[df_gaus['labels']=='class_1'].plot.scatter('var1', 'var2', ax=ax, c='r')
df_gaus[df_gaus['labels']=='class_2'].plot.scatter('var1', 'var2', ax=ax, c='g')
# %%
df_gaus
# %%
df_gaus.to_csv('datasets/dataset_1.csv')
# %%

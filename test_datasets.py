# %%
from matplotlib import pyplot as plt
import pandas as pd
# %%
df = pd.read_csv('datasets/dataset_1.csv')
# %%
f, ax = plt.subplots()
df[df['labels']=='class_1'].plot.scatter('var1', 'var2', ax=ax, c='r')
df[df['labels']=='class_2'].plot.scatter('var1', 'var2', ax=ax, c='g')
# %%

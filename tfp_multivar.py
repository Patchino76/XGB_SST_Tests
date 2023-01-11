#%%
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import matplotlib.pyplot as plt
import numpy as np

print("TF version:", tf.__version__)
print("TFP version:", tfp.__version__)
# %%
normal_diag = tfd.MultivariateNormalDiag(loc = [0,5], scale_diag=[10,15])
normal_diag
# %%
normal_diag.sample(10)
# %%
plt_sample = normal_diag.sample(10000)
plt.scatter(plt_sample[:,0], plt_sample[:,1],marker='.',alpha=0.1)
plt.axis('equal')
plt.show()

# %%
normal_diag2 = tfd.MultivariateNormalDiag(loc = [[0,5],[0,5],[0,5]], scale_diag=[[1,2],[2,1],[2,2]])
normal_diag2
# %%
samples = normal_diag2.sample(5)
samples
# %%
normal_diag2.log_prob(samples)
# %%
plt_sample_batch = normal_diag2.sample(10000)
plt_sample_batch.shape
# %%
fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10,3))
titles = ['Cov1', 'Cov2', 'Cov3']

for i, (ax, title) in enumerate(zip(ax, titles)):
    samples = plt_sample_batch[:,i,:] 
    ax.scatter(samples[:,0],samples[:,1], marker='.', alpha=0.1)
    ax.set_title(title)
# %%

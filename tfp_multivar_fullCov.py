#%%
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import seaborn as sns
import matplotlib.pyplot as plt

print("TF version:", tf.__version__)
print("TFP version:", tfp.__version__)
# %%
spherical_2d_gaussian = tfd.MultivariateNormalDiag(loc=[0., 0.])

N = 1000
x = spherical_2d_gaussian.sample(N)
x1 = x[:, 0]
x2 = x[:, 1]
sns.jointplot(x1, x2, kind='kde', space=0, )
# %%
# creating a two-dimensional Gaussian with non-diagonal covariance
mu = [0., 0.]  # mean
scale_tril = [[1.,  0.],
              [0.6, 0.8]]

sigma = tf.matmul(tf.constant(scale_tril), tf.transpose(tf.constant(scale_tril)))  # covariance matrix
print(sigma)
# %%
nonspherical_2d_gaussian = tfd.MultivariateNormalTriL(loc=mu, scale_tril=scale_tril)
nonspherical_2d_gaussian
# %%
nonspherical_2d_gaussian.mean()
# %%
nonspherical_2d_gaussian.covariance()
# %%
x = nonspherical_2d_gaussian.sample(N)
x1 = x[:, 0]
x2 = x[:, 1]
sns.jointplot(x1, x2, kind='kde', space=0, color='r');
# %%

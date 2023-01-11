#%%
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from scipy.signal import find_peaks, argrelmax

#%%

distribution = tfp.distributions.MultivariateNormalDiag(loc=[0., 0.], scale_diag=[1., 1.])

#%%
# Sample from the distribution using the sample_chain function
samples = tfp.mcmc.sample_chain(
  num_results=100,
  current_state=tf.zeros(2),
  kernel=tfp.mcmc.RandomWalkMetropolis(target_log_prob_fn=distribution.log_prob),
  num_burnin_steps=500
)
# %%
# Use a peak finding algorithm on the samples to identify the peaks
peaks = argrelmax(samples[0].numpy())
# %%

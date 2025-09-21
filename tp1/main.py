import numpy as np
from multiarmbandits.core import experiment, thompson_sampling, ucb
from multiarmbandits.visualization import plot_regret, animate_thompson, animate_ucb
from multiarmbandits.core import create_bandit

# Thompson Sampling Example
K = 5
T = 1000
n_exps = 50
regret = experiment(thompson_sampling, K=K, T=T, n_exps=n_exps)
plot_regret(regret, label="Thompson Regret")

arms, means = create_bandit(K, seed=42)
rewards, alpha_hist, beta_hist, chosen_arm = thompson_sampling(arms, K, T)
animate_thompson(
    alpha_hist, beta_hist, chosen_arm
)  # The GIF may take some time to load (Available in plots/ directory)

# UCB Example
regret_ucb = experiment(ucb, K=K, T=T, n_exps=n_exps)
plot_regret(regret_ucb, label="UCB Regret")
arms_ucb, means_ucb = create_bandit(K, seed=42)
rewards_ucb, X_hist, B_hist, chosen_arm_ucb = ucb(arms_ucb, K, T)
animate_ucb(
    X_hist, B_hist, chosen_arm_ucb, means_ucb
)  # The GIF may take some time to load (Available in plots/ directory)

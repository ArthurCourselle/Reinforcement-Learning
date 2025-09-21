import numpy as np
from scipy.stats import beta
from .arm import ArmBernoulli


def create_bandit(K: int, seed: int = 1) -> tuple[list[ArmBernoulli], np.ndarray]:
    """
    Create a list of Bernoulli arms with random means.
    Args:
        K (int): number of arms
        seed (int): random seed
    Returns:
        tuple[list[ArmBernoulli], np.ndarray]: arms and their means
    """
    np.random.seed(seed)
    means = np.random.random(K)
    arms = [ArmBernoulli(m) for m in means]
    return arms, means


def naive(
    MAB: list[ArmBernoulli], N: int, K: int, T: int, verbose: bool = False
) -> list[float]:
    """
    Naive algorithm: sample each arm N//K times, then exploit the best.
    Args:
        MAB (list[ArmBernoulli]): list of arms
        N (int): total samples for exploration
        K (int): number of arms
        T (int): total rounds
        verbose (bool): print details
    Returns:
        list[float]: rewards
    """
    n_per_arm = N // K
    estimates = [np.mean([arm.sample() for _ in range(n_per_arm)]) for arm in MAB]
    rewards = [
        sample for arm in MAB for sample in [arm.sample() for _ in range(n_per_arm)]
    ]
    best_arm = int(np.argmax(estimates))
    if verbose:
        print(f"The best arm is : {best_arm} among {[k for k in range(K)]}")
    rewards.extend([MAB[best_arm].sample() for _ in range(T - N)])
    if verbose:
        m_star = max([arm.mean for arm in MAB])
        regret = T * m_star - np.sum(rewards)
        print(f"The regret is : {regret}")
    return rewards


def ucb(
    MAB: list[ArmBernoulli], K: int, T: int, verbose: bool = False
) -> tuple[list[float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Upper Confidence Bound (UCB) algorithm.
    Args:
        MAB (list[ArmBernoulli]): list of arms
        K (int): number of arms
        T (int): total rounds
        verbose (bool): print details
    Returns:
        tuple: rewards, X_hist, B_hist, chosen_arm
    """
    estimates = [0.0] * K
    counts = [0] * K
    rewards = []
    X_hist = np.zeros((T, K))
    B_hist = np.zeros((T, K))
    chosen_arm = np.zeros(T, dtype=int)
    for t in range(T):
        arm = int(
            np.argmax(
                [
                    estimate + np.sqrt(2 * np.log(t + 1) / c) if c > 0 else float("inf")
                    for estimate, c in zip(estimates, counts)
                ]
            )
        )
        reward = MAB[arm].sample()
        rewards.append(reward)
        counts[arm] += 1
        estimates[arm] += (reward - estimates[arm]) / counts[arm]
        X_hist[t, :] = estimates
        B_hist[t, :] = [
            np.sqrt(2 * np.log(t + 1) / c) if c > 0 else 0.0 for c in counts
        ]
        chosen_arm[t] = arm
    if verbose:
        m_star = max([arm.mean for arm in MAB])
        regret = T * m_star - np.sum(rewards)
        print(f"The regret is : {regret}")
    return rewards, X_hist, B_hist, chosen_arm


def thompson_sampling(
    MAB: list[ArmBernoulli], K: int, T: int
) -> tuple[list[float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Thompson Sampling algorithm (Bayesian approach).
    Args:
        MAB (list[ArmBernoulli]): list of arms
        K (int): number of arms
        T (int): total rounds
    Returns:
        tuple: rewards, alpha_hist, beta_hist, chosen_arm
    """
    successes = np.ones(K)
    failures = np.ones(K)
    rewards = []
    alpha_hist = np.zeros((T, K))
    beta_hist = np.zeros((T, K))
    chosen_arm = np.zeros(T, dtype=int)
    for t in range(T):
        index = [beta.rvs(a + 1, b + 1) for a, b in zip(successes, failures)]
        arm = int(np.argmax(index))
        reward = MAB[arm].sample()
        rewards.append(reward)
        if reward == 1:
            successes[arm] += 1
        else:
            failures[arm] += 1
        alpha_hist[t, :] = successes
        beta_hist[t, :] = failures
        chosen_arm[t] = arm
    return rewards, alpha_hist, beta_hist, chosen_arm


def experiment(
    algorithm,
    K: int,
    T: int,
    n_exps: int = 200,
    **kwargs,
) -> np.ndarray:
    """
    Run n_exps experiments with the given algorithm.
    Args:
        algorithm (callable): bandit algorithm function
        K (int): number of arms
        T (int): total rounds
        n_exps (int): number of experiments
        **kwargs: extra arguments for algorithm
    Returns:
        np.ndarray: mean regret over all runs
    """
    arms, means = create_bandit(K)
    m_star = max(means)
    all_regret = []
    for _ in range(n_exps):
        if algorithm.__name__ == "naive":
            rewards = algorithm(arms, kwargs.get("N", K), K, T)
        else:
            result = algorithm(arms, K, T)
            rewards = result[0] if isinstance(result, tuple) else result
        cumulative_reward = np.cumsum(rewards)
        regret_t = np.arange(1, T + 1) * m_star - cumulative_reward
        all_regret.append(regret_t)
    mean_regret = np.mean(all_regret, axis=0)
    return mean_regret

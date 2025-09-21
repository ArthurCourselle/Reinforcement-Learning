import numpy as np
import pytest
from multiarmbandits.core import create_bandit, naive, ucb, thompson_sampling


@pytest.fixture
def bandit_fixture():
    # Fixed means for reproducibility
    means = np.array([0.1, 0.5, 0.9])

    class MockArm:
        def __init__(self, mean):
            self.mean = mean

        def sample(self):
            return self.mean

    arms = [MockArm(m) for m in means]
    return arms, means


def test_naive_bandit(bandit_fixture):
    arms, means = bandit_fixture
    rewards = naive(arms, N=9, K=3, T=12)
    # Should exploit the best arm (mean=0.9)
    assert np.isclose(np.mean(rewards[-3:]), 0.9)


def test_ucb_bandit(bandit_fixture):
    arms, means = bandit_fixture
    rewards, _, _, chosen = ucb(arms, K=3, T=12)
    # Should eventually select the best arm
    assert means[np.argmax(means)] in [arms[i].mean for i in chosen[-3:]]


def test_thompson_bandit(bandit_fixture):
    arms, means = bandit_fixture
    rewards, _, _, chosen = thompson_sampling(arms, K=3, T=12)
    # Should exploit the best arm most often
    best_arm = np.argmax(means)
    assert np.sum(chosen == best_arm) > 0

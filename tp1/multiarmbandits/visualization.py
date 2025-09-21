import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional


def animate_ucb(
    X_hist: np.ndarray,
    B_hist: np.ndarray,
    chosen_arm: np.ndarray,
    means: np.ndarray,
    interval: int = 10,
) -> None:
    """
    Animate UCB estimates and confidence bounds over time.
    Args:
        X_hist (np.ndarray): UCB estimates history
        B_hist (np.ndarray): UCB confidence bounds history
        chosen_arm (np.ndarray): Chosen arm at each round
        means (np.ndarray): True means of each arm
        interval (int): Animation interval in ms
    Returns:
        None
    """
    K, T = X_hist.shape[1], X_hist.shape[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    bars_X = ax1.bar(range(K), X_hist[0], color="skyblue", edgecolor="black")
    for m in means:
        ax1.axhline(m, linestyle="--", linewidth=1, alpha=0.5)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(range(K))
    ax1.set_xlabel("Arms")
    ax1.set_ylabel("$X_a$")
    ax1.set_title("$X_a(t)$")
    bars_B = ax2.bar(range(K), B_hist[0], color="lightgreen", edgecolor="black")
    ax2.set_ylim(0, np.max(B_hist) * 1.1)
    ax2.set_xticks(range(K))
    ax2.set_xlabel("Arms")
    ax2.set_ylabel("$B_a$")
    ax2.set_title("$B_a(t)$")
    txt = fig.text(0.5, 0.02, "", ha="center")

    def update(frame):
        for a, bar in enumerate(bars_X):
            bar.set_height(X_hist[frame, a])
            bar.set_color("skyblue")
        bars_X[chosen_arm[frame]].set_color("orange")
        for a, bar in enumerate(bars_B):
            bar.set_height(B_hist[frame, a])
            bar.set_color("lightgreen")
        bars_B[chosen_arm[frame]].set_color("orange")
        txt.set_text(f"t = {frame+1}, chosen arm = {chosen_arm[frame]}")
        return list(bars_X) + list(bars_B) + [txt]

    anim = FuncAnimation(fig, update, frames=T, interval=interval, blit=False)
    anim.save("ucb_animation.gif", writer="pillow")
    plt.close(fig)


def animate_thompson(
    alpha_hist: np.ndarray,
    beta_hist: np.ndarray,
    chosen_hist: np.ndarray,
    interval: int = 50,
) -> None:
    """
    Animate posterior Beta distributions for Thompson Sampling.
    Args:
        alpha_hist (np.ndarray): Alpha parameters history
        beta_hist (np.ndarray): Beta parameters history
        chosen_hist (np.ndarray): Chosen arm at each round
        interval (int): Animation interval in ms
    Returns:
        None
    """
    K, T = alpha_hist.shape[1], alpha_hist.shape[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, K - 0.5)
    ax.set_xticks(range(K))
    ax.set_xlabel("Arms")
    ax.set_ylabel("Probability Density")
    ax.set_title("Posterior Distributions (Thompson Sampling)")
    txt = fig.text(0.5, 0.02, "", ha="center")

    def update(frame):
        ax.cla()
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.5, K - 0.5)
        ax.set_xticks(range(K))
        ax.set_xlabel("Arms")
        ax.set_ylabel("Probability Density")
        ax.set_title("Posterior Distributions (Thompson Sampling)")
        # Draw all arms in blue
        for a in range(K):
            data = np.random.beta(alpha_hist[frame, a], beta_hist[frame, a], 1000)
            parts = ax.violinplot(
                dataset=data,
                positions=[a],
                widths=0.8,
                showmeans=False,
                showextrema=False,
                showmedians=False,
            )

            for pc in parts["bodies"]:  # type: ignore[not-iterable]
                pc.set_facecolor("skyblue")
                pc.set_edgecolor("black")
        # Highlight chosen arm in orange
        a_chosen = chosen_hist[frame]
        data = np.random.beta(
            alpha_hist[frame, a_chosen], beta_hist[frame, a_chosen], 1000
        )
        chosen_parts = ax.violinplot(
            dataset=data,
            positions=[a_chosen],
            widths=0.8,
            showmeans=False,
            showextrema=False,
            showmedians=False,
        )

        for pc in chosen_parts["bodies"]:  # type: ignore[not-iterable]
            pc.set_facecolor("orange")
            pc.set_edgecolor("black")
        txt.set_text(f"t = {frame+1}, chosen arm = {a_chosen}")
        return list(chosen_parts["bodies"]) + [txt]  # type: ignore[not-iterable]

    anim = FuncAnimation(fig, update, frames=T, interval=interval, blit=False)
    anim.save("thompson_animation.gif", writer="pillow")
    plt.close(fig)


def plot_regret(
    regret: np.ndarray,
    label: str = "Regret",
    xlabel: str = "Rounds",
    ylabel: str = "Cumulative Regret",
) -> None:
    """
    Plot the cumulative regret curve.
    Args:
        regret (np.ndarray): Regret values to plot
        label (str): Label for the curve
        xlabel (str): Label for x-axis
        ylabel (str): Label for y-axis
    """
    plt.figure(figsize=(8, 5))
    plt.plot(regret, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{label} Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{label.lower().replace(' ', '_')}_curve.png")
    plt.close()

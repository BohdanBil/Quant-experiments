"""Charts for the risk report. Each function returns a matplotlib Figure."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter

from .risk import sampling_spread, var_es

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, BLUE_DARK, ORANGE, CRITICAL = "#2a78d6", "#1c5cab", "#eb6834", "#d03b3b"

STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10, "text.color": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlecolor": INK, "axes.labelcolor": INK_2, "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": MUTED, "ytick.labelcolor": MUTED,
    "legend.frameon": False, "legend.labelcolor": INK_2, "legend.fontsize": 9,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
}


def save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)


# "\$" stops matplotlib from reading text between two dollar signs as a math formula
def _aud(x, _=None):
    """Compact axis label: A$12.5K."""
    sign, x = ("−" if x < 0 else ""), abs(x)
    return f"{sign}A\\${x / 1000:g}K" if x >= 1000 else f"{sign}A\\${x:,.0f}"


def _aud_exact(x):
    return f"A\\${x:,.0f}"


def _subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")


def _figure_titles(fig, title, subtitle, height_in):
    """Title and subtitle above all panels, kept inside the figure."""
    left = fig.subplotpars.left
    fig.text(left, 1 - 0.1 / height_in, title, fontsize=12, fontweight="bold", va="top")
    fig.text(left, 1 - 0.4 / height_in, subtitle, color=INK_2, fontsize=9, va="top")


def fan_chart(main, comparison, value):
    """Percentile bands of the simulated portfolio value over time."""
    days = np.arange(main.values.shape[1])
    p = np.percentile(main.values, [1, 5, 25, 50, 75, 95, 99], axis=0)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.fill_between(days, p[0], p[6], color=BLUE, alpha=0.10, lw=0, label="98% of paths")
        ax.fill_between(days, p[1], p[5], color=BLUE, alpha=0.16, lw=0, label="90% of paths")
        ax.fill_between(days, p[2], p[4], color=BLUE, alpha=0.26, lw=0, label="50% of paths")
        ax.plot(days, p[3], color=BLUE_DARK, lw=2, label="Median")
        ax.plot(days, np.percentile(comparison.values, 1, axis=0), color=ORANGE, lw=1.6,
                label=f"{comparison.model} model, worst 1%")
        ax.set_xlim(0, days[-1])
        ax.set_xlabel("Trading days ahead")
        ax.yaxis.set_major_formatter(FuncFormatter(_aud))
        ax.legend(loc="upper left")
        ax.set_title(f"Portfolio value over the next {days[-1]} trading days "
                     f"(today: {_aud_exact(value)})", pad=24)
        _subtitle(ax, f"{main.model} model, {len(main.values):,} paths. The lightest band's "
                      "lower edge is its worst 1%.")
    return fig


def pnl_distribution(main, comparison, day):
    """Histogram of P&L at `day` on a log scale, with the main model's VaR and ES marked."""
    a, b = main.pnl(day), comparison.pnl(day)
    bins = np.linspace(*np.percentile(np.concatenate([a, b]), [0.005, 99.95]), 120)
    share_a, share_b = np.full(len(a), 1 / len(a)), np.full(len(b), 1 / len(b))
    var99, es975 = var_es(a, 0.01)[0], var_es(a, 0.025)[1]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.hist(a, bins, weights=share_a, color=BLUE, alpha=0.15)
        ax.hist(a, bins, weights=share_a, histtype="step", color=BLUE, lw=1.4)
        ax.hist(b, bins, weights=share_b, histtype="step", color=ORANGE, lw=1.4)
        ax.set_yscale("log")
        for x, label, y in [(-var99, f"VaR 99%\n{_aud_exact(var99)} loss", 0.97),
                            (-es975, f"ES 97.5%\n{_aud_exact(es975)} loss", 0.80)]:
            ax.axvline(x, color=INK_2, lw=1, ls=(0, (4, 3)))
            ax.annotate(label, xy=(x, y), xycoords=("data", "axes fraction"), xytext=(-5, 0),
                        textcoords="offset points", ha="right", va="top", color=INK_2, fontsize=9)
        ax.xaxis.set_major_formatter(FuncFormatter(_aud))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:g}%"))
        ax.set_xlabel(f"Profit or loss after {day} trading days")
        ax.set_ylabel("Share of paths (log scale)")
        ax.legend(handles=[Patch(facecolor=BLUE + "26", edgecolor=BLUE, label=main.model),
                           Line2D([], [], color=ORANGE, lw=1.4, label=comparison.model)],
                  loc="upper right")
        ax.set_title(f"Distribution of {day}-day profit and loss", pad=24)
        _subtitle(ax, f"Log scale makes the tails visible. Dashed lines: {main.model} risk measures.")
    return fig


def convergence(sim, day, sizes=(100, 250, 500, 1000, 2500, 5000, 10_000)):
    """How the spread of VaR/ES estimates shrinks as the number of simulated paths grows."""
    pnl = sim.pnl(day)
    sizes = [n for n in sizes if len(pnl) // n >= 10]
    height = 4.6
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(9.5, height))
        fig.subplots_adjust(top=1 - 1.2 / height, wspace=0.3)
        for ax, (label, alpha, k) in zip(axes, [("VaR 99%", 0.01, 0), ("ES 97.5%", 0.025, 1)]):
            spread = np.array([sampling_spread(pnl, alpha, n)[k] for n in sizes])
            full = var_es(pnl, alpha)[k]
            ax.fill_between(sizes, spread[:, 0], spread[:, 2], color=BLUE, alpha=0.15, lw=0)
            ax.plot(sizes, spread[:, 1], color=BLUE, lw=2)
            ax.axhline(full, color=INK_2, lw=1, ls=(0, (4, 3)))
            ax.annotate(f"All {len(pnl):,} paths: {_aud_exact(full)}", xy=(sizes[-1], full),
                        xytext=(0, 5), textcoords="offset points", ha="right", color=INK_2,
                        fontsize=9)
            ax.annotate(f"100 paths: {_aud_exact(spread[0, 0])}\nto {_aud_exact(spread[0, 2])}",
                        xy=(sizes[0], spread[0, 2]), xytext=(6, 0), textcoords="offset points",
                        va="top", color=INK_2, fontsize=9)
            ax.set_xscale("log")
            ax.set_xticks(sizes, [f"{n:,}" for n in sizes], fontsize=8)
            ax.minorticks_off()
            ax.yaxis.set_major_formatter(FuncFormatter(_aud))
            ax.set_xlabel("Simulated paths")
            ax.set_title(f"{label}, {day}-day horizon", fontsize=11)
        _figure_titles(fig, "How many simulations are enough?",
                       "Shaded: range of 90% of repeated runs. Line: typical run. "
                       "Dashed: estimate from all paths.", height)
    return fig


def backtest_chart(bt, models, alpha=0.01):
    """Daily losses against each model's 1-day VaR forecast, with breaches marked."""
    losses = bt.losses * 100
    top = 1.05 * max(losses.max(), *(bt.var(m, alpha).max() * 100 for m in models))
    height = 2.9 * len(models) + 1.5
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(models), 1, figsize=(9.5, height), sharex=True, sharey=True)
        fig.subplots_adjust(top=1 - 1.5 / height, hspace=0.35)
        for ax, model in zip(np.atleast_1d(axes), models):
            var = bt.var(model, alpha) * 100
            hit = losses > var
            ax.vlines(bt.dates, 0, np.clip(losses, 0, None), color=AXIS, lw=0.5)
            ax.plot(bt.dates, var, color=BLUE, lw=1.1)
            ax.scatter(bt.dates[hit], losses[hit], s=16, color=CRITICAL, edgecolor=SURFACE,
                       linewidth=0.6, zorder=3)
            ax.set_title(f"{model}: {hit.sum()} breaches, {hit.mean():.1%} of days", fontsize=11)
            ax.set_ylim(0, top)
            ax.set_ylabel("Daily loss")
            ax.yaxis.set_major_locator(MultipleLocator(2))
            ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
        _figure_titles(fig, f"Backtest: VaR {1 - alpha:.0%} should be breached on {alpha:.0%} of days",
                       "Each day's forecast uses only earlier data.", height)
        fig.legend(handles=[Line2D([], [], color=AXIS, lw=2, label="Daily loss"),
                            Line2D([], [], color=BLUE, lw=1.5, label="VaR 99% forecast"),
                            Line2D([], [], color=CRITICAL, marker="o", ls="", markersize=5,
                                   markeredgecolor=SURFACE, label="Breach: loss above VaR")],
                   loc="upper left", bbox_to_anchor=(fig.subplotpars.left - 0.012, 1 - 0.62 / height),
                   ncols=3)
    return fig


def risk_contributions(tickers, weights, shares, day):
    """Each stock's share of the money vs. its share of the portfolio's Expected Shortfall."""
    order = np.argsort(shares)
    y = np.arange(len(tickers))
    h = 0.18
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(tickers) + 1.5))
        money = ax.barh(y + h / 2 + 0.02, np.asarray(weights)[order] * 100, height=h,
                        color=MUTED, label="Portfolio weight")
        risk = ax.barh(y - h / 2 - 0.02, np.asarray(shares)[order] * 100, height=h,
                       color=BLUE, label=f"Share of risk ({day}-day ES 97.5%)")
        for bars in (money, risk):
            ax.bar_label(bars, fmt="%.0f%%", padding=3, color=INK_2, fontsize=9)
        ax.set_yticks(y, np.asarray(tickers)[order])
        ax.tick_params(axis="y", length=0, labelcolor=INK_2)
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=2)
        ax.set_title("Where the risk comes from", pad=24)
        _subtitle(ax, "Euler allocation: each stock's average loss in the worst 2.5% of scenarios.")
    return fig

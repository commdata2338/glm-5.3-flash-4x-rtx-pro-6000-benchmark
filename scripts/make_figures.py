"""Draw the paper's figures from data/*.csv into figures/ (PNG at 200 dpi and SVG).

  python3 scripts/make_figures.py

One style for all figures:
  - 7.4 inches wide with fixed margins, so each figure has the same scale. The paper shows each figure 740 pixels
    wide (the `width` of its `img` tag).
  - IBM Plex Sans (scripts/fonts/, SIL Open Font License), no text below 9.5 points.
  - One line below the plot and no line at the left. Light horizontal lines show the values.
  - The two recipes (TensorFold and Jovian Judgement r24) have thick lines. The TensorFold fork, Jovian Judgement
    r28.1 and the official vLLM have thin lines.
  - A legend above the panels, panel letters, units in the axis labels, a comma in each number of four digits.

Colors are four slots of a color-blind-checked categorical palette; each series also has its own marker, so
identity never rests on color alone. The two configurations of the official vLLM share one hue; the marker and the
line style show which is which. TensorFold and the TensorFold fork share one hue in the same way, and so do the
two releases of Jovian Judgement. Every plotted value is also in a table in the README (scripts/make_tables.py makes
those tables from the same data).

The figure numbers follow the sequence of the paper: Figures 1 to 6 are in the body, Figure 7 is in Appendix A,
Figure 8 is in Appendix C, and Figures 9 to 11 are in Appendix G. Figures 7, 9, 10, and 11 are diagrams, not charts.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullLocator  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT, FONTS = ROOT / "data", ROOT / "figures", Path(__file__).resolve().parent / "fonts"
for font in sorted(FONTS.glob("*.ttf")):
    font_manager.fontManager.addfont(str(font))
FAMILY = "IBM Plex Sans" if any(FONTS.glob("IBMPlexSans-*.ttf")) else "DejaVu Sans"

INK, MUTED, RULE, GRID = "#1a1a1a", "#5d5b55", "#8f8d86", "#e7e5df"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
WIDTH = 7.4   # inches, for each figure
SMALL = 9.5   # points: value labels, notes, and long panel titles
STYLE = {  # run id: (label, color, marker, line style, line width, marker size)
    "tensorfold": ("TensorFold", BLUE, "o", "-", 2.1, 6.0),
    # The TensorFold fork has the hue of TensorFold. The marker and the line style show which is which.
    "tensorfold_fork_best": ("TensorFold fork", BLUE, "D", (0, (4, 2)), 1.3, 4.4),
    "vllm_links_on_16": ("Jovian Judgement r24", ORANGE, "s", "-", 2.1, 5.6),
    # Release r28.1 has the hue of release r24. The marker and the line style show which is which.
    "jovian_r281_16": ("Jovian Judgement r28.1", ORANGE, "P", (0, (4, 2)), 1.3, 5.4),
    "vllm_links_on_8": ("Jovian Judgement r24, links on", ORANGE, "s", "-", 2.1, 5.6),
    "vllm_links_off_8": ("Jovian Judgement r24, links off", AQUA, "^", "-", 2.1, 6.2),
    # One hue for the official vLLM. The marker and the line style show its two configurations.
    "official_default_16_pass2": ("Official vLLM, default", VIOLET, "v", (0, (4, 2)), 1.3, 4.8),
    "official_pcie_16_pass2": ("Official vLLM, tuned", VIOLET, "D", "-", 1.3, 4.4),
}
# The six configurations of the speed tests, in the sequence of the tables. The sparkDash tests show the second
# run of the official vLLM.
SERVERS = ["tensorfold", "tensorfold_fork_best", "vllm_links_on_16", "jovian_r281_16", "official_default_16_pass2",
           "official_pcie_16_pass2"]
# The servers of the long-context chart. Jovian Judgement r28.1 gave no speed in that test (see NO_VALUE).
LONG_SERVERS = [run for run in SERVERS if run != "jovian_r281_16"]
NO_VALUE = {"jovian_r281_16"}   # the test ran, but no time interval had the decode of all its requests

plt.rcParams.update({
    "font.family": FAMILY, "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "semibold",
    "axes.labelsize": SMALL, "axes.titlepad": 7, "axes.labelpad": 5, "axes.edgecolor": RULE, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.axisbelow": True, "xtick.color": RULE, "ytick.color": RULE, "xtick.labelcolor": MUTED,
    "ytick.labelcolor": MUTED, "xtick.labelsize": SMALL, "ytick.labelsize": SMALL, "xtick.major.size": 3,
    "ytick.major.size": 0, "ytick.major.pad": 5, "text.color": INK, "axes.labelcolor": MUTED,
    "legend.frameon": False, "legend.fontsize": SMALL, "lines.linewidth": 1.6, "lines.markersize": 5.4,
    "lines.markeredgecolor": "white", "lines.markeredgewidth": 0.8, "figure.dpi": 100, "savefig.dpi": 200,
    "savefig.facecolor": "white", "svg.fonttype": "path",
})
thousands = FuncFormatter(lambda v, _: f"{v:,.0f}")


def read(name):
    with open(DATA / name, newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(name)


def series(ax, run, points):
    name, color, marker, line, width, size = STYLE[run]
    ax.plot([x for x, _ in points], [y for _, y in points], color=color, marker=marker, linestyle=line,
            linewidth=width, markersize=size, label=name, zorder=4 if width > 2 else 3)


def legend(fig, runs, left, ncol=3, y=0.995):
    """The legend above the panels. Its left edge is the left edge of the plots."""
    handles = [Line2D([], [], color=STYLE[r][1], marker=STYLE[r][2], linestyle=STYLE[r][3], linewidth=STYLE[r][4],
                      markersize=STYLE[r][5], label=STYLE[r][0]) for r in runs]
    fig.legend(handles=handles, loc="upper left", ncol=ncol, bbox_to_anchor=(left - 0.012, y), columnspacing=1.6,
               handlelength=2.3, labelspacing=0.4, borderaxespad=0.2)


def log2_axis(ax, ticks, labels=None):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks, labels or [str(t) for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())


def rows_axis(ax):
    """An axis with one row for each server and a value axis below: light vertical lines show the values."""
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.tick_params(axis="y", length=0, labelcolor=INK, labelsize=10)


FORK_NOTE = "Jovian Judgement is the vLLM fork by local-inference-lab."
NOTE_ROOM = 0.24   # inches below a chart for the line that says what Jovian Judgement is


def with_note(height, bottom, top):
    """The height and the two margins of a chart with one more line below it. The plots keep their size."""
    taller = height + NOTE_ROOM
    return taller, (bottom * height + NOTE_ROOM) / taller, 1 - (1 - top) * height / taller


def fork_note(fig):
    """The line below a chart that shows Jovian Judgement: whose fork it is."""
    fig.text(0.012, 0.06 / fig.get_size_inches()[1], FORK_NOTE, va="bottom", fontsize=SMALL, color=MUTED)


def row_label(run):
    """The name of a server in two lines, for the row of a chart."""
    return STYLE[run][0].replace(", ", "\n").replace("Judgement r", "Judgement\nr")


def sparkdash(file, run, x, y, **filters):
    return sorted((float(r[x]), float(r[y])) for r in read(file)
                  if r["run"] == run and all(r[k] == v for k, v in filters.items()))


# The long-context test: for each server, the runs of each group of sampler settings (file, run or arm names).
LONG_RUNS = {
    "tensorfold": {"k20": ("waves", ("tensorfold_session3",)), "off": ("waves", ("tensorfold", "tensorfold_session3")),
                   "p1": ("waves", ("tensorfold",))},
    "tensorfold_fork_best": {"k20": ("arms", ("fork-best",)), "off": ("arms", ("fork-best",)),
                             "p1": ("arms", ("fork-best",))},
    "vllm_links_on_16": {"k20": ("waves", ("vllm_links_on_16_session3",)),
                         "off": ("waves", ("vllm_links_on_16", "vllm_links_on_16_session3")),
                         "p1": ("waves", ("vllm_links_on_16",))},
    "jovian_r281_16": {},
    "official_default_16_pass2": {"k20": ("waves", ("official_default_16_session5",)),
                                  "off": ("waves", ("official_default_16", "official_default_16_session5")),
                                  "p1": ("waves", ("official_default_16",))},
    "official_pcie_16_pass2": {"k20": ("waves", ("official_pcie_16_session5",)),
                               "off": ("waves", ("official_pcie_16", "official_pcie_16_session5")),
                               "p1": ("waves", ("official_pcie_16",))},
}
SAMPLER = {"k20": ("0.95", "20"), "off": ("0.95", "off"), "p1": ("1.0", "off")}


def long_cells(run, setting, field="total_tok_s"):
    """(concurrent requests, the mean of the runs of that cell) for 4, 8, and 16 requests."""
    if setting not in LONG_RUNS[run]:
        return []
    name, sources = LONG_RUNS[run][setting]
    top_p, top_k = SAMPLER[setting]
    found = defaultdict(list)
    for r in read("concurrent_waves.csv" if name == "waves" else "fork_arms.csv"):
        if (r["run" if name == "waves" else "arm"] in sources and r["probe"] == "long_context_sampled"
                and r["top_p"] == top_p and r["top_k"] == top_k and int(r["concurrency"]) in (4, 8, 16)
                and r[field] != ""):
            found[int(r["concurrency"])].append(float(r[field]))
    return sorted((c, sum(v) / len(v)) for c, v in found.items())


def figure_three_loads():
    """Figure 1. Three loads on each server: the best server is not the same for each load."""
    titles = ["A  Short code answer\n1 request, greedy", "B  Cold 64K prompt\nprefill",
              "C  Shared 56K context\n16 requests\ntop_p 0.95, top_k 20"]
    limits = [(580, [0, 250, 500]), (16000, [0, 6000, 12000]), (1600, [0, 600, 1200])]
    height, bottom, top = with_note(3.7, 0.14, 0.82)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, height), sharey=True)
    for panel, ax in enumerate(axes):
        limit, ticks = limits[panel]
        for position, run in zip(range(len(SERVERS))[::-1], SERVERS):
            if panel == 0:
                value = dict(sparkdash("decode_sparkdash.csv", run, "concurrency", "total_tok_s", output_type="code"))[1]
            elif panel == 1:
                value = dict(sparkdash("prefill_sparkdash.csv", run, "target_tokens", "prefill_tok_s"))[65536]
            else:
                value = dict(long_cells(run, "k20")).get(16)
            if value is None:
                ax.text(limit * 0.03, position, "no value" if run in NO_VALUE else "no test", va="center", color=MUTED,
                        fontsize=SMALL, style="italic")
                continue
            ax.barh(position, value, height=0.56, color=STYLE[run][1], zorder=3)
            ax.text(value + limit * 0.025, position, f"{value:,.0f}", va="center", fontsize=SMALL)
        ax.set(xlim=(0, limit), ylim=(-0.62, len(SERVERS) - 0.38), xticks=ticks)
        ax.xaxis.set_major_formatter(thousands)
        ax.set_title(titles[panel], loc="left", fontsize=SMALL, linespacing=1.25)
        ax.set_xlabel("Prompt tokens/s" if panel == 1 else "Total tokens/s")
        rows_axis(ax)
    axes[0].set_yticks(range(len(SERVERS))[::-1], [row_label(run) for run in SERVERS])
    fig.subplots_adjust(left=0.175, right=0.975, top=top, bottom=bottom, wspace=0.20)
    fork_note(fig)
    save(fig, "figure1-three-loads")


def figure_decode():
    """Figure 2. The decode test of sparkDash on the six configurations, with a logarithmic speed axis."""
    left = 0.105
    height, bottom, top = with_note(4.8, 0.105, 0.82)
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, height), sharey=True)
    panels = (("prose", "A  Prose"), ("code", "B  Code"), ("structured", "C  Count task"), ("json", "D  JSON"))
    for ax, (kind, title) in zip(axes.flat, panels):
        for run in SERVERS:
            series(ax, run, sparkdash("decode_sparkdash.csv", run, "concurrency", "total_tok_s", output_type=kind))
        log2_axis(ax, [1, 2, 4, 8, 16])
        ax.set_yscale("log", base=2)
        ax.set_yticks([250, 500, 1000, 2000])
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(thousands)
        ax.set(xlim=(0.9, 18), ylim=(160, 2500))
        ax.set_title(title, loc="left")
    fig.supylabel("Total decode, tokens/s (log scale)", fontsize=SMALL, color=MUTED, x=0.012)
    fig.supxlabel("Concurrent requests (log scale)", fontsize=SMALL, color=MUTED,
                  y=(0.012 * 4.8 + NOTE_ROOM) / height)
    legend(fig, SERVERS, left)
    fig.subplots_adjust(left=left, right=0.985, bottom=bottom, top=top, hspace=0.36, wspace=0.07)
    fork_note(fig)
    save(fig, "figure2-decode-by-output-type")


def figure_prefill():
    """Figure 3. Prefill of one cold prompt on the six configurations. The speed axis starts at 5,000."""
    left = 0.115
    height, bottom, top = with_note(3.5, 0.15, 0.74)
    fig, ax = plt.subplots(figsize=(WIDTH, height))
    for run in SERVERS:
        series(ax, run, sparkdash("prefill_sparkdash.csv", run, "prompt_tokens", "prefill_tok_s"))
    log2_axis(ax, [8192, 16384, 32768, 65536, 131072, 262144], ["8K", "16K", "32K", "64K", "128K", "256K"])
    ax.set(xlim=(7400, 290000), ylim=(5000, 12000), yticks=list(range(5000, 12001, 1000)),
           xlabel="Cold prompt size, tokens (log scale)", ylabel="Prefill, prompt tokens/s")
    ax.yaxis.set_major_formatter(thousands)
    ax.set_title("The speed axis starts at 5,000", loc="left", fontsize=SMALL, fontweight="normal", color=MUTED)
    legend(fig, SERVERS, left)
    fig.subplots_adjust(left=left, right=0.985, bottom=bottom, top=top)
    fork_note(fig)
    save(fig, "figure3-cold-prompt-prefill")


def figure_long_context():
    """Figure 4. Long shared context with thinking on: decode for three groups of sampler settings, and first token."""
    left = 0.10
    panels = [("k20", "total_tok_s", "A  Decode: top_p 0.95, top_k 20"),
              ("off", "total_tok_s", "B  Decode: top_p 0.95, no top_k"),
              ("p1", "total_tok_s", "C  Decode: top_p 1.0, no top_k"),
              ("off", "median_ttft_s", "D  First token, context in the cache\ntop_p 0.95, no top_k")]
    height, bottom, top = with_note(5.2, 0.095, 0.81)
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, height))
    for ax, (setting, field, title) in zip(axes.flat, panels):
        decode = field == "total_tok_s"
        ends = []
        for run in LONG_SERVERS:
            points = long_cells(run, setting, field)
            if points:
                series(ax, run, points)
                if points[-1][0] == 16:
                    ends.append(points[-1][1])
        log2_axis(ax, [4, 8, 16])
        ax.set(xlim=(3.6, 22.5), ylim=(0, 1200 if decode else 6),
               yticks=list(range(0, 1201, 200)) if decode else [0, 2, 4, 6])
        ax.set_ylabel("Decode, tokens/s" if decode else "First token, seconds")
        if decode:
            ax.yaxis.set_major_formatter(thousands)
        ax.set_title(title, loc="left", fontsize=SMALL, linespacing=1.25)
        # The value of the highest line and of the lowest line at 16 requests. The table has each value.
        for value in (max(ends), min(ends)):
            ax.annotate(f"{value:,.0f}" if decode else f"{value:.1f}", (16, value), xytext=(7, 0),
                        textcoords="offset points", va="center", fontsize=SMALL)
    if not long_cells("official_default_16_pass2", "k20"):
        axes[0, 0].text(0.04, 0.87, "Official vLLM: no test", transform=axes[0, 0].transAxes, fontsize=SMALL,
                        color=MUTED, style="italic")
    alone = long_cells("tensorfold", "p1")[0]
    axes[1, 0].annotate(f"{alone[1]:.0f}", alone, xytext=(8, 5), textcoords="offset points", fontsize=SMALL)
    axes[1, 0].text(0.22, 0.075, "TensorFold: no test at 8 or 16", transform=axes[1, 0].transAxes, fontsize=SMALL,
                    color=MUTED, style="italic")
    for ax in axes[1]:
        ax.set_xlabel("Concurrent requests (log scale)")
    legend(fig, LONG_SERVERS, left)
    fig.subplots_adjust(left=left, right=0.985, bottom=bottom, top=top, hspace=0.50, wspace=0.25)
    fork_note(fig)
    save(fig, "figure4-long-context-thinking-on")


def figure_gpu_links():
    """Figure 5. Jovian Judgement with the direct GPU links off and on: prefill, decode, and cold first token."""
    left = 0.11
    runs = ["vllm_links_off_8", "vllm_links_on_8"]
    height, bottom, top = with_note(4.7, 0.105, 0.87)
    fig = plt.figure(figsize=(WIDTH, height))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
    a, b, c = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    for run in runs:
        series(a, run, sparkdash("prefill_sparkdash.csv", run, "prompt_tokens", "prefill_tok_s"))
    log2_axis(a, [8192, 16384, 32768, 65536, 131072, 262144], ["8K", "16K", "32K", "64K", "128K", "256K"])
    a.set(xlim=(7400, 290000), ylim=(0, 12000), yticks=[0, 4000, 8000, 12000],
          xlabel="Cold prompt size, tokens (log scale)", ylabel="Prefill, tokens/s")
    a.yaxis.set_major_formatter(thousands)
    a.set_title("A  Prefill of a cold prompt", loc="left", fontsize=SMALL)
    waves = [r for r in read("concurrent_waves.csv") if r["probe"] == "long_context_sampled" and r["top_p"] == "1.0"
             and r["top_k"] == "off"]
    cold = defaultdict(list)
    for r in read("chat_reuse_ttft.csv"):
        if r["case"] == "cold":
            cold[(r["run"], r["nominal_size"])].append(float(r["ttft_s"]))
    for offset, run in zip((-0.17, 0.17), runs):
        color = STYLE[run][1]
        for i, concurrency in enumerate((4, 8)):
            value = next(float(r["total_tok_s"]) for r in waves if r["run"] == run and int(r["concurrency"]) == concurrency)
            b.bar(i + offset, value, width=0.3, color=color, zorder=3)
            b.annotate(f"{value:,.0f}", (i + offset, value), xytext=(0, 3), textcoords="offset points", ha="center",
                       fontsize=SMALL)
        for i, size in enumerate(("9000", "56000")):
            value = sum(cold[(run, size)]) / len(cold[(run, size)])
            c.bar(i + offset, value, width=0.3, color=color, zorder=3)
            c.annotate(f"{value:.1f}", (i + offset, value), xytext=(0, 3), textcoords="offset points", ha="center",
                       fontsize=SMALL)
    b.set(xlim=(-0.55, 1.55), ylim=(0, 1000), xticks=[0, 1], yticks=[0, 250, 500, 750, 1000],
          xlabel="Concurrent requests", ylabel="Decode, tokens/s")
    b.set_xticklabels(["4", "8"])
    b.yaxis.set_major_formatter(thousands)
    b.set_title("B  Decode, 56K context, thinking on\ntop_p 1.0, no top_k", loc="left", fontsize=SMALL, linespacing=1.25)
    c.set(xlim=(-0.55, 1.55), ylim=(0, 14), xticks=[0, 1], yticks=[0, 4, 8, 12],
          xlabel="Cold prompt size, tokens", ylabel="First token, seconds")
    c.set_xticklabels(["13.5K", "83.6K"])
    c.set_title("C  First token, cold prompt", loc="left", fontsize=SMALL)
    for ax in (b, c):
        ax.tick_params(axis="x", length=0)
    legend(fig, runs, left, ncol=2)
    fig.subplots_adjust(left=left, right=0.985, bottom=bottom, top=top, hspace=0.82, wspace=0.28)
    fork_note(fig)
    save(fig, "figure5-jovian-judgement-gpu-links")


def figure_quality():
    """Figure 6. Task accuracy through the pi agent: the share of the tasks that pass, with the 95% interval."""
    rows = read("quality_pi_summary.csv")
    order = [("tensorfold", "TensorFold\n(EXL3, 4 bits)", "tensorfold"),
             ("tensorfold_fork_best", "TensorFold fork\n(EXL3, 4 bits)", "tensorfold_fork_best"),
             ("vllm_links_on_8", "Jovian Judgement\nr24 (NVFP4)", "vllm_links_on_16"),
             ("vllm_links_on_8_run2", "Jovian Judgement\nr24, second run", "vllm_links_on_16"),
             ("jovian_r281_8", "Jovian Judgement\nr28.1 (NVFP4)", "jovian_r281_16"),
             ("official_default_16", "Official vLLM,\ndefault (NVFP4)", "official_default_16_pass2")]
    order = [(run, label, style) for run, label, style in order if any(r["run"] == run for r in rows)]
    left = 0.19
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 0.9 + 0.62 * len(order) + NOTE_ROOM), sharey=True)
    for ax, dataset in zip(axes, ("HumanEval+", "MBPP+")):
        tasks = None
        for position, (run, _, style) in zip(range(len(order))[::-1], order):
            cell = next(r for r in rows if r["run"] == run and r["data_set"] == dataset)
            tasks = cell["tasks"]
            color, marker = STYLE[style][1], STYLE[style][2]
            for offset, key, face in ((0.16, "base_tests", "white"), (-0.16, "all_tests", color)):
                value = float(cell[f"{key}_percent"])
                low, high = float(cell[f"{key}_ci95_low"]), float(cell[f"{key}_ci95_high"])
                ax.plot([low, high], [position + offset] * 2, color=color, linewidth=1.6, zorder=2, solid_capstyle="butt")
                ax.plot([value], [position + offset], marker=marker, color=color, markerfacecolor=face,
                        markeredgecolor=color, markeredgewidth=1.3, markersize=6.5, linestyle="none", zorder=3)
                ax.text(high + 0.6, position + offset, f"{value:.1f}", va="center", fontsize=SMALL)
        ax.set_yticks(range(len(order))[::-1], [label for _, label, _ in order])
        ax.set(ylim=(-0.6, len(order) - 0.4), xlim=(76, 104.5), xticks=[80, 85, 90, 95, 100])
        rows_axis(ax)
        ax.set_title(f"{dataset} ({tasks} tasks)", loc="left", fontsize=SMALL)
        ax.set_xlabel("Tasks that pass, %")
    marks = [Line2D([], [], marker="o", color=INK, markerfacecolor="white", markeredgecolor=INK, markeredgewidth=1.3,
                    markersize=6.5, linestyle="-", linewidth=1.6, label="Base tests"),
             Line2D([], [], marker="o", color=INK, markerfacecolor=INK, markeredgecolor=INK, markersize=6.5,
                    linestyle="-", linewidth=1.6, label="Base tests and the added tests of EvalPlus")]
    fig.legend(handles=marks, loc="upper left", ncol=2, bbox_to_anchor=(left - 0.012, 0.995), handlelength=2.6,
               columnspacing=1.6, borderaxespad=0.2)
    height = fig.get_size_inches()[1]
    fig.subplots_adjust(left=left, right=0.985, bottom=(0.56 + NOTE_ROOM) / height, top=1 - 0.69 / height,
                        wspace=0.07)
    fork_note(fig)
    save(fig, "figure6-quality-through-an-agent")


def figure_copy_paths():
    """Figure 7. Schematic: how a tensor gets from one GPU to a different GPU with the direct links off and on."""
    rates = json.loads((DATA / "gpu_copy_paths.json").read_text())["pairs"]
    direct = sum(r["direct_gib_per_s"] for r in rates) / len(rates)
    staged = sum(r["staged_gib_per_s"] for r in rates) / len(rates)
    box = dict(boxstyle="round,pad=0.02,rounding_size=0.10", linewidth=0.7, edgecolor="#b3b1aa", facecolor="#f7f6f2")
    source, destination = 2.7, 7.3  # the x position of the two GPUs and of their PCIe links
    arrow = dict(arrowstyle="-|>", mutation_scale=13, linewidth=2.0, zorder=5, shrinkA=0, shrinkB=0)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.5))
    for ax in axes:
        ax.set(xlim=(0, 10), ylim=(0, 8))
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((2.2, 6.3), 5.6, 1.0, **box))
        ax.text(5.0, 6.8, "Host memory", ha="center", va="center")
        ax.add_patch(FancyBboxPatch((0.6, 3.5), 8.8, 1.3, **box))
        ax.text(5.0, 4.36, "CPU root complex\nwith the IOMMU", ha="center", va="center", fontsize=SMALL, color=MUTED,
                linespacing=1.2)
        for x, name in ((source, "Source GPU"), (destination, "Destination GPU")):
            ax.add_patch(FancyBboxPatch((x - 1.85, 0.6), 3.7, 1.2, **box))
            ax.text(x, 1.2, name, ha="center", va="center")
            ax.plot([x, x], [1.84, 3.46], color="#cfcdc6", linewidth=3.0, solid_capstyle="butt", zorder=1)
        ax.text(5.0, 2.65, "PCIe 5.0 x16 link\nfor each GPU", ha="center", va="center", fontsize=SMALL, color=MUTED,
                linespacing=1.2)
    left, right = axes
    # Links off: one copy up into host memory, then one copy down to the destination.
    left.add_patch(FancyArrowPatch((source, 1.84), (source, 6.26), color=AQUA, **arrow))
    left.add_patch(FancyArrowPatch((destination, 6.26), (destination, 1.86), color=AQUA, **arrow))
    left.text(source - 0.45, 5.55, "1", ha="center", va="center", fontweight="semibold")
    left.text(destination + 0.45, 5.55, "2", ha="center", va="center", fontweight="semibold")
    left.set_title("A  Links off: two copies", loc="left")
    left.text(0.6, 0.0, f"Measured: {staged:.1f} GiB/s", va="bottom", fontsize=10, fontweight="medium")
    # Links on: one write that goes up to the root complex and down to the destination.
    right.plot([source, source, destination], [1.84, 3.74, 3.74], color=ORANGE, linewidth=2.0, zorder=5,
               solid_joinstyle="round")
    right.add_patch(FancyArrowPatch((destination, 3.74), (destination, 1.86), color=ORANGE, **arrow))
    right.set_title("B  Links on: one write", loc="left")
    right.text(0.6, 0.0, f"Measured: {direct:.1f} GiB/s", va="bottom", fontsize=10, fontweight="medium")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.90, wspace=0.04)
    save(fig, "figure7-gpu-to-gpu-copy-paths")


def figure_sampling():
    """Figure 8. TensorFold with four streams and a 1K context: the decode speed for each group of sampler settings."""
    rows = read("tensorfold_sampling_sensitivity.csv")
    order = [("greedy", "Greedy"),
             ("temperature 1, server defaults (top_p 0.95, top_k 20)", "top_p 0.95, top_k 20 (server defaults)"),
             ("temperature 1, top_p 1.0, top_k 50", "top_p 1.0, top_k 50"),
             ("temperature 1, top_p 0.99, no top_k", "top_p 0.99, no top_k"),
             ("temperature 1, top_p 0.95, no top_k", "top_p 0.95, no top_k"),
             ("temperature 1, top_p 1.0, no top_k", "top_p 1.0, no top_k")]
    bars = [(r, label) for thinking in ("on, effort max", "off") for setting, label in order for r in rows
            if r["setting"] == setting and r["thinking"] == thinking]
    count_on = sum(r["thinking"] != "off" for r, _ in bars)
    # One row for each test, with a gap between the tests with thinking on and the tests with thinking off.
    positions = [len(bars) - i + (1 if i < count_on else 0) for i in range(len(bars))]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.7))
    for position, (r, _) in zip(positions, bars):
        value = float(r["total_tok_s"])
        ax.barh(position, value, height=0.56, color=BLUE, zorder=3)
        ax.text(value + 14, position, f"{value:,.0f}", va="center", fontsize=SMALL)
    ax.set_yticks(positions, [label for _, label in bars])
    ax.set(ylim=(min(positions) - 0.6, max(positions) + 0.8), xlim=(0, 1000), xticks=[0, 250, 500, 750, 1000],
           xlabel="Total decode, tokens/s (four streams)")
    ax.xaxis.set_major_formatter(thousands)
    rows_axis(ax)
    across = blended_transform_factory(fig.transFigure, ax.transData)
    for position, label in ((positions[0], "Thinking on"), (positions[count_on], "Thinking off")):
        ax.text(0.02, position, label, transform=across, va="center", fontweight="semibold")
    fig.text(0.02, 0.925, "Sampled rows have temperature 1", fontsize=SMALL, color=MUTED)
    fig.text(0.02, 0.025, "Thinking off: the other sampler settings have no test.", fontsize=SMALL, color=MUTED,
             style="italic")
    fig.subplots_adjust(left=0.50, right=0.95, bottom=0.17, top=0.885)
    save(fig, "figure8-tensorfold-sampling-sensitivity")


BOX = dict(boxstyle="round,pad=0.02,rounding_size=0.10", linewidth=0.7, edgecolor="#b3b1aa", facecolor="#f7f6f2")
CHIP = dict(boxstyle="round,pad=0.02,rounding_size=0.08", linewidth=0.7, edgecolor="#b3b1aa", facecolor="white")
ARROW = dict(arrowstyle="-|>", mutation_scale=13, linewidth=2.0, zorder=5, shrinkA=0, shrinkB=0)


def figure_b12x_work():
    """Figure 9. Diagram: the two types of work for one layer of the model on the four GPUs. Each of the two parts
    of a layer has work in each GPU (white) and then an all-reduce between the GPUs (gray)."""
    fig, ax = plt.subplots(figsize=(WIDTH, 4.4))
    ax.set(xlim=(0, 10), ylim=(0, 9.2))
    ax.axis("off")
    between = dict(BOX, facecolor="#e4e2db", zorder=3)
    columns = [0.25 + i * 2.42 for i in range(4)]
    for i, x in enumerate(columns):
        ax.add_patch(FancyBboxPatch((x, 1.55), 2.24, 7.4, **BOX))
        ax.text(x + 1.12, 8.55, f"GPU {i}", ha="center", va="center", color=MUTED, fontsize=SMALL)
    # The name of the part, the bottom of its row of white boxes, and the bottom of its gray bar.
    for part, (name, row, bar) in enumerate((("Attention", 7.05, 5.35), ("Feed-forward (experts)", 3.6, 1.9))):
        for i, x in enumerate(columns):
            ax.add_patch(FancyBboxPatch((x + 0.14, row), 1.96, 0.9, zorder=2, **CHIP))
            ax.text(x + 1.12, row + 0.45, name, ha="center", va="center", fontsize=SMALL, zorder=4)
            ax.add_patch(FancyArrowPatch((x + 1.12, row - 0.06), (x + 1.12, bar + 0.96), color=RULE, **ARROW))
            if part == 0:   # the sum goes down to the second part of the layer, in each GPU
                ax.add_patch(FancyArrowPatch((x + 1.12, bar - 0.06), (x + 1.12, 4.56), color=RULE, **ARROW))
        ax.text(columns[0] + 1.3, (row + bar + 0.9) / 2, "a part", va="center", color=MUTED, fontsize=SMALL, style="italic")
        ax.add_patch(FancyBboxPatch((0.25, bar), 9.5, 0.9, **between))
        ax.text(5.0, bar + 0.45, "All-reduce: add the four parts, and give the sum to each GPU", ha="center",
                va="center", fontweight="semibold", zorder=4)
    ax.text(columns[0] + 1.3, 4.93, "the sum", va="center", color=MUTED, fontsize=SMALL, style="italic")
    ax.add_patch(FancyArrowPatch((5.0, 1.5), (5.0, 0.7), color=RULE, **ARROW))
    ax.text(5.25, 1.07, "the sum goes to the subsequent layer, in each GPU", va="center", color=MUTED, fontsize=SMALL)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    save(fig, "figure9-b12x-two-types-of-work")


def figure_b12x_fusion():
    """Figure 10. Diagram: three steps as two operations with the sum in GPU memory between them, and the one
    kernel of B12X for the three steps. That kernel writes two results: the new saved value and the normalized
    result."""
    steps = ("Add the four partial results", "Add the saved value", "Normalize the result")
    dash = dict(color="#cfcdc6", linewidth=0.8, linestyle=(0, (3, 3)))
    note = dict(va="center", color=MUTED, fontsize=SMALL, style="italic")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.6))
    for ax in axes:
        ax.set(xlim=(0, 10), ylim=(0, 8.4))
        ax.axis("off")
    left, right = axes
    left.add_patch(FancyBboxPatch((0.9, 6.35), 8.2, 1.0, **BOX))
    left.text(5.0, 6.85, steps[0], ha="center", va="center")
    left.add_patch(FancyArrowPatch((5.0, 6.3), (5.0, 5.02), color=VIOLET, **ARROW))
    left.text(5.3, 5.66, "the sum, in GPU memory", **note)
    left.add_patch(FancyBboxPatch((0.9, 2.55), 8.2, 2.4, **BOX))
    left.text(5.0, 4.35, steps[1], ha="center", va="center")
    left.plot([1.6, 8.4], [3.75, 3.75], **dash)
    left.text(5.0, 3.15, steps[2], ha="center", va="center")
    left.add_patch(FancyArrowPatch((5.0, 2.5), (5.0, 1.42), color=VIOLET, **ARROW))
    left.text(5.3, 1.96, "2 results, in GPU memory", **note)
    left.set_title("A  Two operations", loc="left")
    left.text(0.9, 0.0, "2 operations, 3 results in GPU memory", va="bottom", fontsize=10, fontweight="medium")
    right.add_patch(FancyBboxPatch((0.9, 2.55), 8.2, 4.8, **BOX))
    for j, step in enumerate(steps):
        right.text(5.0, 6.55 - j * 1.6, step, ha="center", va="center")
    for y in (5.75, 4.15):
        right.plot([1.6, 8.4], [y, y], **dash)
    right.add_patch(FancyArrowPatch((5.0, 2.5), (5.0, 1.42), color=ORANGE, **ARROW))
    right.text(5.3, 1.96, "2 results, in GPU memory", **note)
    right.set_title("B  One kernel of B12X", loc="left")
    right.text(0.9, 0.0, "1 kernel, 2 results in GPU memory", va="bottom", fontsize=10, fontweight="medium")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.90, wspace=0.04)
    save(fig, "figure10-b12x-one-kernel-for-three-steps")


def figure_b12x_sizes():
    """Figure 11. Diagram: the method of the all-reduce of Jovian Judgement r24 for each message size. The limits in
    bytes are from the start log. The rows are a calculation from the source code: a row has 4,096 values of 2
    bytes, the two-shot method accepts rows in groups of four, and the DMA method has the batch limit of the
    recipe (8,192 tokens for one step)."""
    kib, mib = 1024, 1024 * 1024
    row = 4096 * 2      # bytes of one row of a message: the hidden state of one token (BF16)
    gpus, batch = 4, 8192
    limits = {r["setting"]: r["value"] for r in read("b12x_allreduce_limits.csv")}
    oneshot, twoshot, dma = (int(limits[k]) for k in ("one_shot_max_bytes", "two_shot_bf16_max_bytes", "dma_min_bytes"))
    first, last = row, batch * row
    rows1, rows2, rows3 = oneshot // row, twoshot // row, -(-dma // row)
    start2 = (rows1 // gpus + 1) * gpus
    bands = [(first, oneshot, "#fde0d2", "One-shot", f"1 to {rows1} rows"),
             (oneshot, twoshot, "#f8c1a6", "Two-shot", f"{start2} to {rows2} rows,\nin groups of {gpus}"),
             (twoshot, dma, GRID, "NCCL", f"{rows2 + 1} to {rows3 - 1} rows"),
             (dma, last, "#f2a37d", "DMA with the\ncopy engines", f"{rows3} to {batch:,} rows")]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.25))
    low_y, high_y = 0.36, 0.98
    for low, high, color, name, count in bands:
        ax.fill_betweenx([low_y, high_y], low, high, color=color, linewidth=0, zorder=2)
        ax.plot([low, low], [low_y, high_y], color="white", linewidth=2, zorder=3)
        middle = (low * high) ** 0.5
        ax.text(middle, 0.81, name, ha="center", va="center", zorder=4, fontweight="medium", linespacing=1.15)
        ax.text(middle, 0.53, count, ha="center", va="center", zorder=4, fontsize=SMALL, color=MUTED, linespacing=1.15)
    examples = [(4 * row, "Check of the draft\ntokens, 1 request", "center"),
                (64 * row, "Check of the draft\ntokens, 16 requests", "center"),
                (batch * row, "Prefill step,\n8,192 tokens", "right")]
    for size, label, align in examples:
        ax.plot([size], [high_y + 0.08], marker="v", color=INK, markersize=6.5, linestyle="none", zorder=4, clip_on=False)
        ax.text(size * (1.18 if align == "right" else 1), high_y + 0.16, label, ha=align, va="bottom", fontsize=SMALL,
                linespacing=1.2)
    ax.set_xscale("log", base=2)
    ticks = [first, oneshot, twoshot, dma, last]
    ax.set_xticks(ticks, [f"{t // kib:,} KiB" if t < mib else f"{t // mib:,} MiB" for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set(xlim=(first * 0.82, last * 1.22), ylim=(low_y, 1.62), yticks=[])
    ax.grid(visible=False)
    ax.spines["bottom"].set_position(("data", low_y))
    ax.set_xlabel("Size of one message (logarithmic scale)")
    fig.text(0.03, 0.035, "NCCL also does each message that a method of B12X does not accept (for example, 11 rows).",
             fontsize=SMALL, color=MUTED)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.31, top=0.97)
    save(fig, "figure11-b12x-method-for-each-size")


if __name__ == "__main__":
    figure_three_loads()
    figure_decode()
    figure_prefill()
    figure_long_context()
    figure_gpu_links()
    figure_quality()
    figure_copy_paths()
    figure_sampling()
    figure_b12x_work()
    figure_b12x_fusion()
    figure_b12x_sizes()

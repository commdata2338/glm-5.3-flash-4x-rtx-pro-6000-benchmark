"""Make the tables of the paper from data/*.csv, or check that the paper contains them.

  python3 scripts/make_tables.py                    print each table, with its name
  python3 scripts/make_tables.py long_context       print one table
  python3 scripts/make_tables.py --check README.md  stop with an error if a table row is not in the file

Each number in a table of README.md comes from this script. To change a number, measure again, run
scripts/build_data.py, and copy the new table into README.md. A cell with more than one run shows the mean of the
runs; data/*.csv has each run.

The body of the paper has short tables (the names that end in _body, and modified_result). The appendixes have the
full tables. The check ignores the padding of the cells, so an editor that aligns the columns does not fail it.
"""
import csv
import os
import re
import sys
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

# PAPER_DATA: a different directory with the CSV files (for a comparison with data/, as --out of build_data.py)
DATA = Path(os.environ.get("PAPER_DATA") or Path(__file__).resolve().parent.parent / "data")
TENSORFOLD, MODIFIED, VLLM = "TensorFold", "TensorFold modified", "Jovian Judgement r24"
JOVIAN28, RUN2 = "Jovian Judgement r28.1", "Jovian Judgement r24, second run"
JOVIAN28A = "Jovian Judgement r28.1, policy aligned"   # with --recurrent-checkpoint-policy aligned
# Release r38 (session 6) with the settings of release r24. Its rows and columns are in a table only when data/ has
# its values, so a table is the same as before until the test has run.
JOVIAN38, JOVIAN38A = "Jovian Judgement r38", "Jovian Judgement r38, policy aligned"
CONTROL = "Jovian Judgement r24, session 6"   # release r24 one more time, in the session of release r38
OFFICIAL, OFFICIAL_PCIE = "Official vLLM, default", "Official vLLM, tuned"
NONE = "no test"
NO_VALUE = "no value"   # the test ran, but no time interval had the decode of all its requests (status no_overlap)


def read(name):
    with open(DATA / name, newline="") as handle:
        return list(csv.DictReader(handle))


def half_up(value, digits=0):
    """Round half up. A mean of two decimal values is first made free of binary noise (0.5499999... is 0.55)."""
    exact = Decimal(repr(round(float(value), 6)))
    return exact.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)


def n0(value):
    return NONE if value is None else f"{int(half_up(value)):,}"


def n1(value):
    return NONE if value is None else f"{half_up(value, 1):,}"


def n2(value):
    return NONE if value is None else f"{half_up(value, 2):,}"


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def measured(pairs, name, key="run"):
    """The (server, run) pairs that the file has rows for."""
    there = {r[key] for r in read(name)}
    return [(server, run) for server, run in pairs if run in there]


def table(header, rows, align=None):
    align = align or ["---"] + ["---:"] * (len(header) - 1)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    return lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]


# --- the sources of each server ---------------------------------------------------------------------------------

DECODE_RUNS = measured([(TENSORFOLD, "tensorfold"), (MODIFIED, "tensorfold_modified_best"), (VLLM, "vllm_links_on_16"),
                        (JOVIAN28, "jovian_r281_16"), (JOVIAN38, "jovian_r38_16"),
                        (OFFICIAL, "official_default_16_pass2"), (OFFICIAL_PCIE, "official_pcie_16_pass2")],
                       "decode_sparkdash.csv")


def wave_cells(top_p, top_k, runs=(), arm=None, probe="long_context_sampled", field="total_tok_s"):
    """concurrency -> the mean of the runs, from concurrent_waves.csv (runs) or modified_arms.csv (arm)."""
    cells = defaultdict(list)
    rows = [r for r in read("concurrent_waves.csv") if r["run"] in runs] if runs else \
        [r for r in read("modified_arms.csv") if r["arm"] == arm]
    for r in rows:
        if r["probe"] == probe and float(r["top_p"]) == top_p and r["top_k"] == top_k and r[field] != "":
            cells[int(r["concurrency"])].append(float(r[field]))
    return {c: mean(v) for c, v in cells.items()}


# server: {sampler setting: keyword arguments for wave_cells}
LONG = {
    TENSORFOLD: {"k20": dict(runs=("tensorfold_session3",)), "off": dict(runs=("tensorfold", "tensorfold_session3")),
                 "p1": dict(runs=("tensorfold",))},
    MODIFIED: {"k20": dict(arm="modified-best"), "off": dict(arm="modified-best"), "p1": dict(arm="modified-best")},
    VLLM: {"k20": dict(runs=("vllm_links_on_16_session3",)),
           "off": dict(runs=("vllm_links_on_16", "vllm_links_on_16_session3")), "p1": dict(runs=("vllm_links_on_16",))},
    # Jovian Judgement r28.1 (session 5): the requests of each wave ran one after another, so there is no speed.
    JOVIAN28: {"k20": dict(runs=("jovian_r281_16_topk",)), "off": dict(runs=("jovian_r281_16_topk",))},
    JOVIAN28A: {"k20": dict(runs=("jovian_r281_16_aligned_topk",)),
                "off": dict(runs=("jovian_r281_16_aligned", "jovian_r281_16_aligned_topk")),
                "p1": dict(runs=("jovian_r281_16_aligned",))},
    # Jovian Judgement r38 (session 6): the suite has the two passes with no top_k, the three passes have top_k 20.
    JOVIAN38: {"k20": dict(runs=("jovian_r38_16_topk",)), "off": dict(runs=("jovian_r38_16", "jovian_r38_16_topk")),
               "p1": dict(runs=("jovian_r38_16",))},
    JOVIAN38A: {"k20": dict(runs=("jovian_r38_16_aligned_topk",)),
                "off": dict(runs=("jovian_r38_16_aligned", "jovian_r38_16_aligned_topk")),
                "p1": dict(runs=("jovian_r38_16_aligned",))},
    # The official vLLM: top_k 20 is from session 5 (two passes). No top_k is from session 2 and session 5.
    OFFICIAL: {"k20": dict(runs=("official_default_16_session5",)),
               "off": dict(runs=("official_default_16", "official_default_16_session5")),
               "p1": dict(runs=("official_default_16",))},
    OFFICIAL_PCIE: {"k20": dict(runs=("official_pcie_16_session5",)),
                    "off": dict(runs=("official_pcie_16", "official_pcie_16_session5")),
                    "p1": dict(runs=("official_pcie_16",))},
}
SAMPLER = {"k20": (0.95, "20"), "off": (0.95, "off"), "p1": (1.0, "off")}
# Release r38 is in the long-context tables when data/ has one of its waves.
_WAVE_RUNS = {r["run"] for r in read("concurrent_waves.csv") if r["probe"] == "long_context_sampled"}
LONG = {server: sources for server, sources in LONG.items() if server not in (JOVIAN38, JOVIAN38A)
        or any(run in _WAVE_RUNS for source in sources.values() for run in source["runs"])}


def long_cell(server, setting, concurrency, field="total_tok_s"):
    if setting not in LONG.get(server, {}):
        return None
    return wave_cells(*SAMPLER[setting], field=field, **LONG[server][setting]).get(concurrency)


def long_text(server, setting, concurrency):
    """The cell of the long-context tables: the speed, "no test", or "no value" for a test that gave no speed."""
    value = long_cell(server, setting, concurrency)
    if (value is None and setting in LONG.get(server, {})
            and long_cell(server, setting, concurrency, field="median_ttft_s")):
        return NO_VALUE
    return n0(value)


def reuse_cells(run, size):
    """case -> the seconds to the first token of each round."""
    cells = defaultdict(list)
    for r in read("chat_reuse_ttft.csv"):
        if r["run"] == run and r["nominal_size"] == size:
            cells[r["case"]].append(float(r["ttft_s"]))
    return cells


def seconds(value):
    return n1(value) if value >= 5 else n2(value)


KINDS = (("prose", "Prose"), ("code", "Code"), ("structured", "Count task"), ("json", "JSON"))
REUSE_RUNS = measured([(TENSORFOLD, "tensorfold"), (MODIFIED, "tensorfold_modified_best"), (VLLM, "vllm_links_on_16"),
                       (JOVIAN28, "jovian_r281_16"), (JOVIAN28A, "jovian_r281_16_aligned"),
                       (JOVIAN38, "jovian_r38_16"), (JOVIAN38A, "jovian_r38_16_aligned"),
                       (OFFICIAL, "official_default_16"), (OFFICIAL_PCIE, "official_pcie_16")], "chat_reuse_ttft.csv")
REUSE_CASES = ("cold", "identical resend", "resend + new turn", "own reply + new turn")
REUSE_HEADER = ["Cold", "Same prompt again", "Same prompt and a new turn", "Answer of the model and a new turn"]


# --- the tables ----------------------------------------------------------------------------------------------------

def decode():
    rows, data, published = [], read("decode_sparkdash.csv"), read("tensorfold_published_decode.csv")
    for kind, name in KINDS:
        for i, (server, run) in enumerate(DECODE_RUNS):
            cells = {int(r["concurrency"]): float(r["total_tok_s"]) for r in data
                     if r["run"] == run and r["output_type"] == kind}
            rows.append([name if i == 0 else "", server, *[n0(cells.get(c)) for c in (1, 2, 4, 8, 16)]])
            reference = {int(r["concurrency"]): float(r["total_tok_s"]) for r in published if r["output_type"] == kind}
            if run == "tensorfold" and reference:
                rows.append(["", "TensorFold, published values", *[n0(reference[c]) for c in (1, 2, 4, 8, 16)]])
    return table(["Output", "Inference stack", "1", "2", "4", "8", "16"], rows, ["---", "---", "---:", "---:", "---:", "---:", "---:"])


def greedy_1k():
    rows = []
    sources = [(TENSORFOLD, dict(runs=("tensorfold",))), (MODIFIED, dict(arm="modified-best")),
               (VLLM, dict(runs=("vllm_links_on_16",))), (JOVIAN28, dict(runs=("jovian_r281_16",))),
               (JOVIAN38, dict(runs=("jovian_r38_16",))),
               (OFFICIAL, dict(runs=("official_default_16",))), (OFFICIAL_PCIE, dict(runs=("official_pcie_16",)))]
    for server, source in sources:
        cells = wave_cells(1.0, "off", probe="greedy_1k", **source)
        if cells:   # a server with no test of this type has no row
            rows.append([server, *[n0(cells.get(c)) for c in (1, 8, 12, 16)]])
    return table(["Inference stack", "1", "8", "12", "16"], rows)


def prefill():
    data = read("prefill_sparkdash.csv")
    published = sorted((int(r["prompt_tokens"]), float(r["prefill_tok_s"])) for r in read("tensorfold_published_prefill.csv"))
    rows = []
    for server, run in ((TENSORFOLD, "tensorfold"), ("TensorFold, published values", None), (MODIFIED, "tensorfold_modified_best"),
                        ("Jovian Judgement r24, direct GPU links on", "vllm_links_on_16"),
                        ("Jovian Judgement r24, direct GPU links off (8 slots)", "vllm_links_off_8"),
                        (JOVIAN28, "jovian_r281_16"), (JOVIAN38, "jovian_r38_16"),
                        (OFFICIAL, "official_default_16_pass2"), (OFFICIAL_PCIE, "official_pcie_16_pass2")):
        cells = [v for _, v in published] if run is None else \
            [v for _, v in sorted((int(r["target_tokens"]), float(r["prefill_tok_s"])) for r in data if r["run"] == run)]
        if cells:   # a server with no test of this type has no row
            rows.append([server, *[n0(v) for v in cells]])
    return table(["Inference stack", "8K", "16K", "32K", "64K", "128K", "256K"], rows)


def decode_body():
    """The decode test at 1 and at 16 concurrent requests: one row for each server."""
    data = read("decode_sparkdash.csv")
    rows = []
    for server, run in DECODE_RUNS:
        cells = {(r["output_type"], int(r["concurrency"])): float(r["total_tok_s"]) for r in data if r["run"] == run}
        rows.append([server, *[n0(cells.get((kind, c))) for kind, _ in KINDS for c in (1, 16)]])
    return table(["Inference stack", *[f"{name}, {c}" for _, name in KINDS for c in (1, 16)]], rows)


def prefill_body():
    """The prefill test for three prompt sizes: one row for each server."""
    data = read("prefill_sparkdash.csv")
    rows = []
    for server, run in DECODE_RUNS:
        cells = {r["target_tokens"]: float(r["prefill_tok_s"]) for r in data if r["run"] == run}
        rows.append([server, *[n0(cells.get(size)) for size in ("8192", "65536", "262144")]])
    return table(["Inference stack", "8K", "64K", "256K"], rows)


def long_context():
    rows = []
    for concurrency in (4, 8, 16):
        for i, server in enumerate(LONG_TABLE):
            rows.append([concurrency if i == 0 else "", server,
                         *[long_text(server, setting, concurrency) for setting in ("k20", "off", "p1")],
                         n1(long_cell(server, "off", concurrency, field="median_ttft_s"))])
    return table(["Requests", "Inference stack", "top_p 0.95, top_k 20", "top_p 0.95, no top_k", "top_p 1.0, no top_k",
                  "Time to first token, s"], rows, ["---:", "---", "---:", "---:", "---:", "---:"])


def round_times():
    data = [r for r in read("tensorfold_round_times.csv") if r["all_requests_found_the_context"] == "True"]

    def cell(run, top_p, top_k, concurrency):
        return mean(float(r["median_round_ms"]) for r in data if r["run"] == run and float(r["top_p"]) == top_p
                    and r["top_k"] == top_k and int(r["concurrency"]) == concurrency)
    rows = [[c, n0(cell("tensorfold_session3", 0.95, "20", c)), n0(cell("tensorfold_session3", 0.95, "off", c)),
             n0(cell("tensorfold_modified_best", 0.95, "20", c)), n0(cell("tensorfold_modified_best", 0.95, "off", c)),
             n0(cell("tensorfold_modified_best", 1.0, "off", c))] for c in (4, 8, 16)]
    return table(["Requests", "TensorFold, top_k 20", "TensorFold, no top_k", "TensorFold modified, top_k 20",
                  "TensorFold modified, no top_k", "TensorFold modified, top_p 1.0"], rows, ["---:"] * 6)


def sampler_sensitivity():
    data = read("tensorfold_sampling_sensitivity.csv")
    order = ["greedy", "temperature 1, server defaults (top_p 0.95, top_k 20)", "temperature 1, top_p 1.0, top_k 50",
             "temperature 1, top_p 0.99, no top_k", "temperature 1, top_p 0.95, no top_k",
             "temperature 1, top_p 1.0, no top_k"]
    rows = []
    for thinking, label in (("on, effort max", "on"), ("off", "off")):
        for setting in order:
            for r in data:
                if r["setting"] == setting and r["thinking"] == thinking:
                    rows.append([setting[0].upper() + setting[1:], label, n0(r["total_tok_s"]), n0(r["median_stream_tok_s"])])
    return table(["Sampler settings", "Thinking", "Total tokens/s", "Tokens/s for each stream"], rows,
                 ["---", "---", "---:", "---:"])


def reuse():
    rows = []
    for size, name in (("9000", "13.5K"), ("56000", "83.6K")):
        for i, (server, run) in enumerate(REUSE_RUNS):
            cells = reuse_cells(run, size)
            rows.append([name if i == 0 else "", server,
                         *[", ".join(seconds(v) for v in cells[case]) for case in REUSE_CASES]])
    return table(["Prompt", "Inference stack", *REUSE_HEADER], rows, ["---", "---", "---:", "---:", "---:", "---:"])


def reuse_body():
    """The four cache cases for the prompt of 83.6K tokens: the two runs of each case."""
    rows = []
    for server, run in REUSE_RUNS:
        cells = reuse_cells(run, "56000")
        rows.append([server, *[", ".join(seconds(v) for v in cells[case]) for case in REUSE_CASES]])
    return table(["Inference stack", *REUSE_HEADER], rows)


def burst():
    data = read("cold_burst.csv")
    release = [r for r in data if r["server"].startswith("TensorFold, recipe")]
    modified = [r for r in data if "burst reuse on" in r["server"]]
    rows = [[i, n1(a["ttft_s"]), n0(a["cached_tokens_reported"]), n1(b["ttft_s"]), n0(b["cached_tokens_reported"])]
            for i, (a, b) in enumerate(zip(release, modified), 1)]
    return table(["Request", "TensorFold: time to first token, s", "TensorFold: cached tokens",
                  "TensorFold modified: time to first token, s", "TensorFold modified: cached tokens"], rows, ["---:"] * 5)


def gpu_links():
    prefill_rows, off, on = read("prefill_sparkdash.csv"), "vllm_links_off_8", "vllm_links_on_8"

    def span(run):
        values = [float(r["prefill_tok_s"]) for r in prefill_rows if r["run"] == run]
        return min(values), max(values)

    def ratio(a, b):
        return f"{'+' if b >= a else '−'}{n0(abs(100 * (b / a - 1)))}%"
    ratios = sorted(100 * (float(b["prefill_tok_s"]) / float(a["prefill_tok_s"]) - 1)
                    for a in prefill_rows if a["run"] == off
                    for b in prefill_rows if b["run"] == on and b["target_tokens"] == a["target_tokens"])
    rows = [["Prefill of a cold prompt, 8K to 256K, tokens/s", f"{n0(span(off)[0])} to {n0(span(off)[1])}",
             f"{n0(span(on)[0])} to {n0(span(on)[1])}", f"+{n0(ratios[0])}% to +{n0(ratios[-1])}%"]]
    for label, case in (("Time to first token, cold chat of 83.6K tokens, s", "cold"),
                        ("Time to first token, new turn on a chat of 83.6K tokens, s", "resend + new turn")):
        a, b = mean(reuse_cells(off, "56000")[case]), mean(reuse_cells(on, "56000")[case])
        rows.append([label, seconds(a), seconds(b), ratio(a, b)])
    for concurrency in (8, 4):
        a = wave_cells(1.0, "off", runs=(off,))[concurrency]
        b = wave_cells(1.0, "off", runs=(on,))[concurrency]
        rows.append([f"Decode, 56K context, thinking on, top_p 1.0, no top_k, {concurrency} requests, tokens/s", n0(a), n0(b),
                     ratio(a, b)])
    return table(["", "Links off", "Links on", "Change"], rows)


ALL_QUALITY_RUNS = [(TENSORFOLD, "tensorfold"), (MODIFIED, "tensorfold_modified_best"), (VLLM, "vllm_links_on_8"),
                    (RUN2, "vllm_links_on_8_run2"), (JOVIAN28, "jovian_r281_8"), (JOVIAN38, "jovian_r38_8"),
                    (OFFICIAL, "official_default_16")]
QUALITY_RUNS = [(server, run) for server, run in ALL_QUALITY_RUNS
                if any(r["run"] == run for r in read("quality_pi_summary.csv"))]
QUALITY_PAIRS = (("tensorfold", "tensorfold_modified_best", "TensorFold and TensorFold modified"),
                 ("tensorfold", "vllm_links_on_8", "TensorFold and Jovian Judgement r24"),
                 ("tensorfold_modified_best", "vllm_links_on_8", "TensorFold modified and Jovian Judgement r24"),
                 ("vllm_links_on_8", "vllm_links_on_8_run2", "Jovian Judgement r24, the first run and the second run"),
                 ("vllm_links_on_8", "jovian_r281_8", "Jovian Judgement r24 and Jovian Judgement r28.1"),
                 ("vllm_links_on_8_run2", "jovian_r281_8",
                  "Jovian Judgement r24, the second run, and Jovian Judgement r28.1"),
                 ("vllm_links_on_8", "jovian_r38_8", "Jovian Judgement r24 and Jovian Judgement r38"),
                 ("vllm_links_on_8_run2", "jovian_r38_8",
                  "Jovian Judgement r24, the second run, and Jovian Judgement r38"),
                 ("jovian_r281_8", "jovian_r38_8", "Jovian Judgement r28.1 and Jovian Judgement r38"),
                 ("vllm_links_on_8", "official_default_16", "Jovian Judgement r24 and the official vLLM, default"),
                 ("tensorfold", "official_default_16", "TensorFold and the official vLLM, default"))


def quality_pass():
    data = {(r["run"], r["data_set"]): r for r in read("quality_pi_summary.csv")}
    rows = []
    for key, name in (("HumanEval+", "HumanEval+, 164 tasks"), ("MBPP+", "MBPP+, 378 tasks"),
                      ("both", "The two data sets, 542 tasks")):
        for i, (tests, label) in enumerate((("base_tests", "Base tests"), ("all_tests", "All tests"))):
            rows.append([name if i == 0 else "", label,
                         *[f"{data[(run, key)][tests + '_pass']} ({data[(run, key)][tests + '_percent']}%)"
                           for _, run in QUALITY_RUNS]])
    return table(["Data set", "Tests", *[s for s, _ in QUALITY_RUNS]], rows, ["---", "---"] + ["---:"] * len(QUALITY_RUNS))


def quality_pass_body():
    """The tasks that pass all tests."""
    data = {(r["run"], r["data_set"]): r for r in read("quality_pi_summary.csv")}
    rows = [[name, *[f"{data[(run, key)]['all_tests_pass']} ({data[(run, key)]['all_tests_percent']}%)"
                     for _, run in QUALITY_RUNS]]
            for key, name in (("HumanEval+", "HumanEval+, 164 tasks"), ("MBPP+", "MBPP+, 378 tasks"),
                              ("both", "The two data sets, 542 tasks"))]
    return table(["Tasks that pass all tests", *[s for s, _ in QUALITY_RUNS]], rows)


def quality_paired(short=False):
    data = {(r["run_a"], r["run_b"]): r for r in read("quality_pi_paired.csv")
            if r["data_set"] == "both" and r["tests"] == "all tests"}
    rows = []
    for a, b, label in QUALITY_PAIRS:
        if (a, b) not in data:
            continue
        r = data[(a, b)]
        difference = f"{float(r['b_minus_a_percentage_points']):+.1f} ({float(r['difference_ci95_low']):+.1f} to " \
                     f"{float(r['difference_ci95_high']):+.1f})"
        rows.append([label, r["both_pass"], r["only_a_passes"], r["only_b_passes"], r["neither_passes"],
                     difference.replace("-", "−"), n2(r["exact_mcnemar_p"])])
    if short:
        return table(["Pair", "Only the first passes", "Only the second passes",
                      "Second minus first, percentage points (95% interval)", "Exact McNemar test, p"],
                     [[r[0], r[2], r[3], r[5], r[6]] for r in rows if r[0] in BODY_PAIRS])
    return table(["Pair", "The two inference stacks pass", "Only the first passes", "Only the second passes", "No inference stack passes",
                  "Second minus first, percentage points (95% interval)", "Exact McNemar test, p"], rows)


BODY_PAIRS = ("TensorFold and TensorFold modified", "TensorFold and Jovian Judgement r24",
              "Jovian Judgement r24, the first run and the second run", "Jovian Judgement r24 and Jovian Judgement r28.1",
              "Jovian Judgement r24 and Jovian Judgement r38", "Jovian Judgement r24 and the official vLLM, default")


def quality_paired_body():
    return quality_paired(short=True)


def quality_work(short=False):
    tasks, calls = read("quality_pi_tasks.csv"), {r["run"]: r for r in read("quality_pi_calls.csv")}
    per = {run: [r for r in tasks if r["run"] == run] for _, run in QUALITY_RUNS}

    def median(values):
        values = sorted(values)
        return (values[(len(values) - 1) // 2] + values[len(values) // 2]) / 2

    def out(run):
        return sum(int(r["output_tokens"] or 0) for r in per[run])
    lines = [
        ("Model calls", lambda run: n0(calls[run]["model_calls"])),
        ("Output tokens, thoughts included", lambda run: f"{n0(out(run) / 1000)}K"),
        ("Tasks with more than 10,000 output tokens", lambda run: sum(int(r["output_tokens"] or 0) > 10000 for r in per[run])),
        ("Median time for a model call, s", lambda run: n2(calls[run]["median_call_s"])),
        ("Sum of the times of the model calls, s", lambda run: n0(calls[run]["sum_of_call_s"])),
        ("Output tokens ÷ sum of the times of the model calls, tokens/s",
         lambda run: n0(out(run) / float(calls[run]["sum_of_call_s"]))),
        ("Median time for a task, s", lambda run: n1(median(float(r["wall_seconds"]) for r in per[run]))),
        ("Tasks that the time limit stopped", lambda run: sum(r["time_limit_hit"] == "True" for r in per[run])),
        ("Tasks with no solution file", lambda run: sum(r["solution_file_present"] != "True" for r in per[run])),
    ]
    if short:
        lines = [line for line in lines if line[0] in WORK_BODY]
    return table(["", *[s for s, _ in QUALITY_RUNS]], [[label, *[f(run) for _, run in QUALITY_RUNS]] for label, f in lines])


WORK_BODY = ("Model calls", "Output tokens, thoughts included", "Median time for a model call, s")


def quality_work_body():
    return quality_work(short=True)


MODIFIED_SETTINGS = [  # (folder of the run, settings on top of the sampler settings, purpose)
    ("modified-sampler", "The sampler settings only", "The reference for this table"),
    ("modified-s+block4", "`TF_GLM_DRAFT_FAST_BLOCK=4`", "Fewer draft rows in each round. The default at 4 to 16 streams is 8 rows."),
    ("modified-s+depth", "`TF_GLM_MULTI_DEPTH=scale:0.25`", "A higher confidence limit for draft tokens when more streams decode together"),
    ("modified-s+l2pf", "`TF_GLM_L2PF_ROWS=128` and `TF_GLM_L2PF_MB=16`", "Weight prefetch for rounds with more than 64 rows"),
    ("modified-s+graphstep", "`TF_GLM_MULTI_GRAPH_STEP=1`", "A CUDA graph for each batch width above 64 rows"),
    ("modified-s+ipc", "`TF_GLM_COMM=ipc`", "CUDA IPC for the all-gathers between the ranks"),
    ("modified-s+prof", "`TF_GLM_SEGPROF=4`", "The profiler of Appendix E.2. It adds work to one of four rounds."),
]


def modified_settings():
    rows = []
    for arm, label, purpose in MODIFIED_SETTINGS:
        cells = wave_cells(0.95, "20", arm=arm)
        rows.append([label, purpose, *[n0(cells[c]) for c in (4, 8, 16)]])
    return table(["Settings", "Purpose", "4 requests", "8 requests", "16 requests"], rows,
                 ["---", "---", "---:", "---:", "---:"])


def modified_prefill_chunks():
    data = read("modified_prefill_direct.csv")
    rows = [[label, *[n0(r["prefill_tok_s"]) for r in data if r["arm"] == arm]]
            for arm, label in (("modified-s+burst", "Chunks of 4,096 rows (the default)"),
                               ("modified-s+prefill8192", "`TF_GLM_PREFILL_ROWS=8192` and `TF_GLM_CE_ARENA_MIB=384`"))]
    return table(["Prefill, tokens/s", "8K", "16K", "32K", "64K", "128K", "256K"], rows)


PROFILE_ROWS = [("56K, top_k 20", "15.2", None), ("56K, top_k 20", "29.1", None), ("56K, top_k 20", "56.8", None),
                ("1K, greedy", "29.8", "8 and 12"), ("1K, greedy", "39.9", "12 and 16")]
# (context, rows a round of the report line, the two waves of a report line that has rounds of two waves)


def round_profile():
    data = {r["rows_a_round"]: r for r in read("modified_round_profile.csv")}
    rows = []
    for context, key, two_waves in PROFILE_ROWS:
        r = data[key]
        assert (r["rounds_from_more_than_one_wave"] == "True") == (two_waves is not None), key
        requests = two_waves or r["requests_in_the_wave"]
        other = float(r["head_ms"]) + float(r["glue_ms"])
        rows.append([context, requests, key, n1(r["verify_gpu_ms"]), n1(r["experts_ms"]), n1(r["dsa_ms"]),
                     n1(r["exchange_ms"]), n1(r["shared_ms"]), n1(r["dense_ms"]), n1(r["kda_ms"]),
                     n1(other).replace("-", "−")])
    return table(["Context", "Requests", "Rows in a round", "GPU time of a round, ms", "Routed experts",
                  "Sparse attention", "Transfers between ranks", "Shared expert", "Dense layers", "Linear attention",
                  "Head and remainder"], rows)


def cold_burst():
    """The seconds to the first token of the four cold requests: (TensorFold, TensorFold modified with burst reuse)."""
    data = read("cold_burst.csv")
    return ([float(r["ttft_s"]) for r in data if r["server"].startswith("TensorFold, recipe")],
            [float(r["ttft_s"]) for r in data if "burst reuse on" in r["server"]])


BURST_VLLM = ("Jovian Judgement r24, 8 slots", "Jovian Judgement r28.1, 8 slots", "Official vLLM, default",
              "Official vLLM, tuned")


def burst_times(server):
    return [float(r["ttft_s"]) for r in read("cold_burst.csv") if r["server"] == server]


def span(values):
    """The lowest and the highest value of a group of times, in one cell."""
    return NONE if not values else f"{n1(min(values))} to {n1(max(values))}"


BURST_R38 = "Jovian Judgement r38, 8 slots"
BURST_VLLM_ALL = ("Jovian Judgement r24, 8 slots", "Jovian Judgement r28.1, 8 slots", "Jovian Judgement r28.1, 16 slots",
                  "Jovian Judgement r28.1, 16 slots, checkpoint policy aligned", BURST_R38,
                  "Jovian Judgement r38, 16 slots", "Jovian Judgement r38, 16 slots, checkpoint policy aligned",
                  "Official vLLM, default", "Official vLLM, tuned")
BURST_BODY = ((TENSORFOLD, "TensorFold, recipe 1.0.1"), (MODIFIED, "TensorFold modified, GPU sampler and burst reuse on"),
              (VLLM, "Jovian Judgement r24, 8 slots"), (JOVIAN28, "Jovian Judgement r28.1, 8 slots"),
              (JOVIAN28A, "Jovian Judgement r28.1, 16 slots, checkpoint policy aligned"),
              (JOVIAN38, BURST_R38), (JOVIAN38A, "Jovian Judgement r38, 16 slots, checkpoint policy aligned"),
              (OFFICIAL, "Official vLLM, default"), (OFFICIAL_PCIE, "Official vLLM, tuned"))


def burst_body():
    """The four cold requests on each server: the seconds to the first token of each request, lowest first."""
    rows = [[server, ", ".join(str(n1(v)) for v in sorted(burst_times(label)))]
            for server, label in BURST_BODY if burst_times(label)]
    return table(["Inference stack", "Time to the first token of each of the four requests, s"], rows, ["---", "---:"])


def burst_vllm():
    """The four cold requests on the vLLM servers: the seconds to the first token, lowest first."""
    servers = [s for s in BURST_VLLM_ALL if burst_times(s)]
    rows = [[i + 1, *[n1(sorted(burst_times(s))[i]) for s in servers]] for i in range(4)]
    return table(["Request, in the sequence of the first tokens", *servers], rows, ["---:"] * (len(servers) + 1))


REUSE_CHECK = (  # (column, server, checkpoint setting) of prefix_reuse_check.csv
    ("Jovian Judgement r24", "Jovian Judgement r24", "release default"),
    ("Jovian Judgement r28.1", "Jovian Judgement r28.1", "release default"),
    ("Jovian Judgement r28.1, dense retention", "Jovian Judgement r28.1", "--prefix-cache-retention-interval None"),
    ("Jovian Judgement r28.1, policy aligned", "Jovian Judgement r28.1", "--recurrent-checkpoint-policy aligned"),
    ("Jovian Judgement r38", "Jovian Judgement r38", "release default"),
    ("Jovian Judgement r38, policy aligned", "Jovian Judgement r38", "--recurrent-checkpoint-policy aligned"),
    ("Official vLLM, default", "Official vLLM 0.31.0", "release default"),
    ("Official vLLM, tuned", "Official vLLM 0.31.0", "release default"),
)
REUSE_STEPS = {
    "user message": (("first request, 16 tokens", "1. A context of 56K tokens and a question"),
                     ("the same request again", "2. The same request again"),
                     ("same context, other question", "3. The same context and a different question"),
                     ("the same request, 300 tokens", "4. The request of step 1, with 300 output tokens"),
                     ("the same request again, after 300 tokens", "5. The request of step 1 again"),
                     ("same context, third question", "6. The same context and a third question")),
    "system message": (("first request, 16 tokens", "1. A system message of 56K tokens and a question"),
                       ("the same request again", "2. The same request again"),
                       ("same context, other question", "3. The same system message and a different question"),
                       ("same context, third question", "4. The same system message and a third question")),
}


REUSE_CHECK_BODY = ("Jovian Judgement r24", "Jovian Judgement r28.1", "Jovian Judgement r28.1, policy aligned",
                    "Jovian Judgement r38", "Jovian Judgement r38, policy aligned", "Official vLLM, default")


def reuse_check(place="user message", only=None):
    """The prompt-reuse check: the seconds to the full reply of each step, for each server that has the check."""
    data = [r for r in read("prefix_reuse_check.csv") if r["shared_context_in"] == place]
    columns = []
    for name, server, setting in REUSE_CHECK:
        if only and name not in only:
            continue
        rows = [r for r in data if r["server"] == server and r["checkpoint_setting"] == setting
                and ("tuned" in name) == ("PCIe" in r["configuration"])]
        if rows:   # the column name gives the slot count: the check ran on servers with 8 slots and with 16 slots
            slots = re.match(r"(\d+) slots", rows[0]["configuration"]).group(1)
            columns.append((f"{name}, {slots} slots", {r["step"]: float(r["seconds_to_the_full_reply"]) for r in rows}))
    rows = [[label, *[n2(cells[step]) for _, cells in columns]] for step, label in REUSE_STEPS[place]]
    return table(["Step", *[name for name, _ in columns]], rows)


def reuse_check_system():
    return reuse_check("system message")


def reuse_check_body():
    """The prompt-reuse check with the context in a user message, for four servers."""
    return reuse_check(only=REUSE_CHECK_BODY)


SYSTEM_FORM = ((VLLM + ", 8 slots", "vllm_links_on_8_session5"), (JOVIAN28 + ", 8 slots", "jovian_r281_8"),
               (JOVIAN28A + ", 16 slots", "jovian_r281_16_aligned"), (JOVIAN38 + ", 8 slots", "jovian_r38_8"),
               (JOVIAN38A + ", 16 slots", "jovian_r38_16_aligned"))


def long_context_system():
    """The long-context test with the shared context as a system message."""
    data = [r for r in read("concurrent_waves.csv") if r["probe"] == "long_context_sampled_system_message"]

    def cell(run, concurrency, top_p, top_k, field, number=None):
        return mean(float(r[field]) for r in data if r["run"] == run and int(r["concurrency"]) == concurrency
                    and float(r["top_p"]) == top_p and r["top_k"] == top_k and r[field] != ""
                    and (number is None or int(r["pass"]) == number))
    rows = []
    for concurrency in (4, 8, 16):
        block = []
        for server, run in SYSTEM_FORM:
            speeds = [cell(run, concurrency, *SAMPLER[setting], "total_tok_s") for setting in ("k20", "off", "p1")]
            if any(v is not None for v in speeds):
                block.append(["", server, *[n0(v) for v in speeds],
                              n1(cell(run, concurrency, 0.95, "20", "median_ttft_s", number=1))])
        if block:
            block[0][0] = concurrency
            rows += block
    return table(["Requests", "Inference stack", "top_p 0.95, top_k 20", "top_p 0.95, no top_k", "top_p 1.0, no top_k",
                  "Time to first token, s"], rows, ["---:", "---", "---:", "---:", "---:", "---:"])


POLICY_RUNS = ((VLLM, "vllm_links_on_16"), (JOVIAN28, "jovian_r281_16"), (JOVIAN28A, "jovian_r281_16_aligned"))
POLICY_BURST = ("Jovian Judgement r24, 8 slots", "Jovian Judgement r28.1, 16 slots",
                "Jovian Judgement r28.1, 16 slots, checkpoint policy aligned")


# Release r38 beside release r24 of session 1 and beside release r24 of the same session (the control).
# (column, run of the suite, server of the long-context tables, server of the cold requests)
R38_RUNS = ((VLLM, "vllm_links_on_16", VLLM, "Jovian Judgement r24, 8 slots"),
            (CONTROL, "vllm_links_on_16_session6", CONTROL, "Jovian Judgement r24, 16 slots, session 6"),
            (JOVIAN38, "jovian_r38_16", JOVIAN38, "Jovian Judgement r38, 16 slots"),
            (JOVIAN38A, "jovian_r38_16_aligned", JOVIAN38A, "Jovian Judgement r38, 16 slots, checkpoint policy aligned"))
LONG[CONTROL] = {"k20": dict(runs=("vllm_links_on_16_session6_topk",)),
                 "off": dict(runs=("vllm_links_on_16_session6", "vllm_links_on_16_session6_topk")),
                 "p1": dict(runs=("vllm_links_on_16_session6",))}
LONG_TABLE = [server for server in LONG if server != CONTROL]   # the control is in the table of release r38 only


def r281_policies(columns=None):
    """Release r24 and release r28.1 with its two checkpoint policies, with 16 slots."""
    decode_rows, prefill_rows = read("decode_sparkdash.csv"), read("prefill_sparkdash.csv")
    columns = columns or [(server, run, server, burst) for (server, run), burst in zip(POLICY_RUNS, POLICY_BURST)]
    servers, runs = [c[0] for c in columns], [c[1] for c in columns]
    long_servers, bursts = [c[2] for c in columns], [c[3] for c in columns]

    def decode(run, kind, concurrency):
        return next((float(r["total_tok_s"]) for r in decode_rows if r["run"] == run and r["output_type"] == kind
                     and int(r["concurrency"]) == concurrency), None)

    def fill(run, size):
        return next((float(r["prefill_tok_s"]) for r in prefill_rows
                     if r["run"] == run and r["target_tokens"] == size), None)

    def first(run, case):
        return mean(reuse_cells(run, "56000")[case])

    def seconds_cell(value):
        return NONE if value is None else seconds(value)
    rows = [[f"Short greedy answers ({name.lower()}), {c} request{'s' if c > 1 else ''}, tokens/s",
             *[n0(decode(run, kind, c)) for run in runs]]
            for kind, name in (("prose", "Prose"), ("code", "Code")) for c in (1, 16)]
    rows += [
        ["Prefill of a cold prompt of 64K tokens, tokens/s", *[n0(fill(run, "65536")) for run in runs]],
        ["Prefill of a cold prompt of 256K tokens, tokens/s", *[n0(fill(run, "262144")) for run in runs]],
        ["56K context in a user message, top_p 0.95, top_k 20, 16 requests, tokens/s",
         *[long_text(s, "k20", 16) for s in long_servers]],
        ["The same test with no top_k: time to the first token, s",
         *[n1(long_cell(s, "off", 16, field="median_ttft_s")) for s in long_servers]],
        ["First token for the same prompt of 83.6K tokens a second time, s",
         *[seconds_cell(first(run, "identical resend")) for run in runs]],
        ["First token for the same prompt and a new turn, s",
         *[seconds_cell(first(run, "resend + new turn")) for run in runs]],
        ["First token for the answer of the model and a new turn, s",
         *[seconds_cell(first(run, "own reply + new turn")) for run in runs]],
        ["First token for four cold requests with a shared prefix of 42K tokens, s",
         *[span(burst_times(server) if server else []) for server in bursts]],
    ]
    return table(["Test", *servers], rows)


def short_context():
    """The short-context probe: the mean probability of the correct subsequent token for each group of lengths."""
    data = read("short_context_probe.csv")
    servers = [s for s in (VLLM, JOVIAN38) if any(r["server"] == s for r in data)]
    cell = {(r["server"], r["first_length"], r["lengths_are_a_multiple_of_4"]):
            f"{float(r['mean_probability_of_the_correct_token']):.3f}" for r in data}
    groups = sorted({(int(r["first_length"]), int(r["last_length"])) for r in data})
    rows = [[f"{first:,} to {last:,}", *[cell[(s, str(first), whole)] for s in servers for whole in ("True", "False")]]
            for first, last in groups]
    header = [f"{s.replace('Jovian Judgement', 'Release')}: {kind}" for s in servers
              for kind in ("a multiple of 4", "the other lengths")]
    return table(["Prompt lengths, tokens", *header], rows)


def r38_releases():
    """Release r38 with its two checkpoint policies, release r24, and release r24 in the session of release r38."""
    there = {r["run"] for r in read("decode_sparkdash.csv")}
    return r281_policies([column for column in R38_RUNS if column[1] in there])


HIDDEN_STATE_BYTES = 4096 * 2   # one row of a message: the hidden state of one token, 4,096 values of 2 bytes (BF16)
TOKENS_FOR_A_REQUEST = 4        # the check of the draft tokens has 4 rows for each request: 1 token and 3 draft tokens
GPUS = 4                        # the two-shot method accepts a number of rows that is a multiple of the GPU count
BATCH_LIMIT_TOKENS = 8192       # MAX_NUM_BATCHED_TOKENS of the recipe: the server reserves the DMA memory for this size


def size_text(count):
    return f"{count // 1024:,} KiB" if count < 1024 * 1024 else f"{count // (1024 * 1024):,} MiB"


def b12x_allreduce():
    """The method of the all-reduce for each message size on Jovian Judgement r24. The limits in bytes are from the
    start log. The rows and the examples are a calculation from the source code: a message has one row for each
    token of a step, the two-shot method accepts rows in groups of four, and the DMA method has the batch limit."""
    limits = {r["setting"]: r["value"] for r in read("b12x_allreduce_limits.csv")}
    one, two, dma = (int(limits[k]) for k in ("one_shot_max_bytes", "two_shot_bf16_max_bytes", "dma_min_bytes"))
    assert limits["fused_max_bytes"] == limits["one_shot_max_bytes"], "the kernel for three steps has its own limit"
    top = BATCH_LIMIT_TOKENS * HIDDEN_STATE_BYTES
    t1, t2, t3 = one // HIDDEN_STATE_BYTES, two // HIDDEN_STATE_BYTES, -(-dma // HIDDEN_STATE_BYTES)
    first = (t1 // GPUS + 1) * GPUS   # the smallest number of rows above the one-shot limit that two-shot accepts
    assert first > t1 and t2 % GPUS == 0 and first % TOKENS_FOR_A_REQUEST == 0 and top > dma
    r1, r2 = t1 // TOKENS_FOR_A_REQUEST, t2 // TOKENS_FOR_A_REQUEST
    rows = [
        ["One-shot", f"{size_text(one)} or less", f"1 to {t1}",
         f"The check of the draft tokens for 1 {'or' if r1 == 2 else 'to'} {r1} requests"],
        ["Two-shot", f"More than {size_text(one)}, to {size_text(two)}", f"{first} to {t2}, in groups of {GPUS}",
         f"The check of the draft tokens for {first // TOKENS_FOR_A_REQUEST} to {r2} requests"],
        ["DMA with the copy engines", f"{size_text(dma)} to {size_text(top)}", f"{t3:,} to {BATCH_LIMIT_TOKENS:,}",
         f"A prefill step with {BATCH_LIMIT_TOKENS:,} tokens"],
        ["NCCL", "Each other size", f"{t2 + 1} to {t3 - 1:,}. Also {t1 + 1} to {t2 - 1}, if the number is not a multiple of {GPUS}.",
         "A prefill step with 500 tokens"],
    ]
    return table(["Method", "Size of one message", "Rows in the message", "Example of a step"], rows,
                 ["---", "---", "---", "---"])


def b12x_measured():
    """Three tests on the servers with the same weights: Jovian Judgement r24 and the official vLLM."""
    decode_rows, prefill_rows, d = read("decode_sparkdash.csv"), read("prefill_sparkdash.csv"), dict(DECODE_RUNS)
    servers = (VLLM, OFFICIAL, OFFICIAL_PCIE)

    def fill(server):
        return next(float(r["prefill_tok_s"]) for r in prefill_rows
                    if r["run"] == d[server] and r["target_tokens"] == "65536")

    def prose(server):
        return next(float(r["total_tok_s"]) for r in decode_rows
                    if r["run"] == d[server] and r["output_type"] == "prose" and int(r["concurrency"]) == 16)
    rows = [["Prefill of a cold prompt of 64K tokens, tokens/s", *[n0(fill(s)) for s in servers]],
            ["56K context, thinking on, top_p 0.95, top_k 20, 16 requests, tokens/s",
             *[long_text(s, "k20", 16) for s in servers]],
            ["Short greedy answers (prose), 16 requests, tokens/s", *[n0(prose(s)) for s in servers]]]
    return table(["Test", *servers], rows)


def modified_result():
    """TensorFold, TensorFold modified, and Jovian Judgement r24 in the tests that its changes apply to."""
    prefill_rows, d = read("prefill_sparkdash.csv"), dict(DECODE_RUNS)

    def fill(run):
        return next(float(r["prefill_tok_s"]) for r in prefill_rows if r["run"] == run and r["target_tokens"] == "65536")
    release, modified = cold_burst()
    servers = (TENSORFOLD, MODIFIED, VLLM)
    rows = [
        ["56K context, top_p 0.95, top_k 20, 16 requests, tokens/s", *[n0(long_cell(s, "k20", 16)) for s in servers]],
        ["56K context, top_p 0.95, no top_k, 16 requests, tokens/s", *[n0(long_cell(s, "off", 16)) for s in servers]],
        ["56K context, top_p 1.0, no top_k, 4 requests, tokens/s", *[n0(long_cell(s, "p1", 4)) for s in servers]],
        ["First token for four cold requests with a shared prefix of 42K tokens, s",
         f"{n0(min(release))} to {n0(max(release))}", n1(mean(modified)), span(burst_times(BURST_VLLM[0]))],
        ["Prefill of a cold prompt of 64K tokens, tokens/s", *[n0(fill(d[s])) for s in servers]],
    ]
    return table(["Test", *servers], rows)


def estimate_check():
    """The total speed of the long-context probe, as a percentage of the sum of the request speeds in the server log."""
    data = [r for r in read("longctx_estimate_check.csv") if r["all_requests_found_the_context"] == "True"]
    rows = []
    for server, run in ((TENSORFOLD, "tensorfold_session3"), (MODIFIED, "tensorfold_modified_best")):
        values = [float(r["probe_percent_of_log"]) for r in data if r["run"] == run]
        rows.append([server, len(values), f"{n0(min(values))}% to {n0(max(values))}%"])
    return table(["Inference stack", "Waves", "Speed from the probe, as a percentage of the speed from the log"], rows)


def overview():
    decode_rows, prefill_rows = read("decode_sparkdash.csv"), read("prefill_sparkdash.csv")
    summary = {r["run"]: r for r in read("quality_pi_summary.csv") if r["data_set"] == "both"}
    calls = {r["run"]: r for r in read("quality_pi_calls.csv")}
    d, reuse_runs = dict(DECODE_RUNS), dict(REUSE_RUNS)
    servers = (TENSORFOLD, MODIFIED, VLLM, JOVIAN28) + ((JOVIAN38,) if JOVIAN38 in d else ())

    def prose(server, concurrency):
        return next((float(r["total_tok_s"]) for r in decode_rows
                     if r["run"] == d[server] and r["output_type"] == "prose" and int(r["concurrency"]) == concurrency),
                    None)

    def fill(server):
        return next((float(r["prefill_tok_s"]) for r in prefill_rows
                     if r["run"] == d[server] and r["target_tokens"] == "65536"), None)

    def first(server, case):
        return mean(reuse_cells(reuse_runs.get(server), "56000")[case])

    def row(label, cell):   # one cell for each server; the two configurations of the official vLLM share a cell
        return [label, *[cell(s) for s in servers], f"{cell(OFFICIAL)} / {cell(OFFICIAL_PCIE)}"]

    def quality(source, field, form):
        """A value of the quality runs for each column. Jovian Judgement r24 has two runs."""
        def cell(*runs):
            values = [form(source[run][field]) for run in runs if run in source]
            return " and ".join(str(v) for v in values) if values else NONE
        return [cell("tensorfold"), cell("tensorfold_modified_best"), cell("vllm_links_on_8", "vllm_links_on_8_run2"),
                cell("jovian_r281_8"), *([cell("jovian_r38_8")] if JOVIAN38 in d else []),
                f"{cell('official_default_16')} / {NONE}"]
    release, modified = cold_burst()
    rows = [
        row("Short greedy answers (prose), 1 request, tokens/s", lambda s: n0(prose(s, 1))),
        row("Short greedy answers (prose), 16 requests, tokens/s", lambda s: n0(prose(s, 16))),
        row("Prefill of a cold prompt of 64K tokens, tokens/s", lambda s: n0(fill(s))),
        row("56K context, thinking on, top_p 0.95, top_k 20, 16 requests, tokens/s", lambda s: long_text(s, "k20", 16)),
        row("The same test with no top_k", lambda s: long_text(s, "off", 16)),
        row("The same test with top_p 1.0 and no top_k, 4 requests", lambda s: long_text(s, "p1", 4)),
        row("First token for a new turn on a chat of 83.6K tokens, s", lambda s: n1(first(s, "own reply + new turn"))),
        row("First token for the same prompt of 83.6K tokens a second time, s",
            lambda s: n1(first(s, "identical resend"))),
        ["First token for four cold requests with a shared prefix of 42K tokens, s",
         f"{n0(min(release))} to {n0(max(release))}", n1(mean(modified)), span(burst_times(BURST_VLLM[0])),
         span(burst_times(BURST_VLLM[1])), *([span(burst_times(BURST_R38))] if JOVIAN38 in d else []),
         f"{span(burst_times(BURST_VLLM[2]))} / {span(burst_times(BURST_VLLM[3]))}"],
        ["Python tasks that pass all tests through the pi agent, of 542", *quality(summary, "all_tests_pass", str)],
        ["Median time for a model call in the agent run, s", *quality(calls, "median_call_s", n2)],
    ]
    return table(["Test", *servers, "Official vLLM: default / tuned"], rows)


TABLES = {f.__name__: f for f in (decode_body, prefill_body, long_context, reuse_body, gpu_links, quality_pass_body,
                                  quality_paired_body, quality_work_body, modified_result, estimate_check, overview, decode,
                                  greedy_1k, prefill, round_times, sampler_sensitivity, reuse, burst, burst_body,
                                  burst_vllm,
                                  reuse_check, reuse_check_system, reuse_check_body, long_context_system,
                                  r281_policies, b12x_allreduce, b12x_measured,
                                  quality_pass,
                                  quality_paired, quality_work, modified_settings, modified_prefill_chunks, round_profile)}
if JOVIAN38 in dict(DECODE_RUNS):   # the table of release r38 is there when data/ has the release
    TABLES["r38_releases"] = r38_releases
if (DATA / "short_context_probe.csv").is_file() and read("short_context_probe.csv"):
    TABLES["short_context"] = short_context


def cells(line):
    """The cells of a table line with no padding. An alignment cell keeps only its colons."""
    line = line.strip()
    if not line.startswith("|"):
        return None
    return tuple(re.sub(r"-{3,}", "---", cell.strip()) for cell in line.strip("|").split("|"))


def main():
    arguments = sys.argv[1:]
    if arguments[:1] == ["--check"]:
        present = {cells(line) for line in Path(arguments[1]).read_text().splitlines()}
        missing = [(name, line) for name, make in TABLES.items() for line in make() if cells(line) not in present]
        for name, line in missing:
            print(f"{name}: not in {arguments[1]}: {line}")
        print(f"{sum(len(make()) - 2 for make in TABLES.values())} table rows examined, {len(missing)} not found")
        return 1 if missing else 0
    for name in arguments or TABLES:
        print(f"\n[{name}]\n" + "\n".join(TABLES[name]()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

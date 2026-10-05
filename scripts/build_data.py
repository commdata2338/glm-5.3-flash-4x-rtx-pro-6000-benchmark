"""Turn the raw benchmark output into the tidy CSV files under data/.

The raw output (sparkDash JSON, probe logs) stays on the test host; the CSV files are what the paper and the
figures use. Run it again whenever a window is re-measured.

  python3 scripts/build_data.py --results raw                  from the copy of the raw output in this repository
  python3 scripts/build_data.py --results /path/to/results     from the result folders of the test host

The two layouts hold the same files. The table in scripts/collect_raw.py gives the name of each host folder in
raw/. Use --out to write the CSV files into a different directory (for a comparison with data/).

Layout on the test host (the folder names are the run labels used on the host):
  tensorfold-vs-vllm-20261003/vllm-seqs16/         Jovian Judgement r24, 16 slots, GPU links off, stock chat template (before)
  tensorfold-vs-vllm-20261003-r2/tensorfold/       TensorFold
  tensorfold-vs-vllm-20261003-r2/vllm-seqs16-p2p/  Jovian Judgement r24, 16 slots, GPU links on, thinking-switch template
  tensorfold-vs-vllm-20261003-r2/vllm-live8-off/   Jovian Judgement r24, 8 slots, GPU links off, same boot
  glm-gpu-links-20261003/after/                    Jovian Judgement r24, 8 slots, GPU links on, same boot
  stock-vllm-v0.31.0/stock-auto-16/                official vLLM 0.31.0, default kernels and all-reduce
  stock-vllm-v0.31.0/stock-pcie-16/                official vLLM 0.31.0, its PCIe all-reduce on, sampled drafts
  stock-vllm-v0.31.0/<name>-pass2/                 the sparkDash jobs a second time on the same server
  tensorfold-topk20-20261003/tensorfold/           TensorFold, second session: the long-context test with top_k 20
  tensorfold-topk20-20261003/vllm-fork-16/         Jovian Judgement r24, 16 slots, second session: the same three passes
  tensorfold-fork-20261004/fork-NAME/              TensorFold modified, one folder for each group of settings
  tensorfold-fork-20261004/burst/                  four cold requests with the same long prefix, on each server
  quality-pi-20261004/RUN/                         task accuracy through the pi agent: records and grades
  followup-20261004/jovian-r281-16/                Jovian Judgement r28.1, 16 slots: the suite, then the three passes
  followup-20261004/jovian-r24-16/                 Jovian Judgement r24, 16 slots, in the same session
  followup-20261004/stock-auto-16/, stock-pcie-16/ official vLLM 0.31.0: the long-context test with top_k 20
  followup-20261004/burst/                         four cold requests with a shared prefix on the vLLM servers
  jovian-r38-test/jovian-r38-16/                   Jovian Judgement r38, 16 slots: the suite, then the three passes
  jovian-r38-test/jovian-r38-16-policy-aligned/    the same with --recurrent-checkpoint-policy aligned
  jovian-r38-test/jovian-r38-8/                    Jovian Judgement r38, 8 slots: the context as a system message
  jovian-r38-test/jovian-r24-16/                   Jovian Judgement r24, 16 slots, in the same session as release r38
  jovian-r38-test/burst/                           four cold requests with a shared prefix on release r38
"""
import argparse
import csv
import json
import math
import re
from pathlib import Path

import collect_raw

DATA = Path(__file__).resolve().parent.parent / "data"
# host folder or file: its name in raw/ (longest host name first, so that a folder wins over its parent)
PUBLIC = sorted(((host, public) for public, host in {**collect_raw.RUNS, **collect_raw.SINGLE}.items()),
                key=lambda pair: -len(pair[0]))


def src(root, relative):
    """A result path: as the test host has it under --results, or as raw/ of this repository has it."""
    path = root / relative
    if path.exists():
        return path
    for host, public in PUBLIC:
        if relative == host or relative.startswith(host + "/"):
            return root / (public + relative[len(host):])
    return path

RUNS = {  # run id: (engine, configuration, folder)
    "tensorfold": ("TensorFold", "recipe 1.0.1, 40 slots", "tensorfold-vs-vllm-20261003-r2/tensorfold"),
    "vllm_links_on_16": ("Jovian Judgement r24", "16 slots, GPU links on, thinking-switch template",
                         "tensorfold-vs-vllm-20261003-r2/vllm-seqs16-p2p"),
    "vllm_links_off_16_before": ("Jovian Judgement r24", "16 slots, GPU links off, stock template, before the IOMMU change",
                                 "tensorfold-vs-vllm-20261003/vllm-seqs16"),
    "vllm_links_off_8": ("Jovian Judgement r24", "8 slots, GPU links off, stock template",
                         "tensorfold-vs-vllm-20261003-r2/vllm-live8-off"),
    "vllm_links_on_8": ("Jovian Judgement r24", "8 slots, GPU links on, stock template", "glm-gpu-links-20261003/after"),
    "official_default_16": ("Official vLLM 0.31.0", "16 slots, default kernels and all-reduce, first pass",
                            "stock-vllm-v0.31.0/stock-auto-16"),
    "official_default_16_pass2": ("Official vLLM 0.31.0", "16 slots, default kernels and all-reduce, second pass",
                                  "stock-vllm-v0.31.0/stock-auto-16-pass2"),
    "official_pcie_16": ("Official vLLM 0.31.0", "16 slots, PCIe all-reduce on, sampled drafts, 22 capture sizes, "
                         "first pass", "stock-vllm-v0.31.0/stock-pcie-16"),
    "official_pcie_16_pass2": ("Official vLLM 0.31.0", "16 slots, PCIe all-reduce on, sampled drafts, 22 capture sizes, "
                               "second pass", "stock-vllm-v0.31.0/stock-pcie-16-pass2"),
}
# Session 5 of the paper (4 October, afternoon): release r28.1 of the fork with the settings of release r24.
FOLLOWUP = "followup-20261004"
RUNS["jovian_r281_16"] = ("Jovian Judgement r28.1", "16 slots, GPU links on, thinking-switch template, session 5",
                          f"{FOLLOWUP}/jovian-r281-16")
# The same with one more server argument, --recurrent-checkpoint-policy aligned (the checkpoint policy of r24).
RUNS["jovian_r281_16_aligned"] = ("Jovian Judgement r28.1", "16 slots, GPU links on, thinking-switch template, "
                                  "checkpoint policy aligned, session 5", f"{FOLLOWUP}/jovian-r281-16-policy-aligned")
# Session 6 of the paper: release r38 of the fork with the settings of release r24, and release r24 one more time in
# the same session (the control). A run that has no result folder gives no rows.
R38 = "jovian-r38-test"
RUNS["jovian_r38_16"] = ("Jovian Judgement r38", "16 slots, GPU links on, thinking-switch template, session 6",
                         f"{R38}/jovian-r38-16")
RUNS["jovian_r38_16_aligned"] = ("Jovian Judgement r38", "16 slots, GPU links on, thinking-switch template, "
                                 "checkpoint policy aligned, session 6", f"{R38}/jovian-r38-16-policy-aligned")
RUNS["vllm_links_on_16_session6"] = ("Jovian Judgement r24", "16 slots, GPU links on, thinking-switch template, "
                                     "session 6", f"{R38}/jovian-r24-16")
# Session 3 of the paper (about four hours after session 1): the long-context test with top_k 20, then with no
# top_k, then with top_k 20 again, on TensorFold and on Jovian Judgement with 16 slots.
TOPK_RUNS = {
    "tensorfold_session3": ("TensorFold", "recipe 1.0.1, 40 slots, session 3",
                            "tensorfold-topk20-20261003/tensorfold"),
    "vllm_links_on_16_session3": ("Jovian Judgement r24", "16 slots, GPU links on, thinking-switch template, session 3",
                                  "tensorfold-topk20-20261003/vllm-fork-16"),
}
# Session 5: the same three passes on the official vLLM and on release r28.1 of the fork. On release r28.1 the
# requests of a wave ran one after another (status no_overlap), so its rows have first-token times and no speed.
TOPK_RUNS.update({
    "jovian_r281_16_topk": ("Jovian Judgement r28.1", "16 slots, GPU links on, thinking-switch template, session 5",
                            f"{FOLLOWUP}/jovian-r281-16"),
    "jovian_r281_16_aligned_topk": ("Jovian Judgement r28.1", "16 slots, GPU links on, thinking-switch template, "
                                    "checkpoint policy aligned, session 5", f"{FOLLOWUP}/jovian-r281-16-policy-aligned"),
    "official_default_16_session5": ("Official vLLM 0.31.0", "16 slots, default kernels and all-reduce, session 5",
                                     f"{FOLLOWUP}/stock-auto-16"),
    "official_pcie_16_session5": ("Official vLLM 0.31.0", "16 slots, PCIe all-reduce on, sampled drafts, 22 capture "
                                  "sizes, session 5", f"{FOLLOWUP}/stock-pcie-16"),
})
# Session 6: the same three passes on release r38 and on the control.
TOPK_RUNS.update({
    "jovian_r38_16_topk": ("Jovian Judgement r38", "16 slots, GPU links on, thinking-switch template, session 6",
                           f"{R38}/jovian-r38-16"),
    "jovian_r38_16_aligned_topk": ("Jovian Judgement r38", "16 slots, GPU links on, thinking-switch template, "
                                   "checkpoint policy aligned, session 6", f"{R38}/jovian-r38-16-policy-aligned"),
    "vllm_links_on_16_session6_topk": ("Jovian Judgement r24", "16 slots, GPU links on, thinking-switch template, "
                                       "session 6", f"{R38}/jovian-r24-16"),
})
TOPK_PASSES = (("longctx-p95-topk20.log", "20", 1), ("longctx-p95-topk-off.log", "off", 1),
               ("longctx-p95-topk20-pass2.log", "20", 2))
# Session 5: the long-context test with the shared context as a system message and the question as the user
# message (longctx.py --context-as-system). In the other long-context tests, the two are one user message.
SYSTEM_RUNS = {
    "vllm_links_on_8_session5": ("Jovian Judgement r24", "8 slots, GPU links on, stock template, session 5",
                                 f"{FOLLOWUP}/jovian-r24-8"),
    "jovian_r281_8": ("Jovian Judgement r28.1", "8 slots, GPU links on, stock template, session 5",
                      f"{FOLLOWUP}/jovian-r281-8"),
    "jovian_r281_16_aligned": ("Jovian Judgement r28.1", "16 slots, GPU links on, thinking-switch template, "
                               "checkpoint policy aligned, session 5", f"{FOLLOWUP}/jovian-r281-16-policy-aligned"),
}
SYSTEM_RUNS.update({
    "jovian_r38_8": ("Jovian Judgement r38", "8 slots, GPU links on, stock template, session 6",
                     f"{R38}/jovian-r38-8"),
    "jovian_r38_16_aligned": ("Jovian Judgement r38", "16 slots, GPU links on, thinking-switch template, "
                              "checkpoint policy aligned, session 6", f"{R38}/jovian-r38-16-policy-aligned"),
})
SYSTEM_PASSES = (("longctx-system-p95-topk20.log", 0.95, "20", 1), ("longctx-system-p95-topk-off.log", 0.95, "off", 1),
                 ("longctx-system-p1.log", 1.0, "off", 1), ("longctx-system-p95-topk20-pass2.log", 0.95, "20", 2))
# Session 4 of the paper: TensorFold modified. Each folder is one start of the server with one group of settings.
# All its switches are off in the release; a row names the switches that are on. The result folders of the test
# host have the name of that time ("fork"); the arm names in data/ say "modified".
MODIFIED = "tensorfold-fork-20261004"
RUNS["tensorfold_modified_best"] = ("TensorFold modified", "GPU sampler, burst reuse and IPC all-gathers on, 40 slots",
                                    f"{MODIFIED}/fork-best")


def arm_name(folder):
    """The name of an arm in data/: the folder of the test host with "modified" in the place of "fork"."""
    return folder.replace("fork-", "modified-", 1)


MODIFIED_ARMS = {   # the folder of the arm on the test host: its settings
    "fork-sampler": "GPU sampler for rows with no top_k",
    "fork-s+burst": "GPU sampler, burst reuse",
    "fork-s+block4": "GPU sampler, draft block of 4 rows",
    "fork-s+depth": "GPU sampler, draft depth scale 0.25",
    "fork-s+l2pf": "GPU sampler, weight prefetch for 128 rows",
    "fork-s+graphstep": "GPU sampler, one CUDA graph for each batch width",
    "fork-s+prefill8192": "GPU sampler, prompt chunks of 8,192 rows",
    "fork-s+prof": "GPU sampler, segment profiler on",
    "fork-s+ipc": "GPU sampler, IPC all-gathers",
    "fork-best": "GPU sampler, burst reuse, IPC all-gathers",
}
MODIFIED_PROBES = {  # log name: (probe, top_p, top_k, thinking)
    "longctx-topk20": ("long_context_sampled", 0.95, "20", "on, effort max"),
    "longctx-topkoff": ("long_context_sampled", 0.95, "off", "on, effort max"),
    "longctx-p1": ("long_context_sampled", 1.0, "off", "on, effort max"),
    "longctx-p1short": ("short_context_sampled", 1.0, "off", "on, effort max"),
    "longctx-greedy": ("greedy_1k", 1.0, "off", "off"),
}
BURST = {  # log name: (server, note)
    "burst-release": ("TensorFold, recipe 1.0.1", ""),
    "burst-fork-off": ("TensorFold modified, all switches off",
                       "first long prompt after the server start: the prompt kernels were not yet used"),
    "burst-fork-s+burst": ("TensorFold modified, GPU sampler and burst reuse on", ""),
}
BURST_VLLM = {  # log name under followup-20261004/burst: server. These servers give no reply token ids.
    "burst-jovian-r24-8slot": "Jovian Judgement r24, 8 slots",
    "burst-jovian-r281-8slot": "Jovian Judgement r28.1, 8 slots",
    "burst-jovian-r281-16slot-default-policy": "Jovian Judgement r28.1, 16 slots",
    "burst-jovian-r281-16slot-policy-aligned": "Jovian Judgement r28.1, 16 slots, checkpoint policy aligned",
    "burst-official-default": "Official vLLM, default",
    "burst-official-tuned": "Official vLLM, tuned",
}
BURST_R38 = {  # log name under jovian-r38-test/burst: server
    "burst-jovian-r38-8slot": "Jovian Judgement r38, 8 slots",
    "burst-jovian-r38-16slot-default-policy": "Jovian Judgement r38, 16 slots",
    "burst-jovian-r38-16slot-policy-aligned": "Jovian Judgement r38, 16 slots, checkpoint policy aligned",
    # release r24 in the same session: the control with 16 slots, and the usual configuration before release r38
    "burst-jovian-r24-16slot-session6": "Jovian Judgement r24, 16 slots, session 6",
    "burst-jovian-r24-8slot-session6": "Jovian Judgement r24, 8 slots, session 6",
}
# Session 5: one prompt of 56K tokens sent again, and the same context with a different question
# (probes/prefix_reuse_check.py). folder under followup-20261004: (server, configuration, checkpoint setting)
REUSE_CHECK = {
    "jovian-r24-8": ("Jovian Judgement r24", "8 slots, GPU links on, stock template", "release default"),
    "jovian-r281-16": ("Jovian Judgement r28.1", "16 slots, GPU links on, thinking-switch template", "release default"),
    "jovian-r281-8": ("Jovian Judgement r28.1", "8 slots, GPU links on, stock template", "release default"),
    "jovian-r281-16-aligned": ("Jovian Judgement r28.1", "16 slots, GPU links on, thinking-switch template",
                               "--prefix-cache-retention-interval None"),
    "jovian-r281-16-policy-aligned": ("Jovian Judgement r28.1", "16 slots, GPU links on, thinking-switch template",
                                      "--recurrent-checkpoint-policy aligned"),
    "stock-auto-16": ("Official vLLM 0.31.0", "16 slots, default kernels and all-reduce", "release default"),
    "stock-pcie-16": ("Official vLLM 0.31.0", "16 slots, PCIe all-reduce on, sampled drafts, 22 capture sizes",
                      "release default"),
}
# Session 6, release r38. As for release r28.1, the check with a user message is from the server with 16 slots and
# the check with a system message is from the server with 8 slots; the policy "aligned" has the two forms.
# folder under jovian-r38-test: (server, configuration, checkpoint setting, the forms that the tables use)
REUSE_CHECK_R38 = {
    "jovian-r38-16": ("Jovian Judgement r38", "16 slots, GPU links on, thinking-switch template", "release default",
                      ("user message",)),
    "jovian-r38-8": ("Jovian Judgement r38", "8 slots, GPU links on, stock template", "release default",
                     ("system message",)),
    "jovian-r38-16-policy-aligned": ("Jovian Judgement r38", "16 slots, GPU links on, thinking-switch template",
                                     "--recurrent-checkpoint-policy aligned", ("user message", "system message")),
}
# Session 6: the short-context probe (probes/short_context.py) on the two releases with 8 slots.
# folder under jovian-r38-test: (server, configuration)
SHORT_CONTEXT = {
    "jovian-r24-live": ("Jovian Judgement r24", "8 slots, GPU links on, stock template"),
    "jovian-r38-8": ("Jovian Judgement r38", "8 slots, GPU links on, stock template"),
}
SEGMENTS = ("experts", "shared", "dsa", "kda", "dense", "head", "exchange", "glue")
QUALITY = {  # run folder under quality-pi-20261004: (run id, engine, configuration)
    "tensorfold-release-full-c8": ("tensorfold", "TensorFold", "recipe 1.0.1"),
    "tensorfold-fast-full-c8": ("tensorfold_modified_best", "TensorFold modified",
                                "GPU sampler, burst reuse and IPC all-gathers on"),
    "vllm-full-c8": ("vllm_links_on_8", "Jovian Judgement r24", "8 slots, GPU links on, stock template"),
    "jovian-r24-full-c8-run2": ("vllm_links_on_8_run2", "Jovian Judgement r24",
                                "8 slots, GPU links on, stock template, second run (session 5)"),
    "jovian-r281-full-c8": ("jovian_r281_8", "Jovian Judgement r28.1", "8 slots, GPU links on, stock template"),
    "official-default-full-c8": ("official_default_16", "Official vLLM 0.31.0",
                                 "16 slots, default kernels and all-reduce"),
    "jovian-r38-full-c8": ("jovian_r38_8", "Jovian Judgement r38", "8 slots, GPU links on, stock template"),
}
QUALITY_PAIRS = (("tensorfold", "vllm_links_on_8"), ("tensorfold", "tensorfold_modified_best"),
                 ("tensorfold_modified_best", "vllm_links_on_8"),
                 # session 5: two runs of one configuration, the newer release, and the official vLLM
                 ("vllm_links_on_8", "vllm_links_on_8_run2"), ("vllm_links_on_8", "jovian_r281_8"),
                 ("vllm_links_on_8_run2", "jovian_r281_8"),
                 ("vllm_links_on_8", "official_default_16"), ("tensorfold", "official_default_16"),
                 # session 6: release r38 against the two runs of release r24 and against release r28.1
                 ("vllm_links_on_8", "jovian_r38_8"), ("vllm_links_on_8_run2", "jovian_r38_8"),
                 ("jovian_r281_8", "jovian_r38_8"))
TYPES = ("prose", "code", "structured", "json")
# The eight sampling-sensitivity probes on TensorFold, in the order they were run (four streams, about 1K context).
SENSITIVITY = [
    ("greedy", 0.0, 1.0, "off", False), ("temperature 1, top_p 1.0, no top_k", 1.0, 1.0, "off", False),
    ("greedy", 0.0, 1.0, "off", True), ("temperature 1, top_p 1.0, no top_k", 1.0, 1.0, "off", True),
    ("temperature 1, top_p 0.95, no top_k", 1.0, 0.95, "off", True),
    ("temperature 1, top_p 0.99, no top_k", 1.0, 0.99, "off", True),
    ("temperature 1, top_p 1.0, top_k 50", 1.0, 1.0, "50", True),
    # the request gave no sampler settings: the TensorFold server then uses top_p 0.95 and top_k 20
    ("temperature 1, server defaults (top_p 0.95, top_k 20)", 1.0, 0.95, "20 (server default)", True),
]


def write(name, header, rows):
    with open(DATA / name, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def wave_rows(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.startswith("{") and '"concurrency"' in line]


def modified_rows(root):
    """TensorFold modified: the waves of each arm, the cold burst, the direct prompt probe, the parts of a round."""
    arms, burst, prefill, profile = [], [], [], []
    for folder, settings in MODIFIED_ARMS.items():
        logs = []
        directory = src(root, f"{MODIFIED}/{folder}")
        for log in directory.glob("longctx-*.log"):
            stem, number = log.stem, 1
            match = re.fullmatch(r"(.+)-run(\d+)", stem)
            if match:
                stem, number = match.group(1), int(match.group(2))
            if stem == "longctx-p1" and number > 1 and (directory / "longctx-p1short.log").is_file():
                number -= 1  # the window tool counted the short-context run as the first run of this name
            if stem in MODIFIED_PROBES:
                logs.append((list(MODIFIED_PROBES).index(stem), number, log))
        for index, number, log in sorted(logs):
            probe, top_p, top_k, thinking = MODIFIED_PROBES[list(MODIFIED_PROBES)[index]]
            for r in wave_rows(log):
                arms.append([arm_name(folder), settings, probe, 0.0 if probe == "greedy_1k" else 1.0, top_p, top_k, number,
                             thinking, r["concurrency"], r.get("context_tokens"), r.get("output_tokens"),
                             r.get("aggregate_decode_tps"), r.get("median_decode_tps"), r.get("median_ttft_s"),
                             r.get("status")])
        path = directory / "prefill-direct.jsonl"
        if path.is_file():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                prefill.append([arm_name(folder), settings, r["target_tokens"], r["prompt_tokens"], r["prefill_tok_s"], r["ttft_s"]])
    for name, (server, note) in BURST.items():
        path = src(root, f"{MODIFIED}/burst") / f"{name}.jsonl"
        if path.is_file():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                usage = r["usage"]
                burst.append([server, r["name"], usage["prompt_tokens"], usage["prompt_tokens_details"]["cached_tokens"],
                              usage["completion_tokens"], round(r["ttft_s"], 3), r["token_sha256"][:16], note])
    for folder, names in ((f"{FOLLOWUP}/burst", BURST_VLLM), (f"{R38}/burst", BURST_R38)):
        for name, server in names.items():
            path = src(root, folder) / f"{name}.jsonl"
            if path.is_file():
                for line in path.read_text().splitlines():
                    r = json.loads(line)
                    usage = r["usage"]
                    burst.append([server, r["name"], usage["prompt_tokens"],
                                  (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                                  usage["completion_tokens"], round(r["ttft_s"], 3), "",
                                  "the same requests with a different first line and no priority field"])
    log = src(root, f"{MODIFIED}/fork-s+prof") / "rank0.log"
    if log.is_file():
        # A report line covers the 160 rounds before it. Its context is the prompt size of the last request that
        # ended with a full answer; before the first such request, it is the size of the short request that starts
        # each wave. The wave in progress is the group of full answers that end next. If a short request ended
        # between the report before and this report, the 160 rounds can be from two waves.
        lines = log.read_text(errors="replace").splitlines()
        done = [(i, re.search(r"\] done req-\w+ prompt=(\d+) cached=\d+ thinking=\w+ tokens=(\d+) ", line))
                for i, line in enumerate(lines)]
        done = [(i, int(m.group(1)), int(m.group(2))) for i, m in done if m]
        context, full, last_report = None, False, -1
        for i, line in enumerate(lines):
            for j, prompt, tokens in done:
                if j == i and (tokens > 16 or not full):
                    context, full = prompt, full or tokens > 16
            match = re.search(r"segment profile, (\d+) rounds .*?, ([\d.]+) rows a round.*verify GPU ([\d.]+) =", line)
            if not match:
                continue
            wave = []
            for j, prompt, tokens in done:
                if j > i:
                    if tokens <= 16:
                        break
                    wave.append(j)
            mixed = any(last_report < j < i and tokens <= 16 for j, prompt, tokens in done)
            # the full answers of the wave that ended before this line also count
            before = []
            for j, prompt, tokens in reversed(done):
                if j < i:
                    if tokens <= 16:
                        break
                    before.append(j)
            parts = {k: float(v) for k, v in re.findall(r"(\w+) (-?[\d.]+) \(", line)}
            spans = dict(re.findall(r"(sampler|commit) span ([\d.]+)", line))
            profile.append([context, len(wave) + len(before), mixed, int(match.group(1)), float(match.group(2)),
                            float(match.group(3)), *[parts[k] for k in SEGMENTS], float(spans["sampler"]),
                            float(spans["commit"])])
            last_report = i
    write("modified_arms.csv", ["arm", "settings", "probe", "temperature", "top_p", "top_k", "pass", "thinking", "concurrency",
                            "context_tokens", "output_tokens", "total_tok_s", "median_stream_tok_s", "median_ttft_s",
                            "status"], arms)
    write("cold_burst.csv", ["server", "request", "prompt_tokens", "cached_tokens_reported", "completion_tokens",
                                  "ttft_s", "reply_tokens_sha256_first16", "note"], burst)
    write("modified_prefill_direct.csv", ["arm", "settings", "target_tokens", "prompt_tokens", "prefill_tok_s", "ttft_s"],
          prefill)
    write("modified_round_profile.csv", ["context_tokens", "requests_in_the_wave", "rounds_from_more_than_one_wave",
                                     "profiled_rounds", "rows_a_round", "verify_gpu_ms",
                                     *[f"{k}_ms" for k in SEGMENTS], "sampler_span_ms", "commit_span_ms"], profile)


B12X_LIMITS = re.compile(r"Using B12X PCIe all-reduce \(algorithm=(\w+), one-shot max=(\d+), fused max=(\d+), "
                         r"two-shot bf16 max=(\d+), DMA min=(\d+)\)")


def b12x_limit_rows(root):
    """The size limits of the B12X PCIe all-reduce, from the start log of Jovian Judgement r24 with the links on."""
    path = src(root, "glm-gpu-links-20261003/start/allreduce-lines.log")
    match = B12X_LIMITS.search(path.read_text(errors="replace"))
    assert match, f"{path}: no line with the limits of the B12X PCIe all-reduce"
    names = ("algorithm_for_four_gpus", "one_shot_max_bytes", "fused_max_bytes", "two_shot_bf16_max_bytes", "dma_min_bytes")
    write("b12x_allreduce_limits.csv", ["setting", "value"], [list(pair) for pair in zip(names, match.groups())])


def reuse_check_rows(root):
    """The prompt-reuse check: the seconds to the full reply (16 or 300 tokens) of each step, for each server."""
    rows = []
    for session, checks in ((FOLLOWUP, REUSE_CHECK), (R38, REUSE_CHECK_R38)):
        for folder, (server, config, setting, *forms) in checks.items():
            for name, place in (("cache-check.jsonl", "user message"), ("cache-check-system.jsonl", "system message")):
                if forms and place not in forms[0]:
                    continue
                # On the 8-slot server of release r28.1, the first check with a system message is
                # early-look-system.jsonl. A second check (cache-check-system.jsonl) ran after other tests had put
                # that system message in the cache.
                if folder == "jovian-r281-8" and place == "system message":
                    name = "early-look-system.jsonl"
                path = src(root, f"{session}/{folder}") / name
                if not path.is_file():
                    continue
                for number, line in enumerate(path.read_text().splitlines(), 1):
                    r = json.loads(line)
                    rows.append([server, config, setting, r.get("context_in", place), number, r["step"],
                                 r["shared_context_tokens"], r["prompt_tokens"], r["completion_tokens"],
                                 r["seconds"]])
    write("prefix_reuse_check.csv", ["server", "configuration", "checkpoint_setting", "shared_context_in",
                                     "step_number", "step", "shared_context_tokens", "prompt_tokens", "output_tokens",
                                     "seconds_to_the_full_reply"], rows)


def short_context_rows(root):
    """The short-context probe: for each group of 8 prompt lengths, the mean probability of the correct subsequent
    token, for the lengths that are a multiple of 4 and for the other lengths."""
    rows = []
    for folder, (server, config) in SHORT_CONTEXT.items():
        path = src(root, f"{R38}/{folder}") / "shortctx.jsonl"
        if not path.is_file():
            continue
        groups = {}
        for line in path.read_text().splitlines():
            r = json.loads(line)
            logprob = r["expected_logprob"]
            groups.setdefault((r["run"], r["length"] % 4 == 0), []).append(
                (math.exp(logprob) if logprob is not None else 0.0, r["generated"] == r["expected"]))
        for (first, whole), values in sorted(groups.items(), key=lambda item: (item[0][0], not item[0][1])):
            rows.append([server, config, first, first + 7, whole, len(values),
                         round(sum(p for p, _ in values) / len(values), 4),
                         round(100 * sum(hit for _, hit in values) / len(values), 1)])
    write("short_context_probe.csv", ["server", "configuration", "first_length", "last_length",
                                      "lengths_are_a_multiple_of_4", "prompts", "mean_probability_of_the_correct_token",
                                      "correct_first_choice_percent"], rows)


DONE = re.compile(r"done req-\w+ prompt=(\d+) cached=(\d+) thinking=\w+ tokens=(\d+) sha=\w+ finish=\w+ "
                  r"tok/s=([\d.]+) ttft=[\d.]+s prefill=[\d.]+s rounds=(\d+) accepted=(\d+)/(\d+)")
# The long-context waves in the request log of a TensorFold server, in the sequence of the session:
# (top_p, top_k, pass, concurrent requests)
ROUND_WAVES = {
    "tensorfold-topk20-20261003/tensorfold": ("tensorfold_session3", [
        (0.95, "20", 1, 4), (0.95, "20", 1, 8), (0.95, "20", 1, 16), (0.95, "off", 1, 4), (0.95, "off", 1, 8),
        (0.95, "off", 1, 16), (0.95, "20", 2, 4), (0.95, "20", 2, 8), (0.95, "20", 2, 16)]),
    f"{MODIFIED}/fork-best": ("tensorfold_modified_best", [
        (0.95, "20", 1, 4), (0.95, "20", 1, 8), (0.95, "20", 1, 16), (0.95, "off", 1, 4), (0.95, "off", 1, 8),
        (0.95, "off", 1, 16), (1.0, "off", 1, 4), (1.0, "off", 1, 8), (1.0, "off", 1, 12), (1.0, "off", 1, 16),
        (0.95, "20", 2, 4), (0.95, "20", 2, 8), (0.95, "20", 2, 16)]),
}


def probe_totals(run):
    """(top_p, top_k, pass, concurrent requests) -> the total tokens/s of the long-context probe for one run."""
    name, key, value = ("modified_arms.csv", "arm", "modified-best") if run == "tensorfold_modified_best" else \
        ("concurrent_waves.csv", "run", run)
    with open(DATA / name, newline="") as handle:
        return {(float(r["top_p"]), r["top_k"], int(r["pass"]), int(r["concurrency"])): float(r["total_tok_s"])
                for r in csv.DictReader(handle)
                if r[key] == value and r["probe"] == "long_context_sampled" and r["total_tok_s"] != ""}


def round_time_rows(root):
    """The time of a decode round of TensorFold, from the `done` lines of its request log.

    Each request of the long-context test writes 2,000 tokens. Its decode time is tokens / (tokens/s), and one round
    is that time divided by the number of rounds. A wave is a group of such requests between two short requests.
    The same lines give a check of the token estimate of the long-context probe (longctx_estimate_check.csv).
    """
    out, check = [], []
    for folder, (run, expected) in ROUND_WAVES.items():
        log = src(root, folder) / "rank0.log"
        if not log.is_file():
            continue
        waves, current = [], []
        for line in log.read_text(errors="replace").splitlines():
            match = DONE.search(line)
            if not match:
                continue
            prompt, cached, tokens, tps, rounds, accepted, drafted = (float(v) for v in match.groups())
            if tokens != 2000 or prompt < 50000:
                if current:
                    waves.append(current)
                    current = []
                continue
            current.append((1000 * tokens / tps / rounds, tokens / rounds, accepted, drafted, cached > 0, tps))
        if current:
            waves.append(current)
        waves = waves[:len(expected)]
        assert [len(w) for w in waves] == [e[3] for e in expected], f"{folder}: the waves are not as expected"
        probe = probe_totals(run)
        for (top_p, top_k, number, concurrency), wave in zip(expected, waves):
            times = sorted(w[0] for w in wave)
            middle = (times[(len(times) - 1) // 2] + times[len(times) // 2]) / 2
            out.append([run, top_p, top_k, number, concurrency, all(w[4] for w in wave), round(middle, 1),
                        round(times[0], 1), round(times[-1], 1), round(sum(w[1] for w in wave) / len(wave), 2),
                        round(sum(w[2] for w in wave) / sum(w[3] for w in wave), 2)])
            total = probe.get((top_p, top_k, number, concurrency))
            if total is not None:
                from_log = sum(w[5] for w in wave)
                check.append([run, top_p, top_k, number, concurrency, all(w[4] for w in wave), total,
                              round(from_log, 1), round(100 * total / from_log, 1)])
    write("tensorfold_round_times.csv", ["run", "top_p", "top_k", "pass", "concurrency", "all_requests_found_the_context",
                                         "median_round_ms", "min_round_ms", "max_round_ms", "tokens_a_round",
                                         "accepted_share_of_draft_tokens"], out)
    # The long-context probe scales characters to tokens for the TensorFold servers. The request log of the server
    # has the decode speed of each request. This file compares the two values for each wave.
    write("longctx_estimate_check.csv", ["run", "top_p", "top_k", "pass", "concurrency", "all_requests_found_the_context",
                                         "probe_total_tok_s", "log_sum_of_request_tok_s", "probe_percent_of_log"], check)


def wilson(k, n):
    """The 95% Wilson interval for k of n, in percent."""
    z = 1.959963984540054
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return round(100 * max(0, center - radius), 1), round(100 * min(1, center + radius), 1)


def mcnemar(b, c):
    """The exact two-sided McNemar test: Binomial(b + c, 1/2) on the tasks where only one run passes."""
    n = b + c
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def quality_rows(root):
    """Task accuracy through the pi agent: one row for each task, a summary, and the paired comparisons."""
    tasks, summary, paired, grades, exceptions = [], [], [], {}, []
    names = {"humaneval": "HumanEval+", "mbpp": "MBPP+"}
    for folder, (run, engine, config) in QUALITY.items():
        directory = src(root, f"quality-pi-20261004/{folder}")
        if not (directory / "graded.json").is_file():
            continue
        graded = {g["task_id"]: g for g in json.loads((directory / "graded.json").read_text())["tasks"]}
        records = [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines()]
        assert {r["task_id"] for r in records} == set(graded), f"{folder}: records and grades differ"
        grades[run] = graded
        first = {}
        if (directory / "first_calls.jsonl").is_file():
            first = {r["task_id"]: r["prompt_tokens"]
                     for r in map(json.loads, (directory / "first_calls.jsonl").read_text().splitlines())}
        for r in records:
            g = graded[r["task_id"]]
            tasks.append([run, engine, config, names[g["dataset"]], r["task_id"], g["base_pass"], g["plus_pass"],
                          r["status"], r["limit_hit"], r["solution_present"], round(r["wall_seconds"], 3),
                          r["model_calls"], first.get(r["task_id"]), r["total_input_tokens"], r["output_tokens"]])
        if (directory / "exceptions.jsonl").is_file():
            for line in (directory / "exceptions.jsonl").read_text().splitlines():
                e = json.loads(line)
                last = e["model_calls"][-1]
                g = graded[e["task_id"]]
                exceptions.append([run, engine, e["task_id"], e["status"], e["time_limit_hit"], e["solution_file_present"],
                                   g["base_pass"], g["plus_pass"], len(e["model_calls"]), last["output_tokens"],
                                   last["thought_tokens"], last["stop_reason"], e["tool_calls_started"],
                                   e["tool_calls_ended"], (e["last_tool_call"] or {}).get("tool")])
        for dataset in ("humaneval", "mbpp", "both"):
            rows = [g for g in graded.values() if dataset == "both" or g["dataset"] == dataset]
            n, base, plus = len(rows), sum(g["base_pass"] for g in rows), sum(g["plus_pass"] for g in rows)
            summary.append([run, engine, config, names.get(dataset, "both"), n, base, plus, round(100 * base / n, 1),
                            *wilson(base, n), round(100 * plus / n, 1), *wilson(plus, n)])
    for a, b in QUALITY_PAIRS:
        if a not in grades or b not in grades:
            continue
        assert grades[a].keys() == grades[b].keys(), "a paired comparison needs the same tasks"
        for dataset in ("humaneval", "mbpp", "both"):
            ids = [t for t, g in grades[a].items() if dataset == "both" or g["dataset"] == dataset]
            for tests in ("base", "plus"):
                pa = [grades[a][t][f"{tests}_pass"] for t in ids]
                pb = [grades[b][t][f"{tests}_pass"] for t in ids]
                both = sum(x and y for x, y in zip(pa, pb))
                only_a = sum(x and not y for x, y in zip(pa, pb))
                only_b = sum(y and not x for x, y in zip(pa, pb))
                n = len(ids)
                difference = 100 * (only_b - only_a) / n     # pass rate of b minus pass rate of a, percentage points
                error = 100 * math.sqrt(only_a + only_b - (only_b - only_a) ** 2 / n) / n
                paired.append([a, b, names.get(dataset, "both"), "base tests" if tests == "base" else "all tests",
                               n, both, only_a, only_b, n - both - only_a - only_b,
                               round(mcnemar(only_a, only_b), 4), round(difference, 4),
                               round(difference - 1.959964 * error, 4), round(difference + 1.959964 * error, 4)])
    # The relay of the harness records each model call: its start time and its duration. A run's calls are those
    # between the start of its first task and the end of its last task (timing.json, from the task folders).
    calls = []
    for folder, (run, engine, config) in QUALITY.items():
        directory = src(root, f"quality-pi-20261004/{folder}")
        if not ((directory / "timing.json").is_file() and (directory / "relay.jsonl").is_file()):
            continue
        timing = json.loads((directory / "timing.json").read_text())
        seconds = sorted(r["seconds"] for r in map(json.loads, (directory / "relay.jsonl").read_text().splitlines())
                         if r["path"].endswith("/chat/completions") and r.get("status") == 200
                         and timing["first_task_start"] <= r["time"] <= timing["last_task_end"])
        middle = (seconds[(len(seconds) - 1) // 2] + seconds[len(seconds) // 2]) / 2
        calls.append([run, engine, config, len(seconds), round(middle, 2),
                      round(sum(seconds) / len(seconds), 2), round(seconds[int(0.9 * len(seconds))], 2),
                      round(sum(seconds)), round((timing["last_task_end"] - timing["first_task_start"]) / 60, 1)])
    write("quality_pi_calls.csv", ["run", "engine", "configuration", "model_calls", "median_call_s", "mean_call_s",
                                   "p90_call_s", "sum_of_call_s", "run_minutes"], calls)
    write("quality_pi_tasks.csv", ["run", "engine", "configuration", "data_set", "task_id", "base_tests_pass",
                                   "all_tests_pass", "agent_status", "time_limit_hit", "solution_file_present",
                                   "wall_seconds", "model_calls", "first_prompt_tokens", "input_tokens",
                                   "output_tokens"], tasks)
    write("quality_pi_exceptions.csv", ["run", "engine", "task_id", "agent_status", "time_limit_hit",
                                        "solution_file_present", "base_tests_pass", "all_tests_pass", "model_calls",
                                        "last_call_output_tokens", "last_call_thought_tokens", "last_call_stop_reason",
                                        "tool_calls_started", "tool_calls_ended", "last_tool"], exceptions)
    write("quality_pi_summary.csv", ["run", "engine", "configuration", "data_set", "tasks", "base_tests_pass",
                                     "all_tests_pass", "base_tests_percent", "base_tests_ci95_low", "base_tests_ci95_high",
                                     "all_tests_percent", "all_tests_ci95_low", "all_tests_ci95_high"], summary)
    write("quality_pi_paired.csv", ["run_a", "run_b", "data_set", "tests", "tasks", "both_pass", "only_a_passes",
                                    "only_b_passes", "neither_passes", "exact_mcnemar_p",
                                    "b_minus_a_percentage_points", "difference_ci95_low", "difference_ci95_high"], paired)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=DATA, help="directory for the CSV files (default: data/)")
    args = parser.parse_args()
    root = args.results
    globals()["DATA"] = args.out
    DATA.mkdir(parents=True, exist_ok=True)

    decode, prefill, waves, reuse, integrity, forms, recheck = [], [], [], [], [], [], []
    for run, (engine, config, folder) in RUNS.items():
        directory = src(root, folder)
        for kind in TYPES:
            path = directory / f"sparkdash-decode-{kind}.json"
            if path.is_file():
                for r in json.loads(path.read_text())["results"]:
                    decode.append([run, engine, config, kind, r["concurrency"], r["streamsOk"], r["streamsFailed"],
                                   r["aggregateDecodeTps"], r["medianDecodeTps"], r["minDecodeTps"], r["maxDecodeTps"],
                                   r["medianTtftMs"]])
        path = directory / "sparkdash-prefill.json"
        if path.is_file():
            for r in json.loads(path.read_text())["results"]:
                prefill.append([run, engine, config, r["targetTokens"], r["promptTokens"], r["prefillTps"],
                                round(r["ttftMs"] / 1000, 3)])
        for log, probe, top_p, thinking in (("greedy.log", "greedy_1k", 1.0, "off"),
                                            ("longctx-sampled.log", "long_context_sampled", 1.0, "on, effort max"),
                                            ("longctx-sampled-p95.log", "long_context_sampled", 0.95, "on, effort max")):
            for r in wave_rows(directory / log):
                waves.append([run, engine, config, probe, 0.0 if probe == "greedy_1k" else 1.0, top_p, "off", 1,
                              thinking, r["concurrency"], r.get("context_tokens"), r.get("output_tokens"),
                              r.get("aggregate_decode_tps"), r.get("median_decode_tps"), r.get("median_ttft_s"),
                              r.get("status")])
        path = directory / "warm-chat.jsonl"
        if path.is_file():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                reuse.append([run, engine, config, r["size"], r["round"], r["case"], r["prompt_tokens"], r["cached_tokens"],
                              r["ttft_s"]])
        path = directory / "integrity.jsonl"
        if path.is_file():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                integrity.append([run, engine, config, r["target_tokens"], r["prompt_tokens"], r["ok"], r["seconds"]])
        path = directory / "integrity-recheck.jsonl"
        if path.is_file():  # the same question with room to answer, where a first answer was cut at 40 tokens
            for line in path.read_text().splitlines():
                r = json.loads(line)
                recheck.append([run, engine, config, r["target_tokens"], r["prompt_tokens"], r["max_tokens"],
                                r["completion_tokens"], r["finish_reason"], r["ok"], r["seconds"]])
        path = directory / "thinkoff.jsonl"
        if path.is_file():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                forms.append([run, engine, config, r["round"], r["type"], r["form"], r["prompt_tokens"],
                              r["completion_tokens"], r["decode_tps"]])
    quick = src(root, "tensorfold-vs-vllm-20261003-r2/thinkoff-live8-quicklook.jsonl")
    if quick.is_file():  # the stock template on the 8-slot server: the chat form still thinks
        for line in quick.read_text().splitlines():
            r = json.loads(line)
            forms.append(["vllm_links_off_8", "Jovian Judgement r24", "8 slots, GPU links off, stock template", r["round"], r["type"],
                          r["form"], r["prompt_tokens"], r["completion_tokens"], r["decode_tps"]])

    write("decode_sparkdash.csv", ["run", "engine", "configuration", "output_type", "concurrency", "streams_ok",
                                   "streams_failed", "total_tok_s", "median_stream_tok_s", "min_stream_tok_s",
                                   "max_stream_tok_s", "median_ttft_ms"], decode)
    write("prefill_sparkdash.csv", ["run", "engine", "configuration", "target_tokens", "prompt_tokens", "prefill_tok_s",
                                    "ttft_s"], prefill)
    for run, (engine, config, folder) in TOPK_RUNS.items():
        for log, top_k, number in TOPK_PASSES:
            for r in wave_rows(src(root, folder) / log):
                waves.append([run, engine, config, "long_context_sampled", 1.0, 0.95, top_k, number, "on, effort max",
                              r["concurrency"], r.get("context_tokens"), r.get("output_tokens"),
                              r.get("aggregate_decode_tps"), r.get("median_decode_tps"), r.get("median_ttft_s"),
                              r.get("status")])
    for run, (engine, config, folder) in SYSTEM_RUNS.items():
        for log, top_p, top_k, number in SYSTEM_PASSES:
            for r in wave_rows(src(root, folder) / log):
                waves.append([run, engine, config, "long_context_sampled_system_message", 1.0, top_p, top_k, number,
                              "on, effort max", r["concurrency"], r.get("context_tokens"), r.get("output_tokens"),
                              r.get("aggregate_decode_tps"), r.get("median_decode_tps"), r.get("median_ttft_s"),
                              r.get("status")])
    write("concurrent_waves.csv", ["run", "engine", "configuration", "probe", "temperature", "top_p", "top_k", "pass",
                                   "thinking",
                                   "concurrency", "context_tokens", "output_tokens", "total_tok_s",
                                   "median_stream_tok_s", "median_ttft_s", "status"], waves)
    write("chat_reuse_ttft.csv", ["run", "engine", "configuration", "nominal_size", "round", "case", "prompt_tokens",
                                  "cached_tokens_reported", "ttft_s"], reuse)
    write("integrity.csv", ["run", "engine", "configuration", "target_tokens", "prompt_tokens", "passphrase_recalled",
                            "seconds"], integrity)
    write("integrity_recheck.csv", ["run", "engine", "configuration", "target_tokens", "prompt_tokens", "max_tokens",
                                    "completion_tokens", "finish_reason", "passphrase_recalled", "seconds"], recheck)
    write("thinking_off_forms.csv", ["run", "engine", "configuration", "round", "output_type", "form", "prompt_tokens",
                                     "completion_tokens", "decode_tok_s"], forms)

    rows = wave_rows(src(root, "tensorfold-vs-vllm-20261003-r2/tensorfold") / "isolate.log")
    assert len(rows) == len(SENSITIVITY), f"expected {len(SENSITIVITY)} sensitivity probes, found {len(rows)}"
    write("tensorfold_sampling_sensitivity.csv",
          ["setting", "temperature", "top_p", "top_k", "thinking", "concurrency", "context_tokens", "output_tokens",
           "total_tok_s", "median_stream_tok_s"],
          [[label, temperature, top_p, top_k, "on, effort max" if thinking else "off", r["concurrency"],
            r["context_tokens"], r["output_tokens"], r["aggregate_decode_tps"], r["median_decode_tps"]]
           for (label, temperature, top_p, top_k, thinking), r in zip(SENSITIVITY, rows)])

    modified_rows(root)
    b12x_limit_rows(root)
    reuse_check_rows(root)
    short_context_rows(root)
    round_time_rows(root)
    quality_rows(root)

    matrix = json.loads(src(root, "tensorfold-vs-vllm-20261003-r2/p2p-matrix.json").read_text())
    write("gpu_to_gpu_copy.csv", ["source_gpu", "destination_gpu", "buffer_mib", "data_matches", "gib_per_s"],
          [[r["src"], r["dst"], matrix["buffer_mib"], r["match"], r["gib_per_s"]] for r in matrix["pairs"]])


if __name__ == "__main__":
    main()

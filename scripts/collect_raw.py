"""Copy the raw benchmark output into raw/, without private details of the test host.

  python3 scripts/collect_raw.py --results /path/to/results

The script copies only result files (tool output, probe output, engine start logs). It replaces host paths with
placeholders, then it examines each copied file. A file that still contains a private pattern is not kept: the
script writes its name to raw/NOT_INCLUDED.txt and stops with an error, so that a person can decide.
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
RUNS = {  # folder in raw/: folder under --results
    "tensorfold": "tensorfold-vs-vllm-20261003-r2/tensorfold",
    "jovian-judgement-r24-links-on-16-slots": "tensorfold-vs-vllm-20261003-r2/vllm-seqs16-p2p",
    "jovian-judgement-r24-links-off-16-slots-before": "tensorfold-vs-vllm-20261003/vllm-seqs16",
    "jovian-judgement-r24-links-off-8-slots": "tensorfold-vs-vllm-20261003-r2/vllm-live8-off",
    "jovian-judgement-r24-links-on-8-slots": "glm-gpu-links-20261003/after",
    "vllm-official-default-16-slots": "stock-vllm-v0.31.0/stock-auto-16",
    "vllm-official-default-16-slots-pass2": "stock-vllm-v0.31.0/stock-auto-16-pass2",
    "vllm-official-pcie-16-slots": "stock-vllm-v0.31.0/stock-pcie-16",
    "vllm-official-pcie-16-slots-pass2": "stock-vllm-v0.31.0/stock-pcie-16-pass2",
    "tensorfold-session3-top-k": "tensorfold-topk20-20261003/tensorfold",
    "jovian-judgement-r24-links-on-16-slots-session3-top-k": "tensorfold-topk20-20261003/vllm-fork-16",
    # TensorFold modified: one folder for each start of the server with one group of settings
    "tensorfold-modified/all-settings-off": "tensorfold-fork-20261004/fork-off",
    "tensorfold-modified/sampler": "tensorfold-fork-20261004/fork-sampler",
    "tensorfold-modified/sampler-burst": "tensorfold-fork-20261004/fork-s+burst",
    "tensorfold-modified/sampler-draft-block-4": "tensorfold-fork-20261004/fork-s+block4",
    "tensorfold-modified/sampler-draft-depth": "tensorfold-fork-20261004/fork-s+depth",
    "tensorfold-modified/sampler-weight-prefetch": "tensorfold-fork-20261004/fork-s+l2pf",
    "tensorfold-modified/sampler-graph-step": "tensorfold-fork-20261004/fork-s+graphstep",
    "tensorfold-modified/sampler-prefill-8192": "tensorfold-fork-20261004/fork-s+prefill8192",
    "tensorfold-modified/sampler-profiler": "tensorfold-fork-20261004/fork-s+prof",
    "tensorfold-modified/sampler-ipc": "tensorfold-fork-20261004/fork-s+ipc",
    "tensorfold-modified/sampler-burst-ipc": "tensorfold-fork-20261004/fork-best",
    "tensorfold-modified/burst-test": "tensorfold-fork-20261004/burst",
    "tensorfold-modified/reply-equality": "tensorfold-fork-20261004/corpus",
    "tensorfold-modified/gates": "tensorfold-fork-20261004/gates",
    # task accuracy through the pi agent: the records, the files that the agent wrote, and the grades
    "quality-pi/tensorfold": "quality-pi-20261004/tensorfold-release-full-c8",
    "quality-pi/tensorfold-modified": "quality-pi-20261004/tensorfold-fast-full-c8",
    "quality-pi/jovian-judgement-r24": "quality-pi-20261004/vllm-full-c8",
    # session 5: Jovian Judgement r28.1, the official vLLM with top_k 20, the cold requests, the prompt-reuse check,
    # and three more quality runs
    "session5/jovian-judgement-r28.1-16-slots": "followup-20261004/jovian-r281-16",
    "session5/jovian-judgement-r28.1-16-slots-dense-retention": "followup-20261004/jovian-r281-16-aligned",
    "session5/jovian-judgement-r28.1-16-slots-policy-aligned": "followup-20261004/jovian-r281-16-policy-aligned",
    "session5/jovian-judgement-r28.1-8-slots": "followup-20261004/jovian-r281-8",
    "session5/jovian-judgement-r24-8-slots": "followup-20261004/jovian-r24-8",
    "session5/vllm-official-default-16-slots": "followup-20261004/stock-auto-16",
    "session5/vllm-official-pcie-16-slots": "followup-20261004/stock-pcie-16",
    "session5/cold-requests": "followup-20261004/burst",
    "quality-pi/jovian-judgement-r24-run2": "quality-pi-20261004/jovian-r24-full-c8-run2",
    "quality-pi/jovian-judgement-r28.1": "quality-pi-20261004/jovian-r281-full-c8",
    "quality-pi/vllm-official-default": "quality-pi-20261004/official-default-full-c8",
}
SINGLE = {  # file in raw/: file under --results
    "gpu-to-gpu-copy-matrix.json": "tensorfold-vs-vllm-20261003-r2/p2p-matrix.json",
    "host-state.txt": "tensorfold-vs-vllm-20261003-r2/host-state.txt",
    "tensorfold-prepare.log": "tensorfold-vs-vllm-20261003-r2/tensorfold-prepare.log",
    "tensorfold-start.log": "tensorfold-vs-vllm-20261003-r2/tensorfold-start.log",
    "tensorfold-health-at-start.json": "tensorfold-vs-vllm-20261003-r2/tensorfold-health.json",
    "jovian-judgement-r24-links-on-8-slots-memory.json": "glm-gpu-links-20261003/start/memory.json",
    "jovian-judgement-r24-links-on-8-slots-allreduce-lines.log": "glm-gpu-links-20261003/start/allreduce-lines.log",
    # the official vLLM: the start attempts that did not give a server (the lines of each start log that matter)
    "vllm-official-start-attempt1-instanttensor.log": "stock-vllm-v0.31.0/stock-auto-16/attempt1-instanttensor-failed/startup.log",
    "vllm-official-start-attempt2-mtp-quantization-map.log": "stock-vllm-v0.31.0/stock-auto-16/attempt2-mtp-quant-map-failed/startup.log",
    "vllm-official-start-attempt3-autotune.log": "stock-vllm-v0.31.0/stock-auto-16/attempt3-autotune-stalled/startup.log",
    "vllm-official-b12x-experts-refused.log": "stock-vllm-v0.31.0/stock-b12x-16/startup.log",
    "tensorfold-modified/sampler-check-on-four-gpus.log": "tensorfold-fork-20261004/cuda-check-2.log",
    "jovian-judgement-r24-links-off-8-slots-thinking-off-quicklook.jsonl": "tensorfold-vs-vllm-20261003-r2/thinkoff-live8-quicklook.jsonl",
    # receipts: the check of the chat template, the tests of TensorFold modified with no GPU, and the comparison of
    # the installed package of its image with the patched source tree
    "chat-template-check.log": "tensorfold-vs-vllm-20261003-r2/template-check.log",
    "tensorfold-modified/tests-with-no-gpu.log": "tensorfold-fork-20261004/cpu-tests.log",
    "tensorfold-modified/package-compare.log": "tensorfold-fork-20261004/package-compare.log",
}
KEEP = re.compile(r"\.(json|jsonl|log|txt)$")
SKIP = re.compile(r"^(load\.log|warmup.*|wave-check.*)$")  # helper output that holds no result
REPLACE = [  # longest first
    (re.compile(r"/home/[A-Za-z0-9_.-]+/models/tensorfold-glm53"), "<tensorfold-dir>"),
    (re.compile(r"/home/[A-Za-z0-9_.-]+/models/[A-Za-z0-9_.-]+/results"), "<results-dir>"),
    (re.compile(r"/home/[A-Za-z0-9_.-]+/\.cache/[A-Za-z0-9_.-]+"), "<cache-dir>"),
    (re.compile(r"/media/[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]*huggingface"), "<weights-dir>"),
    (re.compile(r"/home/[A-Za-z0-9_.-]+"), "<home>"),
    (re.compile(r"/media/[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]*"), "<data-dir>"),
    (re.compile(r"\b172\.(1[6-9]|2\d|3[01])\.\d+\.\d+(:\d+)?"), "<container-address>"),  # the container network
    # NCCL log lines start with a time, then the host name, then process and thread numbers
    (re.compile(r"(?m)^(\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] )[A-Za-z0-9_.-]+(:\d+:\d+ \[)"), r"\1<host>\2"),
]
PRIVATE = re.compile(r"/home/|/media/|@[A-Za-z0-9-]+\.(com|net|org)|\b100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+\b", re.I)


def clean(text):
    for pattern, placeholder in REPLACE:
        text = pattern.sub(placeholder, text)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--private-words", default="", help="comma-separated words that must not occur (host, user)")
    args = parser.parse_args()
    words = [w for w in args.private_words.split(",") if w]
    pairs = [(RAW / name, args.results / source) for name, source in SINGLE.items()]
    for name, folder in RUNS.items():
        if not (args.results / folder).is_dir():
            print(f"no folder for {name}: {folder}")
            continue
        for path in sorted((args.results / folder).iterdir()):
            if path.is_file() and KEEP.search(path.name) and not SKIP.match(path.name):
                pairs.append((RAW / name / path.name, path))
    bad, count = [], 0
    for target, source in pairs:
        if not source.is_file():
            continue
        text = clean(source.read_text(encoding="utf-8", errors="replace"))
        hits = [w for w in words if re.search(re.escape(w), text, re.I)]
        if PRIVATE.search(text) or hits:
            bad.append(str(target.relative_to(ROOT)))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        count += 1
    print(f"copied {count} files into raw/")
    # The list of refused files is always written, so that an old list cannot stay after a correction.
    (RAW / "NOT_INCLUDED.txt").write_text(("\n".join(bad) if bad else "none") + "\n")
    if bad:
        print("NOT INCLUDED (private pattern found):", *bad, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

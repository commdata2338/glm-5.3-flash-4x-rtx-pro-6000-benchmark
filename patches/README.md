# Patches

`tensorfold-modified-engine.patch` contains the changes of TensorFold modified (section 9 and Appendix E.4 of the paper).

- **Base.** TensorFold v0.6.2 (`56e2e3ec55bc`) with the 87 patches of recipe release 1.0.1 (`bdf4f18`). The command
  `scripts/apply-patches.sh` of the recipe makes this tree.
- **How to apply.** In the tree, run `git apply tensorfold-modified-engine.patch`. The patch applies with no offset.
- **Contents.** The patch changes 17 files. It adds the sampler on the GPU for rows with no top_k, and the reuse for
  requests that arrive together. It also adds the cache causes and the scheduler credit from measured time. It
  contains one correction to the profiler, the tests, and two tools.
- **Settings.** Each change has a setting, and each setting is off by default.
  [`../recipes/tensorfold-modified.md`](../recipes/tensorfold-modified.md) gives the settings and the tests.
- **License.** TensorFold has the Apache License 2.0. The patch is a change to TensorFold and has the same license.
- **Status.** This patch is not a release of TensorFold, and the TensorFold project did not examine it. AI tools for
  code wrote the changes. Our tests are in sections 2 to 9 of the paper.

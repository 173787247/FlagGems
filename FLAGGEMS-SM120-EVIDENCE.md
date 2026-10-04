# FlagGems on sm_120 (RTX 5080): environment, a reproducible crash, and what #4968 changes

Measured 2026-10-04 on WSL2 / Ubuntu 24.04, NVIDIA GeForce RTX 5080 (sm_120),
driver 591.86, CUDA 13.0.88.

This is an outsider's report, produced while evaluating FlagGems as a place to
contribute. Everything below is reproducible from the commands shown. It is not
an upstream statement, and nothing here has been reviewed by the maintainers.

**Summary: two independent problems, not one.**

| # | symptom | status |
|---|---|---|
| ① | `flag_gems.ops.add` twice in a row → **SIGSEGV** | an open PR, #4968, removes the crash |
| ② | after ① is fixed, the launch raises `TypeError` | **not** covered by #4968 |

---

## 1. Working environment (verified)

```sh
python3 -m venv ~/flag-gems-venv                       # Python 3.12.3
~/flag-gems-venv/bin/pip install torch==2.12.0 \
    --index-url https://download.pytorch.org/whl/cu130  # → torch 2.12.0+cu130, triton 3.7.0
~/flag-gems-venv/bin/pip install --no-deps flag-gems packaging sqlalchemy greenlet "PyYAML==6.0.3"
~/flag-gems-venv/bin/pip install numpy==1.26.4
```

- `torch` brings its own `triton`; 2.12.0+cu130 brings **3.7.0**, not the 3.4.0 that
  `pyproject.toml` lists. That gap is what problem ② comes from.
- **numpy must be installed explicitly** when using `--no-deps`: without it the first
  `add` call segfaults inside numpy rather than raising `ImportError`.
- The `numpy==1.26.4` pin is annotated in `pyproject.toml` as a Python 3.10 constraint;
  on 3.12 the plain PyPI numpy is what got installed here.

`[INFO] : FlagGems Unsupported GPU arch specialization` is printed on this machine.
It is informational — it also appears on runs that succeed — and is **not** the cause
of either problem below.

## 2. Problem ①: the second call segfaults

```python
import torch, flag_gems
a = torch.randn((1, 4096), dtype=torch.bfloat16, device="cuda")
b = torch.randn((1, 4096), dtype=torch.bfloat16, device="cuda")
flag_gems.ops.add(a, b)   # ok
flag_gems.ops.add(a, b)   # SIGSEGV, exit 139
```

`faulthandler` trace:

```
Fatal Python error: Segmentation fault
  triton/backends/nvidia/driver.py:328 in __call__
  triton/compiler/compiler.py:501 in runner
  flag_gems/utils/libentry.py:812 in run
  ~/.flaggems/code_cache/pointwise_dynamic_..._add_func_kernel_rank_1_bptr.py:42
  flag_gems/utils/pointwise_dynamic.py:1307 in _call_real_impl
  flag_gems/utils/pointwise_dynamic.py:1301 in __call__
  flag_gems/ops/add.py:87 in add
```

A bare, hand-written Triton kernel launched three times in a row is **fine** in this
same environment, which rules out Triton, the GPU and the driver:

```
triton 3.7.0 : launch 1 ok / launch 2 ok / launch 3 ok
triton 3.8.0 : launch 1 ok / launch 2 ok / launch 3 ok
```

So the fault is specific to the kernel `pointwise_dynamic` generates.

## 3. What #4968 does to it

Applying [#4968](https://github.com/flagos-ai/FlagGems/pull/4968) to a checkout of
`main` (`2d50c88`) and installing it editable changes the failure mode:

```
before : second call → Segmentation fault (exit 139)
after  : second call → TypeError (exit 1)
```

```
  File ".../triton/backends/nvidia/driver.py", line 328, in __call__
    self.launch(gridX, gridY, gridZ, stream, function, self.launch_cooperative_grid, ...)
TypeError: 'type' object cannot be interpreted as an integer
```

That is a useful result in itself: the wrong element dtype in the block-pointer store
was producing a **memory-level** failure, and #4968 turns it into a diagnosable error.
A library should not be able to segfault a Python process, so the PR is worth having
on those grounds alone — but it does not make the operator work here.

## 4. Problem ②: the launch signature

After #4968 the call reaches the launcher and fails there. The launcher is Triton's,
but the arguments come from FlagGems (`flag_gems/utils/libentry.py`, the caching
launcher). FlagGems targets triton 3.4; this environment has 3.7.

**Not investigated further.** No claim is made about which side needs to change — the
likely explanation is the version gap, and testing it needs a torch build that pairs
with triton 3.4, which is a deliberate environment decision rather than a quick edit.

## 5. The other failure mode, for completeness

In a *different* environment (tilelang's venv: torch 2.14.0+cu130, triton **3.8.0**,
numpy 2.5.3) the same operator fails earlier, during codegen:

```
AttributeError("'_aggregate_type' object has no attribute 'element_ty'")
    in the generated kernel: out0_bptr.type.element_ty
```

That is exactly what #4968 fixes, and it does not reach a launch at all. Two Triton
versions, two different symptoms, same file.

## 6. What is established, and what is not

**Established**

- FlagGems runs on sm_120: a single `flag_gems.ops.add` produced correct results on
  `(1,4096)`, `(16,4096)`, `(4096,4096)` and `(32768,4096)`, all `allclose`.
- The second consecutive call segfaults, reproducibly, at the point of the first one.
- A hand-written Triton kernel does not, in either Triton version available here.
- #4968 removes the segfault but leaves a `TypeError` at launch.

**Not established**

- Whether the `TypeError` is FlagGems' or Triton's, or which version pair is "right".
- Whether the segfault also happens on the Triton version FlagGems targets.
- Whether any of this reproduces on another GPU.

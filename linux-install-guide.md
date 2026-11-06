# Running Edge0 on Ubuntu Linux - Complete Guide

This documents the exact working procedure used to run Edge0-8B on Ubuntu via the llama.cpp engine path. The Python/MLX framework is macOS-only, so this guide adapts the Windows llama.cpp patch band to Linux.

**Result:** `llama-server` serving Edge0-8B (GGUF + LoRA) at `http://127.0.0.1:8080` with an OpenAI-compatible API.

---

## 1. Prerequisites

```bash
sudo apt update
sudo apt install -y git python3 python3-venv cmake ninja-build g++ \
    libvulkan-dev glslc spirv-headers glslang-tools
```

Why each package:

| Package | Why |
|---|---|
| `cmake`, `ninja-build`, `g++` | Build llama.cpp |
| `libvulkan-dev` | Vulkan backend headers |
| `glslc` | Shader compiler — required by llama.cpp's Vulkan backend (`FindVulkan` fails without it) |
| `spirv-headers` | CMake config `SPIRV-HeadersConfig.cmake` required by `ggml-vulkan` |
| `glslang-tools` | Provides `glslangValidator`, a required Vulkan component |

If you have no GPU, you can skip the Vulkan packages and build CPU-only (noted below).

## 2. Clone the repo and set up Python

```bash
cd ~
git clone https://github.com/Edge0-AI/Edge0.git
cd Edge0/python
python3 -m venv .venv
.venv/bin/pip install huggingface_hub numpy
```

## 3. Download the model checkpoint

```bash
cd ~/Edge0/python
.venv/bin/hf download Edge0/Edge0-8B-A1B-preview --local-dir models/edge0-8b
```

**Modification vs. README:** the README says `huggingface-cli`, but `huggingface_hub` 2.x renamed the CLI to `hf`. Using `huggingface-cli` gives `No such file or directory`.

Note the download lands in `~/Edge0/python/models/edge0-8b` (relative to where you run it).

## 4. Clone pinned llama.cpp and create a build worktree

```bash
cd ~/Edge0
PIN=$(cut -d' ' -f1 vendor.llama.pin)   # 7ab4ee7baad2d920464cbacfad4f4b07cf111fd2
git clone --filter=blob:none https://github.com/ggml-org/llama.cpp vendor/llama.cpp
git -C vendor/llama.cpp checkout --detach $PIN
git -C vendor/llama.cpp worktree add --detach "$PWD/wt/lin" $PIN
git -C wt/lin reset --hard $PIN
```

## 5. Replay the edge0 patch bands

```bash
cd ~/Edge0
for f in "$PWD"/patches/llama.cpp/common/0*.patch "$PWD"/patches/llama.cpp/windows/0*.patch; do
    git -C wt/lin am --3way "$f"
done
git -C wt/lin rev-parse 'HEAD^{tree}'
# MUST print: df9d12dcb59aa3314f3356c2cdb777ee4ca4919c
```

**Modification:** patch paths must be **absolute** (`"$PWD"/...`). `git -C wt/lin am` resolves relative paths inside `wt/lin`, so relative paths fail with "could not open ... for reading".

**Why the windows band:** there is no Linux band. The 6 `common` patches wire in the edge0 model features (LoRA into main graph, router-input capture, `src/edge0/*.cc` glob, mem-budget/prefetch hooks, expert-base resolver). The 2 `windows` patches only add `--mem-budget-mb` and `--pool-mb` CLI flags — platform-neutral code.

The golden tree hash `df9d12dc...` verifies the replay is byte-exact.

If `git am` fails mid-band: `git -C wt/lin am --abort && git -C wt/lin reset --hard $PIN`, then retry.

## 6. Copy serve pieces and stub the Windows-only files

```bash
cd ~/Edge0
mkdir -p wt/lin/src/edge0
cp windows/serve/*.cc windows/serve/*.h wt/lin/src/edge0/
```

**Required modification:** `mem_budget.cc` and `prefetch.cc` are Windows-only — they `#include <windows.h>` and use `VirtualAlloc`, `CRITICAL_SECTION`, `PrefetchVirtualMemory` unconditionally. They cannot compile on Linux. However, the server patches still *call* `edge0_apply_mem_budget`, `edge0_pref_init`, and `edge0_pref_on_step`, so deleting them causes link errors. Solution: replace them with no-op stubs.

```bash
rm wt/lin/src/edge0/mem_budget.cc wt/lin/src/edge0/prefetch.cc

cat > wt/lin/src/edge0/edge0_stub.cc <<'EOF'
// Linux stubs for the Windows-only edge0 serve pieces.
// --mem-budget-mb / --pool-mb / E0_PREROUTER become no-ops; the core
// model path (LoRA wiring, -cmoe, router-input capture) is unaffected.
#include "mem_budget.h"
#include "prefetch.h"
#include <cstdio>

bool edge0_apply_mem_budget(int mb) {
    if (mb > 0)
        fprintf(stderr, "[mem-budget] stub: --mem-budget-mb=%d ignored on Linux\n", mb);
    return false;
}

bool edge0_pref_init(llama_model *, llama_context *, int pool_mb, const char *) {
    if (pool_mb > 0)
        fprintf(stderr, "[pref-trace] stub: --pool-mb=%d ignored on Linux\n", pool_mb);
    return false;
}

void edge0_pref_on_step(llama_context *, int64_t) {}
EOF
```

**What this loses:** `--mem-budget-mb` (Win32 working-set cap — no direct Linux equivalent; use cgroups/`systemd-run` instead), `--pool-mb` private-page L1 expert pool, and the `E0_PREROUTER` advisory prefetch. These are memory/perf features, not correctness — inference works without them.

**What stays:** LoRA wiring, `-cmoe` expert offload, router-input capture — all live in the patched llama.cpp core.

## 7. Build llama-server

```bash
cd ~/Edge0
cmake -S wt/lin -B wt/lin/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
ninja -C wt/lin/build llama-server
```

CPU-only alternative: drop `-DGGML_VULKAN=ON`.

**Gotcha:** if you add/remove files in `src/edge0/` after configuring, re-run the `cmake` command — the unity build's generated source list doesn't pick up glob changes on its own.

Binary: `wt/lin/build/bin/llama-server`

## 8. Convert the checkpoint to GGUF

```bash
cd ~/Edge0
python/.venv/bin/python windows/tools/convert_mlx_to_gguf.py \
    --dir python/models/edge0-8b --out models/edge0-8b-gguf
```

**Modification:** `--dir` must point at the actual download location (`python/models/edge0-8b`), not `models/edge0-8b`.

The converter does a bit-exact MLX int4 → GGUF Q4_1 repack (not a requant), converts the LoRA adapter, and runs self-check gates. Expected output:

```
[self-check] edge0-8b.gguf: tensors=526 kv=48 OK
[self-check] adapter tensors=294 OK
DONE 8b: gguf=4.66GiB lora=14.8MiB -> models/edge0-8b-gguf/manifest.json
```

## 9. Run the server

```bash
cd ~/Edge0
wt/lin/build/bin/llama-server \
    -m models/edge0-8b-gguf/edge0-8b.gguf \
    --lora models/edge0-8b-gguf/lora_edge0_8b-gguf.gguf \
    -ngl 99 -cmoe --ctx-size 8192 --flash-attn on \
    -t 8 --port 8080
```

Flag reference:

| Flag | Meaning |
|---|---|
| `-m` | Base GGUF model |
| `--lora` | Edge0 LoRA adapter (required — the model is designed around it) |
| `-ngl 99` | Offload layers to GPU (Vulkan). No-op without a GPU; drop it for CPU-only |
| `-cmoe` | CPU-expert offload for the MoE architecture (edge0 patch) |
| `--ctx-size 8192` | Matches the app's engine contract |
| `--flash-attn on` | Flash attention |
| `-t 8` | CPU threads — set to your physical core count (default was 4) |
| `--port 8080` | Listen port |

**Shell gotcha:** when copying multi-line commands, a trailing space after `\` breaks line continuation — each line then runs as a separate command (`-m: command not found`). Either remove trailing spaces or paste as one line.

## 10. Start / stop / use

**Start** (foreground): the command in step 9. Logs go to the terminal.

**Start** (background):

```bash
cd ~/Edge0
nohup wt/lin/build/bin/llama-server \
    -m models/edge0-8b-gguf/edge0-8b.gguf \
    --lora models/edge0-8b-gguf/lora_edge0_8b-gguf.gguf \
    -ngl 99 -cmoe --ctx-size 8192 --flash-attn on \
    -t 8 --port 8080 > llama-server.log 2>&1 &
echo $! > llama-server.pid
```

**Stop:**

```bash
kill $(cat llama-server.pid)   # or: pkill -f llama-server
```

**Use:**

- Web UI: `http://127.0.0.1:8080`
- OpenAI-compatible API: `http://127.0.0.1:8080/v1/chat/completions`

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":128}'
```

**Optional RAM guard** (replaces the stubbed `--mem-budget-mb`):

```bash
systemd-run --scope -p MemoryMax=10G wt/lin/build/bin/llama-server ...
```

## Summary of all modifications vs. the documented Windows path

| # | Change | Reason |
|---|---|---|
| 1 | `hf` instead of `huggingface-cli` | `huggingface_hub` 2.x renamed the CLI |
| 2 | Absolute paths in the `git am` loop | `git -C` resolves paths inside the worktree |
| 3 | Extra apt packages `glslc`, `spirv-headers`, `glslang-tools` | Vulkan backend build deps not in a default Ubuntu install |
| 4 | Deleted `mem_budget.cc` + `prefetch.cc`, added `edge0_stub.cc` | Both files are unconditionally Windows-only (`windows.h`, `VirtualAlloc`); stubs satisfy the symbols the server patches call |
| 5 | Re-run `cmake` after changing `src/edge0/` contents | Unity build doesn't regenerate its source list from glob changes |
| 6 | `--dir python/models/edge0-8b` for the converter | Model downloaded under `python/` |
| 7 | `-t 8` added | Default thread count (4) underuses the CPU |

## Known limitations

- `--mem-budget-mb`, `--pool-mb`, `E0_PREROUTER` are no-ops (stubbed).
- CPU-only speed on the reference run: ~2.4 tok/s prompt, ~4.7 tok/s generation at 4 threads. A Vulkan-capable GPU + drivers (`mesa-vulkan-drivers` for AMD/Intel) improves this substantially.
- This configuration (windows band on Linux) is untested upstream — if output is garbled or mixed-language, report it at `github.com/Edge0-AI/Edge0/issues`.

# GPU Profiling Blog Series — 8-Week Execution Plan

*Budget: $100–200 | Time: 10–15 hrs/week | Hardware: Dell XPS laptop + cloud GPU instances*

---

## Pre-Work Checklist (Complete Before Week 1)

Do these on your Dell XPS with no GPU cost:

- [ ] Create accounts on Vast.ai, RunPod, and Lambda Labs (all free to sign up)
- [ ] Add $50 credits to Vast.ai and $50 to RunPod as initial funding
- [ ] Install Docker Desktop on your XPS
- [ ] Create a GitHub repo for the project (all scripts, configs, Dockerfiles, results)
- [ ] Choose your reference inference workload: **vLLM serving a quantized Llama-3-8B-Instruct (AWQ 4-bit)** — this fits in T4 16GB VRAM comfortably and is representative of real production inference
- [ ] Download the model weights locally (or note the HuggingFace path for on-instance download)
- [ ] Set up an Obsidian or Notion notebook for capturing observations, screenshots, and raw notes as you go — this becomes your blog drafting workspace

---

## Week 1: Environment Build & Baseline (Local + First GPU Session)

**Goal:** Build a reproducible cloud environment and establish a performance baseline with no profiling tools active.

**Hours target:** 12 hrs (8 local, 4 on GPU)

### Milestone 1.1 — Build the Master Docker Image (Local, ~4 hrs)

Build a single Docker image that contains everything you'll need across all 8 weeks. This avoids reinstalling tools on every GPU session.

- [ ] Create a Dockerfile based on `nvcr.io/nvidia/pytorch:24.01-py3` (includes CUDA toolkit, PyTorch, Nsight CLI tools)
- [ ] Add to image: vLLM, GPUprobe dependencies (Rust toolchain, bpftool, libbpf-dev), Prometheus, Grafana, dcgm-exporter, python benchmarking scripts
- [ ] Add to image: `stress-ng`, `htop`, `perf`, `bcc-tools`, `bpftrace` for OS-level profiling
- [ ] Add a `scripts/` directory with placeholder benchmark scripts
- [ ] Build and test the image locally (it will fail GPU operations but should build cleanly)
- [ ] Push to Docker Hub or GitHub Container Registry

### Milestone 1.2 — Write the Benchmark Harness (Local, ~4 hrs)

Create a standardized benchmarking script that you'll use consistently across all weeks:

- [ ] `benchmark/run_inference.py` — Sends N requests to vLLM's OpenAI-compatible API endpoint with configurable concurrency (1, 4, 8, 16 concurrent requests), prompt lengths (128, 512, 1024 tokens), and generation lengths (64, 256 tokens)
- [ ] `benchmark/collect_baseline.sh` — Runs the benchmark at each concurrency level, captures timestamps, throughput (tokens/sec), and latency percentiles (p50, p95, p99)
- [ ] `benchmark/results_to_csv.py` — Parses raw output into a standardized CSV format for later comparison
- [ ] Test the harness locally against a mock HTTP endpoint to verify the measurement logic

### Milestone 1.3 — First GPU Session: Baseline Run (Cloud, ~4 hrs)

- [ ] Spin up a T4 instance on Vast.ai (community cloud, ~$0.12/hr)
- [ ] Pull your Docker image, launch vLLM with the quantized Llama-3-8B
- [ ] Run the full benchmark harness with NO profiling tools active
- [ ] Record: tokens/sec at each concurrency level, p50/p95/p99 latency, GPU utilization via nvidia-smi, VRAM usage, power draw
- [ ] Save all results to your GitHub repo
- [ ] **This is your ground truth** — all subsequent profiling measurements will be compared to this baseline to quantify profiling overhead

### Week 1 Deliverable

A `results/week1_baseline/` directory containing: raw benchmark CSVs, a `baseline_summary.md` with key numbers, the Dockerfile, and all benchmark scripts. You should be able to answer: "My reference workload does X tokens/sec at concurrency Y with p99 latency of Z ms, with no profiling active."

**Estimated GPU cost:** ~$0.50 (4 hrs × $0.12/hr)

---

## Week 2: nvidia-smi & DCGM — The Production Monitoring Layer

**Goal:** Instrument the baseline workload with production-grade monitoring, build a Grafana dashboard, and quantify the overhead of always-on monitoring.

**Hours target:** 12 hrs (4 local, 8 on GPU)

### Milestone 2.1 — nvidia-smi Deep Dive (Cloud, ~3 hrs)

- [ ] Run `nvidia-smi dmon -s pucvmet -d 1` during the full benchmark suite and capture output to a file
- [ ] Write `scripts/parse_smi_dmon.py` to parse dmon output into time-series CSV (timestamp, gpu_util, mem_util, temp, power, sm_clock, mem_clock)
- [ ] Run the benchmark at concurrency=8 three times: (1) no nvidia-smi, (2) nvidia-smi dmon at 1s interval, (3) nvidia-smi dmon at 100ms interval
- [ ] Compare throughput and latency to quantify nvidia-smi polling overhead (expect: negligible)
- [ ] Capture the `nvidia-smi -q` full query output for your T4 — document all available fields
- [ ] Experiment with `nvidia-smi pmon` (process monitoring) to correlate per-process GPU usage

### Milestone 2.2 — DCGM Exporter + Prometheus + Grafana Stack (Cloud, ~5 hrs)

- [ ] Launch the DCGM Exporter container alongside your vLLM container: `docker run --gpus all --cap-add SYS_ADMIN -p 9400:9400 nvcr.io/nvidia/k8s/dcgm-exporter:3.3.8-3.6.0-ubuntu22.04`
- [ ] Verify metrics endpoint: `curl localhost:9400/metrics | grep DCGM_FI_DEV_GPU_UTIL`
- [ ] Launch Prometheus (containerized) configured to scrape DCGM Exporter at 9400 and vLLM's metrics endpoint
- [ ] Launch Grafana (containerized) and import the NVIDIA DCGM dashboard (ID: 12239)
- [ ] Create a custom Grafana dashboard panel showing: GPU utilization, VRAM usage, power draw, and SM clock over time, correlated with your benchmark's request rate
- [ ] Run the full benchmark suite with DCGM active and compare throughput/latency to Week 1 baseline
- [ ] Document which DCGM fields are available on T4 vs. which require A100/H100 (the `PROF` fields)
- [ ] Export Grafana dashboard JSON to your repo for reproducibility

### Milestone 2.3 — Write the "Production Monitoring" Blog Section Draft (Local, ~4 hrs)

- [ ] Draft the nvidia-smi section: what it shows, what it misses, overhead numbers from your experiments
- [ ] Draft the DCGM section: setup walkthrough, Grafana screenshots, the "gap" (high utilization but no idea which requests are slow)
- [ ] Include a side-by-side comparison table: nvidia-smi fields vs. DCGM fields
- [ ] Save as `drafts/02_production_monitoring.md`

### Week 2 Deliverable

`results/week2_monitoring/` with: nvidia-smi dmon parsed CSVs, DCGM Grafana dashboard screenshots and JSON export, overhead comparison spreadsheet, and the first blog section draft.

**Estimated GPU cost:** ~$1.00 (8 hrs × $0.12/hr)

---

## Week 3: PyTorch Profiler — The ML Engineer's View

**Goal:** Profile the inference workload with PyTorch Profiler, generate Chrome/Perfetto traces, and quantify the overhead penalty.

**Hours target:** 12 hrs (4 local, 8 on GPU)

### Milestone 3.1 — Basic PyTorch Profiler Integration (Cloud, ~4 hrs)

- [ ] Write `scripts/profile_pytorch.py` that wraps the vLLM inference loop with `torch.profiler.profile()` using a schedule (wait=2, warmup=1, active=3, repeat=2)
- [ ] Run with minimal settings first: `activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]` only
- [ ] Export Chrome trace: `prof.export_chrome_trace("trace_basic.json")`
- [ ] Open in Perfetto UI (https://ui.perfetto.dev/) — screenshot the timeline view
- [ ] Run `prof.key_averages().table(sort_by="cuda_time_total")` — capture the top-20 operators table
- [ ] Identify the hot CUDA kernels (these become your Nsight Compute targets in Week 5)

### Milestone 3.2 — Progressive Overhead Measurement (Cloud, ~3 hrs)

Run the benchmark at concurrency=8 under four configurations, comparing each to your Week 1 baseline:

- [ ] Config A: `profile(activities=[CPU, CUDA])` — basic tracing
- [ ] Config B: Config A + `record_shapes=True`
- [ ] Config C: Config B + `with_stack=True`
- [ ] Config D: Config C + `profile_memory=True`
- [ ] For each config, record: throughput (tokens/sec), p50/p95/p99 latency, trace file size
- [ ] Build a table showing progressive overhead: "each additional feature costs X% throughput"

### Milestone 3.3 — TensorBoard Integration & Analysis (Local, ~3 hrs)

- [ ] Copy trace files to your XPS
- [ ] Launch TensorBoard with PyTorch Profiler plugin: `tensorboard --logdir ./traces`
- [ ] Screenshot and annotate each TensorBoard view: Overview, Operator View, Kernel View, Trace View, Memory View
- [ ] Identify key insights: what does PyTorch Profiler reveal that nvidia-smi/DCGM cannot? What does it miss?

### Milestone 3.4 — Blog Section Draft (Local, ~2 hrs)

- [ ] Draft the PyTorch Profiler section with annotated screenshots
- [ ] Include the overhead comparison table
- [ ] Explain the "scheduled profiling" pattern and why Meta built Dynolog for on-demand triggering
- [ ] Save as `drafts/03_pytorch_profiler.md`

### Week 3 Deliverable

`results/week3_pytorch_profiler/` with: exported Chrome traces (.json), TensorBoard screenshots, overhead comparison CSV, operator rankings, and blog draft.

**Estimated GPU cost:** ~$1.00 (8 hrs × $0.12/hr)

---

## Week 4: Nsight Systems — The System-Wide Timeline

**Goal:** Capture and analyze Nsight Systems traces of the inference workload, compare with PyTorch Profiler results, and understand CPU-GPU interaction patterns.

**Hours target:** 12 hrs (3 local, 9 on GPU)

### Milestone 4.1 — Nsight Systems CLI Profiling (Cloud, ~4 hrs)

- [ ] Run a basic Nsight Systems capture: `nsys profile --stats=true --output=inference_nsys -t cuda,osrt,nvtx python run_inference.py`
- [ ] Experiment with trace flags: `--cuda-memory-usage=true`, `--cudabacktrace=all`, `--gpuctxsw=true`
- [ ] Capture a short (30-second) inference run under load (concurrency=8)
- [ ] Export the `.nsys-rep` file and the `--stats=true` summary output
- [ ] Run `nsys stats inference_nsys.nsys-rep` to generate the summary tables (CUDA API stats, GPU kernel stats, memory operation stats)

### Milestone 4.2 — Timeline Analysis (Cloud + Local, ~4 hrs)

- [ ] Open the `.nsys-rep` in Nsight Systems GUI (on your XPS — the GUI works without a GPU for analysis)
- [ ] If GUI isn't available, use `nsys export --type=json` to export and analyze programmatically
- [ ] Identify and screenshot: CPU-GPU overlap (or lack thereof), kernel launch gaps (CPU bottleneck indicators), memory copy operations (H2D, D2H), CUDA synchronization points
- [ ] Compare the kernel execution timeline with PyTorch Profiler's trace for the same workload
- [ ] Document what Nsight Systems reveals that PyTorch Profiler doesn't: OS-level context, driver overhead, precise kernel launch timing

### Milestone 4.3 — Overhead Quantification (Cloud, ~2 hrs)

- [ ] Run the benchmark at concurrency=8 under Nsight Systems profiling
- [ ] Compare throughput and latency to Week 1 baseline — quantify the slowdown
- [ ] Test with varying trace detail levels (minimal flags vs. full flags)
- [ ] Attempt a 5-minute capture and document the resulting trace file size and analysis experience (loading time, usability)
- [ ] Document the practical session length limits

### Milestone 4.4 — Blog Section Draft (Local, ~2 hrs)

- [ ] Draft the Nsight Systems section with annotated timeline screenshots
- [ ] Include a direct comparison with PyTorch Profiler on the same workload: what each reveals, what each misses
- [ ] Emphasize the "Nsight Systems → identify hot kernels → Nsight Compute" workflow
- [ ] Save as `drafts/04_nsight_systems.md`

### Week 4 Deliverable

`results/week4_nsight_systems/` with: `.nsys-rep` files, exported stats, timeline screenshots with annotations, overhead measurements, PyTorch Profiler vs. Nsight Systems comparison table, and blog draft.

**Estimated GPU cost:** ~$1.10 (9 hrs × $0.12/hr)

---

## Week 5: Nsight Compute & CUPTI — The Deep Dive

**Goal:** Profile specific hot kernels with Nsight Compute, understand the kernel replay mechanism, and document CUPTI's role as the underlying API.

**Hours target:** 12 hrs (4 local, 8 on GPU)

### Milestone 5.1 — Nsight Compute on Hot Kernels (Cloud, ~4 hrs)

Using the hot kernels you identified in Weeks 3 and 4:

- [ ] Run Nsight Compute on a single-request inference with basic metrics: `ncu --set basic --target-processes all python run_single_inference.py`
- [ ] Run with the full section set on a specific kernel: `ncu --set full --kernel-name "regex_for_your_hot_kernel" --launch-skip 10 --launch-count 5 python run_single_inference.py`
- [ ] Capture the roofline analysis for your top-3 kernels
- [ ] Export the `.ncu-rep` report file
- [ ] Document: was the hot kernel compute-bound or memory-bound? What does the guided analysis say?

### Milestone 5.2 — Understand Kernel Replay (Cloud, ~2 hrs)

- [ ] Run the same kernel with `--replay-mode kernel` (default) and `--replay-mode application` and compare total profiling time
- [ ] Measure wall-clock time for profiling 1 kernel launch vs. 5 vs. 20 with `--set full`
- [ ] Build a table showing: "profiling N kernel launches with section set X takes Y minutes" — this quantifies the 20–200× overhead claim
- [ ] Attempt to run Nsight Compute during active inference serving (concurrency=4) and document what happens (it will be very slow — document *how* slow)

### Milestone 5.3 — CUPTI Exploration (Cloud, ~2 hrs)

- [ ] Write a minimal CUPTI callback example in C/Python that subscribes to `CUPTI_CB_DOMAIN_RUNTIME_API` and logs every CUDA API call during inference
- [ ] Alternatively, use CUPTI via PyTorch's Kineto integration and show the raw CUPTI activity records
- [ ] Document the `CUDA_INJECTION64_PATH` mechanism — show how a shared library can be injected into any CUDA process without code changes
- [ ] Explain CUPTI's role as the foundation layer for PyTorch Profiler, Nsight Systems, and the eBPF-based tools

### Milestone 5.4 — Blog Section Draft (Local, ~4 hrs)

- [ ] Draft the Nsight Compute section with roofline plots and guided analysis screenshots
- [ ] Draft the CUPTI section explaining it as the "API behind the tools"
- [ ] Include the overhead measurement table showing kernel replay costs
- [ ] Create a visual diagram showing the tool dependency chain: PyTorch Profiler → Kineto → CUPTI → GPU hardware counters
- [ ] Save as `drafts/05_nsight_compute_cupti.md`

### Week 5 Deliverable

`results/week5_nsight_compute/` with: `.ncu-rep` files, roofline screenshots, kernel replay timing table, CUPTI callback example code, tool dependency diagram, and blog draft.

**Estimated GPU cost:** ~$1.00 (8 hrs × $0.12/hr)

---

## Week 6: eBPF Tools — The Emerging Middle Ground

**Goal:** Deploy GPUprobe and attempt parca-agent/xpu-perf, demonstrating eBPF-based GPU observability with production-grade overhead.

**Hours target:** 14 hrs (4 local, 10 on GPU — this is the most technically challenging week)

### Milestone 6.1 — GPUprobe Deployment (Cloud, ~5 hrs)

- [ ] Build GPUprobe from source on your GPU instance (requires Rust toolchain, bpftool, vmlinux.h generation)
- [ ] Run `bpftool btf dump file /sys/kernel/btf/vmlinux format c > vmlinux.h` to generate BTF headers
- [ ] Start GPUprobe daemon with all probes: `./gpu_probe --memleak --cudatrace --bandwidth-util --metrics-addr 0.0.0.0:9000`
- [ ] Launch vLLM inference and verify GPUprobe is capturing events
- [ ] Hit the Prometheus endpoint: `curl localhost:9000/metrics` — document every metric exposed
- [ ] Add GPUprobe as a Prometheus scrape target alongside DCGM — show both in Grafana side by side
- [ ] Run the full benchmark at concurrency=8 with GPUprobe active and compare to Week 1 baseline — quantify overhead
- [ ] Create a deliberate CUDA memory leak (allocate without free in a test script) and verify GPUprobe detects it

### Milestone 6.2 — parca-agent GPU Profiling (Cloud, ~3 hrs)

- [ ] Install parca-agent v0.43.0+ and configure for GPU profiling
- [ ] Set up the CUPTI injection: `export CUDA_INJECTION64_PATH=/path/to/libparcagpucupti.so`
- [ ] Run vLLM under parca-agent and verify GPU kernel profiling data is being collected
- [ ] If Polar Signals Cloud is available (free tier), push profiles there for visualization; otherwise capture local output
- [ ] Compare parca-agent's view (CPU stack traces correlated with GPU kernel timing) to PyTorch Profiler's view
- [ ] Measure overhead with parca-agent active vs. baseline

### Milestone 6.3 — xpu-perf Flamegraph Attempt (Cloud, ~2 hrs)

- [ ] Clone and build xpu-perf from the eunomia-bpf repo
- [ ] Attempt the merge mode: `sudo ./xpu-perf merge -o merged.folded python run_inference.py`
- [ ] Generate a CPU→GPU flamegraph: `perl flamegraph.pl merged.folded > inference_flamegraph.svg`
- [ ] If this works, it's a compelling visualization — a single flamegraph showing the full Python → PyTorch → CUDA → GPU kernel call chain
- [ ] Note: this project is WIP, so document any build issues or failures — that's useful information for your readers too

### Milestone 6.4 — Blog Section Draft (Local, ~4 hrs)

- [ ] Draft the eBPF tools section — this is the centerpiece of your blog series
- [ ] Include the GPUprobe Grafana dashboard alongside the DCGM dashboard — show the "middle ground" visually
- [ ] If the flamegraph worked, feature it prominently
- [ ] Include a "lessons learned" subsection about building and running experimental eBPF tools on cloud GPU instances
- [ ] Discuss the fundamental architecture: eBPF uprobes on libcudart.so, BPF ring buffers, Prometheus export
- [ ] Save as `drafts/06_ebpf_tools.md`

### Week 6 Deliverable

`results/week6_ebpf/` with: GPUprobe metrics output and Grafana screenshots, parca-agent profiles, flamegraph SVG (if successful), build notes and troubleshooting log, overhead comparison data, and blog draft.

**Estimated GPU cost:** ~$1.50 (10 hrs × $0.15/hr — using RunPod for more reliable root/kernel access)

---

## Week 7: A100 Comparison Session & Research Frontier

**Goal:** Run a subset of experiments on an A100 to compare profiling fidelity, and cover the research frontier tools (eInfer, eGPU) for the "where things are going" narrative.

**Hours target:** 12 hrs (6 local, 6 on GPU)

### Milestone 7.1 — A100 Comparison Run (Cloud, ~4 hrs on Lambda Labs A100)

- [ ] Spin up an A100 40GB on Lambda Labs (~$1.29/hr)
- [ ] Run the same quantized Llama-3-8B workload (for fair comparison, use the same quantization and batch sizes)
- [ ] Run: nvidia-smi dmon, DCGM with PROF fields enabled, PyTorch Profiler (basic), Nsight Systems (30s), and one Nsight Compute kernel profile
- [ ] Document the DCGM PROF fields available on A100 that were missing on T4: `DCGM_FI_PROF_GR_ENGINE_ACTIVE`, `DCGM_FI_PROF_DRAM_ACTIVE`, `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`, etc.
- [ ] Compare Nsight Compute's roofline analysis: A100 vs. T4 for the same kernels
- [ ] Run GPUprobe on A100 and compare metrics to T4 session
- [ ] Build a "T4 vs. A100 profiling fidelity" comparison table showing what additional data you get on A100

### Milestone 7.2 — Research Frontier Survey (Local, ~4 hrs)

Read and summarize the key research papers — you won't implement these, but understanding them is essential for the blog's forward-looking section:

- [ ] Read the eInfer paper (ACM DL: 10.1145/3748355.3748372) — summarize the Cooperative Kernel Proxy architecture and the ~1% overhead claim
- [ ] Read the eGPU paper (ACM HCDS'25) — summarize the PTX injection mechanism for on-GPU eBPF
- [ ] Read the gpu_ext paper (arXiv:2512.12615) — summarize the "GPU driver as programmable OS subsystem" vision
- [ ] Read the ProfInfer paper (arXiv:2601.20755) — summarize the llama.cpp profiling results and edge-device applicability
- [ ] For each paper, note: key insight, maturity level, what it enables that current tools can't, and timeline to production readiness

### Milestone 7.3 — Blog Section Drafts (Local, ~4 hrs)

- [ ] Draft the "A100 vs. T4" comparison section with side-by-side data
- [ ] Draft the "Research Frontier" section covering eInfer, eGPU, gpu_ext, ProfInfer — frame this as "where the field is heading in 2026–2028"
- [ ] Include a timeline/maturity graphic showing each tool on a spectrum from "production-ready" to "research prototype"
- [ ] Save as `drafts/07_a100_comparison.md` and `drafts/08_research_frontier.md`

### Week 7 Deliverable

`results/week7_a100/` with: A100 benchmark results, DCGM PROF field data, Nsight Compute roofline comparison, T4 vs. A100 comparison table, research paper summaries, and two blog section drafts.

**Estimated GPU cost:** ~$5.50 (4 hrs × $1.29/hr on Lambda A100 + $0.36 for any T4 re-runs)

---

## Week 8: Synthesis, Visualization & Publication

**Goal:** Assemble all drafts into a polished blog post (or multi-part series), create compelling visualizations, and publish.

**Hours target:** 14 hrs (all local — no GPU needed)

### Milestone 8.1 — Create the Master Comparison Artifacts (Local, ~4 hrs)

- [ ] Build the final comparison matrix table from the survey document, updated with your empirical overhead numbers from Weeks 2–7
- [ ] Create a "tool selection decision flowchart": Are you in production? → DCGM. Debugging a specific kernel? → Nsight Compute. Need always-on app-level insight? → eBPF tools. Etc.
- [ ] Create a "stack layer diagram" showing all tools mapped to the layers they operate on (this is the single most important visual in the entire post)
- [ ] Create an "overhead vs. insight depth" scatter plot with all tools positioned on it
- [ ] Build a "cost of observability" table: tool → overhead % → equivalent $ cost per GPU-hour in lost throughput at current cloud rates

### Milestone 8.2 — Assemble the Blog Post(s) (Local, ~6 hrs)

Decide on format — either a single long-form post (~5000 words) or a 3-part series:

Option A (Single Post):
- [ ] Part 1: The problem & the stack model (from survey doc section 1–2)
- [ ] Part 2: The established tools (nvidia-smi, DCGM, PyTorch Profiler, Nsight Systems/Compute, CUPTI) with your empirical data
- [ ] Part 3: The eBPF frontier (GPUprobe, parca-agent, xpu-perf, eInfer, eGPU) with your hands-on experience
- [ ] Part 4: The decision framework and what's next

Option B (3-Part Series — recommended for Substack engagement):
- [ ] Post 1: "The GPU Observability Gap" — problem statement, stack model, nvidia-smi/DCGM (publish Week 8)
- [ ] Post 2: "Profiling Deep Dives" — PyTorch Profiler, Nsight Systems/Compute, CUPTI (publish Week 9)
- [ ] Post 3: "The eBPF Revolution" — GPUprobe, parca-agent, research frontier (publish Week 10)

Regardless of format:
- [ ] Write the introduction: the tension between observability depth and production safety
- [ ] Write the conclusion: your personal recommendation stack for different scenarios
- [ ] Add all empirical data tables, screenshots, and diagrams
- [ ] Include a "reproduce this yourself" section with links to your GitHub repo

### Milestone 8.3 — Review, Polish & Publish (Local, ~4 hrs)

- [ ] Proofread the entire post(s) for technical accuracy
- [ ] Ensure all screenshots have proper captions and context
- [ ] Verify all overhead numbers are consistent across sections
- [ ] Add proper attributions for research papers and open-source projects
- [ ] Write a compelling Substack title and subtitle (suggested: "The Complete Guide to GPU Profiling for AI Inference — From nvidia-smi to eBPF on the GPU")
- [ ] Publish Part 1 (or the single post)
- [ ] Share on LinkedIn, Twitter/X, HackerNews — the eBPF angle is likely to get traction on HN

### Week 8 Deliverable

Published blog post(s) on Substack, with a companion GitHub repo containing all code, Docker configs, benchmark scripts, and raw results for reproducibility.

**Estimated GPU cost:** $0 (all local work)

---

## Budget Summary

| Week | Activity | GPU Hours | Instance Type | Cost |
|---|---|---|---|---|
| 1 | Baseline | 4 hrs | T4 @ Vast.ai | $0.50 |
| 2 | nvidia-smi + DCGM | 8 hrs | T4 @ Vast.ai | $1.00 |
| 3 | PyTorch Profiler | 8 hrs | T4 @ Vast.ai | $1.00 |
| 4 | Nsight Systems | 9 hrs | T4 @ Vast.ai | $1.10 |
| 5 | Nsight Compute + CUPTI | 8 hrs | T4 @ Vast.ai | $1.00 |
| 6 | eBPF tools | 10 hrs | T4 @ RunPod | $1.50 |
| 7 | A100 comparison | 4 hrs + 2 hrs | A100 @ Lambda + T4 | $5.50 |
| 8 | Writing (local only) | 0 hrs | — | $0.00 |
| | **Buffer for re-runs, failures, extended sessions** | ~10 hrs | Mixed | ~$5.00 |
| | **Total** | **~63 hrs GPU** | | **~$16.60** |

Your total GPU cost is approximately **$15–25**, leaving $75–185 of your budget as buffer. Even if you double your GPU time due to debugging and re-runs, you'll stay well under $50.

---

## Risk Mitigation

**Risk: Vast.ai instance gets terminated mid-experiment (spot/interruptible)**
- Mitigation: Script your experiments end-to-end so they can be re-run. Save results to a mounted volume or sync to cloud storage (S3, GCS) periodically. Use RunPod's "on-demand" tier for the most critical experiments (Weeks 5–6).

**Risk: GPUprobe or xpu-perf fails to build on the cloud instance**
- Mitigation: Dedicate the first 1–2 hours of Week 6 purely to building and debugging. Document failures — a "what went wrong" section is valuable blog content. Have a fallback plan: if GPUprobe won't build, focus on parca-agent (which is more mature).

**Risk: DCGM doesn't support T4 for all fields**
- Mitigation: Already accounted for — the A100 session in Week 7 covers the richer PROF fields. T4 still supports the basic DCGM fields (utilization, temp, power, memory). Document the gap explicitly — it's useful information for readers.

**Risk: Nsight Systems GUI won't install on Dell XPS (Linux/Mac issues)**
- Mitigation: Use `nsys stats` and `nsys export --type=json` for CLI-based analysis. Alternatively, install the GUI on a Windows/Mac system or use NVIDIA's remote analysis workflow. The CLI output is sufficient for the blog.

**Risk: Running behind schedule**
- Mitigation: Weeks 4 and 5 (Nsight Systems and Nsight Compute) can be compressed into a single week if needed — the tools are well-documented and predictable. Week 7's A100 session is "nice to have" and can be cut entirely if budget or time is tight; the T4 data alone tells a complete story.

**Risk: vLLM quantized model doesn't fit on T4**
- Mitigation: AWQ 4-bit Llama-3-8B requires ~5–6GB VRAM; T4 has 16GB. This should work with room to spare. If it doesn't, fall back to a smaller model (Llama-3.2-3B or Phi-3-mini) or use llama.cpp with GGUF quantization instead of vLLM.

---

## Key Files & Repository Structure

```
gpu-profiling-survey/
├── README.md                        # Project overview and reproduction instructions
├── Dockerfile                       # Master image with all tools
├── docker-compose.yml               # Full stack: vLLM + DCGM + Prometheus + Grafana + GPUprobe
├── benchmark/
│   ├── run_inference.py             # Configurable load generator
│   ├── collect_baseline.sh          # Full benchmark orchestration
│   └── results_to_csv.py            # Output parser
├── scripts/
│   ├── parse_smi_dmon.py            # nvidia-smi dmon → CSV
│   ├── profile_pytorch.py           # PyTorch Profiler wrapper
│   ├── cupti_callback_example.py    # Minimal CUPTI demo
│   └── setup_gpuprobe.sh            # GPUprobe build + run automation
├── configs/
│   ├── prometheus.yml               # Prometheus scrape config
│   ├── grafana_dcgm_dashboard.json  # Exported Grafana dashboard
│   └── dcgm_custom_fields.csv       # Custom DCGM field list
├── results/
│   ├── week1_baseline/
│   ├── week2_monitoring/
│   ├── week3_pytorch_profiler/
│   ├── week4_nsight_systems/
│   ├── week5_nsight_compute/
│   ├── week6_ebpf/
│   └── week7_a100/
├── drafts/
│   ├── 01_introduction.md
│   ├── 02_production_monitoring.md
│   ├── 03_pytorch_profiler.md
│   ├── 04_nsight_systems.md
│   ├── 05_nsight_compute_cupti.md
│   ├── 06_ebpf_tools.md
│   ├── 07_a100_comparison.md
│   └── 08_research_frontier.md
└── figures/
    ├── stack_layer_diagram.svg
    ├── overhead_vs_insight.svg
    ├── tool_selection_flowchart.svg
    └── ... (screenshots, flamegraphs, etc.)
```

---

## Success Criteria

By Week 8, you should be able to answer these questions with empirical data:

1. What is the overhead of always-on GPU monitoring (nvidia-smi, DCGM) in terms of throughput loss?
2. How much does PyTorch Profiler slow down inference, and how does each feature flag contribute?
3. What does Nsight Systems reveal that PyTorch Profiler cannot, and at what cost?
4. For a specific hot kernel, is it compute-bound or memory-bound? (Nsight Compute answer)
5. Can GPUprobe detect a CUDA memory leak in a running inference server with negligible overhead?
6. What does an eBPF-based GPU flamegraph look like for an LLM inference request?
7. What additional GPU metrics become available on A100 vs. T4?
8. What are the next 2–3 years likely to bring in GPU observability tooling?

If your blog post answers all eight questions with your own hands-on data, it will be one of the most comprehensive GPU profiling guides available.

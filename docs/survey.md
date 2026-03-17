# The Definitive Guide to GPU Profiling & Monitoring for AI Inference Observability

*A practitioner's survey of every major GPU profiling approach — from NVIDIA's first-party tools to the emerging eBPF frontier — with overhead characteristics, production readiness, and hands-on testing guidance for the solo developer.*

---

## 1. Why This Survey Exists

AI inference is eating the world, but observability hasn't kept up. If you're running an LLM inference server in production, you face a fundamental tension: the tools that give you the deepest GPU insights (Nsight Compute) will slow your workload by 20–200×, while the tools safe for production (nvidia-smi, DCGM) can't tell you *why* a particular request is slow. A new generation of eBPF-based tools is attempting to bridge this gap, but they're in various stages of maturity.

This survey maps the entire landscape so you can make informed decisions about what to use, when, and why.

---

## 2. The Observability Stack Model

Every GPU profiling tool operates at one or more layers of the stack. Understanding which layer a tool targets is the single most important factor in predicting what it can see, what it costs in overhead, and whether it belongs in production.

| Layer | What Lives Here | Example Signals |
|---|---|---|
| **Application** | PyTorch, TensorFlow, vLLM, TensorRT-LLM | Operator timings, batch sizes, model-level latency |
| **Framework Runtime** | CUDA Runtime API (libcudart.so) | cudaMalloc, cudaLaunchKernel, cudaMemcpy calls |
| **Driver** | CUDA Driver API, kernel module (nvidia.ko) | Context switches, memory management, command submission |
| **Device Hardware** | SM units, HBM, NVLink, PCIe | HW perf counters, warp occupancy, memory bandwidth, thermal |
| **OS/Kernel** | Linux scheduler, eBPF subsystem, PCIe subsystem | Syscalls, interrupts, scheduling latency, page faults |

---

## 3. Tool-by-Tool Analysis

### 3.1 nvidia-smi

**Layer:** Device Hardware (via NVML library, which queries the GPU's onboard management controller)

**What it can see:** GPU utilization percentage, memory usage (allocated vs. total), temperature, power draw, clock speeds, running processes and their memory footprint, PCIe throughput, ECC error counts, and driver/CUDA version information.

**What it cannot see:** Anything about *what* the GPU is doing — no kernel names, no operator timings, no memory allocation patterns, no per-request breakdown. The utilization percentage is a coarse time-sampled metric (typically at 1-second or longer intervals) that tells you the GPU was "busy" but not how efficiently busy.

**Overhead:** Effectively zero for occasional queries. Polling with `nvidia-smi dmon` at 1-second intervals introduces negligible load. The underlying NVML library is designed for continuous monitoring.

**Production readiness:** Excellent — this is *the* baseline tool that every GPU operator uses. Pre-installed with the NVIDIA driver on every system. Available in containers, VMs, and bare metal.

**Deployment requirements:** NVIDIA GPU driver installed. No special privileges beyond read access to `/dev/nvidia*` devices. Works on consumer and data center GPUs alike.

**Best for:** Quick sanity checks, capacity monitoring dashboards, detecting gross problems (GPU idle, OOM, thermal throttling). It's your "is the GPU even doing anything?" tool.

---

### 3.2 NVIDIA DCGM (Data Center GPU Manager)

**Layer:** Device Hardware + Driver (via NVML and extended telemetry APIs, including profiling-grade metrics on supported GPUs)

**What it can see:** Everything nvidia-smi sees, plus richer profiling metrics on data-center GPUs (Volta+): SM occupancy, tensor core utilization, NVLink bandwidth, memory bandwidth utilization, PCIe bandwidth, XID errors, GPU health diagnostics, and power governance controls. DCGM can tag metrics with job/container metadata for multi-tenant attribution. On newer GPUs (A100, H100), it exposes profiling-grade counters like `DCGM_FI_PROF_GR_ENGINE_ACTIVE` and `DCGM_FI_PROF_DRAM_ACTIVE` that are more accurate than nvidia-smi's utilization percentage.

**What it cannot see:** Application-level semantics. DCGM knows the GPU is 80% utilized but not *which model* or *which request* is responsible. No kernel names, no operator-level breakdown, no CUDA API call tracing. It operates below the application/runtime boundary.

**Overhead:** Approximately 5% or less, depending on the polling frequency and number of fields collected. NVIDIA explicitly markets it as low-overhead and suitable for always-on monitoring in production. The profiling metrics (PROF fields) use hardware sampling and add minimal CPU overhead.

**Production readiness:** Purpose-built for production. Runs as a daemon (nv-hostengine) or as the DCGM Exporter container that exposes Prometheus-compatible metrics on port 9400. Integrates natively with Kubernetes (via the NVIDIA GPU Operator), Prometheus, Grafana, and Datadog. Pre-built Grafana dashboards are available.

**Deployment requirements:** Data center GPUs (Tesla/A-series/H-series). Does not support consumer GeForce GPUs. Requires `SYS_ADMIN` capability for the container. Linux only (x86_64, ARM, POWER).

**Best for:** Production GPU fleet monitoring, multi-tenant utilization tracking, health monitoring and alerting, capacity planning. This is the "are my GPUs earning their keep?" tool.

---

### 3.3 NVIDIA Nsight Systems

**Layer:** Application + Framework Runtime + Driver + Device (system-wide timeline profiler)

**What it can see:** A unified timeline view showing CPU activity (threads, OS runtime, Python/C++ code), CUDA API calls (kernel launches, memory copies, synchronizations), GPU kernel execution timelines, NVTX annotations, memory allocations, NCCL communication, NVLink/PCIe data transfers, and CPU-GPU synchronization points. It answers the question "where is time being spent across my entire application?"

**What it cannot see:** Deep hardware performance counters *within* a kernel. Nsight Systems tells you a kernel ran for 500μs but not whether it was compute-bound or memory-bound internally. For that, you need Nsight Compute. It also does not see inside Python-level logic (though it captures the CUDA calls that Python triggers).

**Overhead:** Moderate during profiling — roughly 2–10× slowdown depending on the workload and trace detail level. The tool instruments the CUDA runtime and driver layers. NVIDIA does not officially support profiling sessions longer than about 5 minutes, as trace files grow large (multi-GB) and become unwieldy to analyze.

**Production readiness:** Not suitable for production. Designed for development-time investigation — profile a representative workload, analyze the trace, iterate. The overhead and session-length limitations make it impractical for always-on use.

**Deployment requirements:** Nsight Systems CLI (`nsys`) is included in the CUDA Toolkit and in NVIDIA's NGC containers. Requires CUDA 10.1+ and Volta or later GPUs for full feature set. The GUI runs on Windows, Linux, or macOS (for remote analysis). Root/elevated privileges may be needed for certain OS-level tracing features.

**Best for:** Initial performance investigation — "why is my inference pipeline slow?" It's the starting point in the recommended NVIDIA profiling workflow: Nsight Systems first (broad view), then Nsight Compute (deep dive on specific kernels).

---

### 3.4 NVIDIA Nsight Compute

**Layer:** Device Hardware (kernel-level micro-architecture profiler)

**What it can see:** Extremely detailed per-kernel metrics: SM throughput, memory throughput, warp occupancy, instruction mix, cache hit rates, register usage, shared memory usage, instruction-level source correlation, roofline analysis, and hundreds of hardware performance counters. It provides guided analysis with NVIDIA-authored rules that diagnose common issues (memory-bound, compute-bound, latency-bound, occupancy limiters).

**What it cannot see:** The broader application context. It profiles individual kernel launches in isolation. It cannot tell you about CPU-GPU overlap, data loading bottlenecks, framework overhead, or end-to-end request latency. It also doesn't see inter-kernel relationships well.

**Overhead:** Extreme — 20–200× slowdown. Nsight Compute uses "kernel replay" to collect hardware counters: it re-executes each profiled kernel multiple times (once per pass of counter collection), saving and restoring GPU memory state between replays. The `--set full` option requires the most passes and produces the highest overhead. Application replay mode duplicates the entire host-side execution as well.

**Production readiness:** Absolutely not. This is a lab/development tool only. The kernel serialization and replay mechanism fundamentally alters execution behavior. It's designed for analyzing specific kernels you've already identified as bottlenecks.

**Deployment requirements:** Same as Nsight Systems — included in the CUDA Toolkit. CLI tool is `ncu`. Requires Volta+ GPUs. The tool intercepts the CUDA user-mode driver, so it needs appropriate privileges.

**Best for:** Deep kernel optimization — "this matmul kernel is my bottleneck, is it compute-bound or memory-bound, and what specific hardware resources are saturated?" This is where you go after Nsight Systems has pointed you to the hot kernels.

---

### 3.5 CUPTI (CUDA Profiling Tools Interface)

**Layer:** Framework Runtime + Driver (NVIDIA's profiling API layer)

**What it can see:** CUPTI is not a tool but the *API* that most other tools build on. It provides callbacks for CUDA runtime/driver API calls, activity records for kernel execution and memory transfers, hardware performance counter access, and PC sampling. PyTorch Profiler, Nsight Systems, Nsight Compute, and many third-party tools use CUPTI under the hood.

**What it cannot see:** CUPTI operates from the CPU side — it intercepts host-side API calls and can query device counters, but it cannot instrument code *running on the GPU*. It reports "what happened and when" but relies on hardware counters for "why."

**Overhead:** Varies dramatically by configuration. Subscribing to API callbacks alone is relatively cheap (single-digit percentage overhead). Enabling activity tracing adds more. Collecting hardware performance counters can require kernel replay (like Nsight Compute) and becomes very expensive. The Polar Signals team found that combining CUPTI activity tracing with USDT probes and eBPF achieves near-negligible overhead when profiling is disabled (the USDT probes compile to NOP instructions).

**Production readiness:** CUPTI itself is a library, not a standalone production tool. However, it's the foundation for production-grade integrations. The key innovation is the `CUDA_INJECTION64_PATH` mechanism, which allows injecting a CUPTI-based library into any CUDA application without code changes — this is how Polar Signals' parca-agent achieves "zero instrumentation" GPU profiling.

**Deployment requirements:** Included with the CUDA Toolkit. Available on any system with CUDA. Requires that the profiled application links dynamically against libcudart.so (statically linked binaries cannot be intercepted via uprobes or injection).

**Best for:** Building custom profiling solutions, integrating GPU metrics into existing observability stacks, and serving as the bridge layer between GPU hardware counters and higher-level analysis tools.

---

### 3.6 PyTorch Profiler

**Layer:** Application + Framework Runtime (with GPU visibility via CUPTI/Kineto)

**What it can see:** PyTorch operator execution times (both CPU and CUDA), operator call stacks (with `with_stack=True`), tensor memory allocation patterns (with `profile_memory=True`), FLOPS estimates for supported ops, CUDA kernel launch timelines, and module hierarchy. Results export to Chrome/Perfetto trace format or TensorBoard for interactive visualization. Meta's Holistic Trace Analysis (HTA) library can process traces from distributed training to identify temporal breakdowns, communication bottlenecks, and idle time across ranks.

**What it cannot see:** Anything outside the PyTorch runtime. No OS-level context (scheduling, I/O), no visibility into non-PyTorch CUDA activity, no hardware performance counters beyond what CUPTI surfaces via Kineto. It also cannot see inside compiled Triton kernels at the hardware level.

**Overhead:** Moderate to high. Basic CPU+CUDA activity profiling adds noticeable overhead. Enabling `record_shapes=True` temporarily holds tensor references (which can interfere with memory optimizations). Enabling `with_stack=True` adds further overhead from capturing Python call stacks. The profiler supports a schedule API (wait/warmup/active cycles) specifically to limit the window of active tracing and reduce overall impact. The eInfer research paper notes that PyTorch Profiler can increase latency by "orders of magnitude" in production scenarios.

**Production readiness:** Limited. The recommended pattern is scheduled profiling — capture a few iterations, then stop. Meta uses Dynolog (a telemetry daemon) to trigger on-demand PyTorch profiling across large fleets via the `KINETO_USE_DAEMON=True` environment variable, allowing remote trace collection without code changes. But this is burst-mode profiling, not continuous monitoring.

**Deployment requirements:** PyTorch installation with CUDA support. For GPU tracing, Kineto must be built with CUPTI support (this is the default for PyTorch CUDA builds). TensorBoard with the PyTorch Profiler plugin for visualization. Perfetto UI (free, web-based) works for viewing exported traces.

**Best for:** Framework-level performance analysis during development — "which PyTorch operators are slowest?", "where is the CPU-GPU synchronization overhead?", "is my data pipeline a bottleneck?" It's the natural first tool for ML engineers who work in PyTorch.

---

### 3.7 GPUprobe (eBPF)

**Layer:** OS/Kernel + Framework Runtime boundary (eBPF uprobes on libcudart.so)

**What it can see:** CUDA runtime API calls intercepted via eBPF uprobes — specifically memory allocation patterns (cudaMalloc/cudaFree), CUDA memory leaks (allocation/deallocation mismatches tracked per-process), kernel launch frequencies and patterns (via cudaLaunchKernel), and approximate PCIe bandwidth utilization (via cudaMemcpy interception). All metrics are exposed as Prometheus-compatible endpoints for integration with Grafana.

**What it cannot see:** Anything happening *inside* the GPU. GPUprobe operates at the CUDA runtime API boundary — it sees the function calls the CPU makes into libcudart.so but cannot observe device-side execution, hardware counters, warp-level behavior, or kernel-internal performance characteristics. It also cannot see CUDA driver API calls (only runtime API).

**Overhead:** Very low — by design. eBPF uprobes fire only on the instrumented function calls (cudaMalloc, cudaLaunchKernel, etc.) and execute a small, verified BPF program in kernel space. The data is written to BPF ring buffers and consumed asynchronously. The tool's author positions it explicitly in the "middle ground between Nsight's deep but heavyweight profiling and DCGM's high-level system monitoring."

**Production readiness:** Experimental but promising. The daemon architecture (Prometheus metrics endpoint, configurable polling interval) is designed for production use. However, the project is still relatively young and the feature set is limited compared to mature tools. It requires root/CAP_BPF privileges and an eBPF-compatible kernel (5.x+).

**Deployment requirements:** Linux with eBPF support (kernel 5.x+). `bpftool` for generating vmlinux.h. The daemon is written in Rust (using libbpf-rs). Requires that the CUDA application dynamically links against libcudart.so. NVIDIA GPU with CUDA runtime installed on the host.

**Best for:** Lightweight, continuous monitoring of CUDA API-level patterns in production — especially memory leak detection and kernel launch frequency tracking. Good for the "is something wrong?" layer of observability without the overhead of full profiling.

---

### 3.8 eInfer (eBPF Research — ACM Workshop Paper)

**Layer:** OS/Kernel + Framework Runtime + Driver (distributed eBPF tracing for LLM inference)

**What it can see:** eInfer reconstructs end-to-end per-request execution timelines across distributed LLM inference pipelines. It traces GPU driver calls, memory operations, scheduling decisions, and network communication using eBPF, and infers request-level semantics (mapping low-level events to individual user requests) without application modifications. It introduces "Cooperative Kernel Proxies" — lightweight modules embedded in user-space components (PyTorch, TensorFlow) that extract GPU metadata without modifying drivers, bridging the visibility gap for accelerators.

**What it cannot see:** Like GPUprobe, it operates from the CPU/OS side and cannot directly observe device-side execution internals (warp behavior, cache misses, SM occupancy). The "near-parity with CUPTI" claim refers to timing accuracy for kernel-level events, not hardware counter access.

**Overhead:** The paper reports approximately 1% overhead, in stark contrast to PyTorch Profiler's severe performance degradation. The three-tier hierarchical data reduction (kernel-level filtering → node-level aggregation → cluster-level merging) with dynamic sampling rate adjustment is key to keeping overhead low.

**Production readiness:** Research prototype. Evaluated on NVIDIA A6000 systems. The paper was published at the ACM eBPF and Kernel Extensions Workshop (2025). Not available as a turnkey open-source tool — it's a research contribution demonstrating the feasibility of eBPF-based distributed LLM inference tracing.

**Deployment requirements:** Linux with eBPF, CUDA, and the cooperative kernel proxy components. The research was done with NVIDIA GPUs, but the design claims vendor-agnostic aspirations (with CKP interface supporting heterogeneous accelerators).

**Best for:** Understanding the frontier of what's possible in production-grade distributed inference observability. If you're writing a survey blog post, eInfer represents the "where things are heading" chapter.

---

### 3.9 bpftime/eGPU and gpu_ext

**Layer:** Device Hardware + OS/Kernel (eBPF programs executing *on the GPU*)

**What it can see:** This is the most radical approach in the survey. eGPU (part of the bpftime project) compiles eBPF bytecode into NVIDIA PTX intermediate representation and injects it into running GPU kernels at runtime. This allows device-side instrumentation — profiling counters, memory checks, custom analytics — embedded directly into GPU kernel execution. The follow-up work, gpu_ext, extends this to a full policy runtime treating the GPU driver and device as a programmable OS subsystem, enabling adaptive memory prefetching, fine-grained kernel preemption, and dynamic work-stealing schedulers.

**What it cannot see:** This is less about visibility limitations and more about maturity limitations. The technology is cutting-edge research — the eGPU paper was published at the ACM HCDS Workshop in April 2025, and gpu_ext in December 2025. The instrumentation is functional but the ecosystem (tooling, documentation, production hardening) is nascent.

**Overhead:** eGPU reports low instrumentation overhead in micro-benchmarks. gpu_ext demonstrates policy improvements (up to 4.8× throughput improvement, 2× tail latency reduction) with low overhead for instrumentation-only deployments. But these are research evaluations, not production measurements at scale.

**Production readiness:** Research prototype. This is 2–3 years away from production readiness at minimum. The significance is conceptual: it proves that eBPF's programmability, safety guarantees, and dynamic attachment can extend to GPUs, opening a path toward device-resident observability that doesn't require vendor-specific tooling.

**Deployment requirements:** The bpftime eBPF runtime, NVIDIA GPU with CUDA, and the eGPU/gpu_ext extensions (available in the bpftime main branch). Requires the NVIDIA open GPU kernel modules for gpu_ext's driver extensions.

**Best for:** Academic interest and forward-looking analysis in your blog post. This represents the "5-year horizon" for GPU observability.

---

### 3.10 Additional Notable Tools

**Polar Signals parca-agent (CUPTI + USDT + eBPF):** Released in v0.43.0 (October 2025), this combines CUPTI activity tracing with USDT probes and eBPF for what they claim is the first open-source always-on GPU profiler suitable for continuous production profiling. Uses the `CUDA_INJECTION64_PATH` mechanism for zero-code-change deployment. Captures kernel timing and correlates it with CPU stack traces via CUPTI correlation IDs. Supports both regular kernel launches and CUDA graph executions.

**ProfInfer (eBPF):** A fine-grained eBPF-based profiler specifically targeting LLM inference engines like llama.cpp. Attaches uprobes to runtime functions across multiple layers, producing rich visualizations of operators, graphs, timelines, and hardware counter trends. Notably includes on-device/edge evaluation (Rubik Pi) alongside server-class hardware.

**xpu-perf (eunomia-bpf):** Combines eBPF CPU tracing with CUPTI GPU monitoring. Three modes: merge (CPU+GPU unified flamegraphs), GPU-only causality (which CPU code launched which kernels), and CPU-only sampling. Claims <1% CPU overhead, <5% GPU overhead. Still work-in-progress.

**Ingero:** An eBPF-based GPU causal observability agent that traces CUDA Runtime/Driver APIs via uprobes and host kernel events via tracepoints. Builds causal chains explaining GPU latency with full Python-to-CUDA stack traces.

**zymtrace:** A commercial offering (from the team behind the OTel eBPF CPU Profiler) that targets cluster-scale continuous GPU profiling. Positions itself as the production-grade alternative to Nsight for distributed workloads.

---

## 4. Comparison Matrix

| Tool | Layer | Overhead | Production? | Sees Kernel Internals? | Sees App Semantics? | Code Changes? |
|---|---|---|---|---|---|---|
| nvidia-smi | Device HW | ~0% | ✅ Yes | ❌ No | ❌ No | None |
| DCGM | Device HW + Driver | ~5% | ✅ Yes | ❌ No (coarse counters only) | ❌ No | None |
| Nsight Systems | All layers (timeline) | 2–10× | ❌ Dev only | ❌ No | Partial (NVTX) | Optional (NVTX) |
| Nsight Compute | Device HW (per-kernel) | 20–200× | ❌ Dev only | ✅ Yes (deep) | ❌ No | None |
| CUPTI | Runtime + Driver | Varies (1%–100×) | Depends on config | Via HW counters | ❌ No | None (injection) |
| PyTorch Profiler | Application + Runtime | 10–100%+ | Limited (burst) | ❌ No | ✅ Yes | Minimal (wrapper) |
| GPUprobe | OS/Runtime boundary | <5% | Experimental | ❌ No | ❌ No | None |
| eInfer | OS + Runtime + Driver | ~1% | Research | ❌ No | Inferred | None |
| eGPU/gpu_ext | Device HW (on-GPU eBPF) | Low (research) | Research | ✅ Yes (novel) | ❌ No | None |
| parca-agent | Runtime (CUPTI+eBPF) | ~1% (inactive), low (active) | Emerging | ❌ No | ❌ No | None |

---

## 5. Additional Tools & Tests to Consider

Beyond the core nine tools, here are additional approaches that would strengthen your survey:

**Holistic Trace Analysis (HTA):** Meta's open-source library for analyzing PyTorch Profiler traces at scale. It computes temporal breakdowns (compute vs. communication vs. idle), identifies stragglers in distributed training, and highlights communication bottlenecks. Running HTA on your collected traces would add an analytical layer to your raw profiling data.

**NVIDIA Nsight Deep Learning Designer (NDDL) and TensorRT Profiler:** If you're profiling TensorRT-optimized models (common for production inference), the TensorRT profiler provides engine-level breakdowns (layer execution times, memory usage, reformatting overhead) that none of the tools above capture.

**vLLM's built-in metrics:** If you're profiling LLM inference specifically, vLLM exports request-level metrics (time-to-first-token, inter-token latency, throughput) that represent the *user-facing* observability layer above all the GPU-level tools.

**pcm (Intel Performance Counter Monitor) for CPU side:** GPU inference performance is often bottlenecked by CPU-side preprocessing, tokenization, or scheduling. Profiling the CPU alongside the GPU gives a complete picture.

**Triton Profiler:** If you're working with OpenAI Triton kernels (increasingly common in PyTorch 2.x compiled models), Triton has its own profiling utilities that bridge the gap between PyTorch Profiler and Nsight Compute.

**Custom CUDA event timing:** For minimal-overhead production latency measurement, inserting CUDA events (`cudaEventRecord`) around critical sections gives you sub-microsecond timing with negligible overhead — useful as a ground-truth comparison for the other tools.

**Network profiling (NCCL, NVLink, PCIe):** For multi-GPU or distributed inference, communication is often the bottleneck. NCCL's `NCCL_DEBUG=INFO` logging, combined with DCGM's NVLink/PCIe counters, exposes the inter-device communication layer.

---

## 6. Recommended Test Matrix

For a compelling blog post, I'd recommend structuring your experiments around a consistent workload (e.g., vLLM serving Llama-3-8B with synthetic load) and profiling it with each tool tier:

**Tier 1 — Always-on baseline (run these simultaneously):**
- nvidia-smi dmon (1s polling)
- DCGM Exporter + Prometheus + Grafana
- GPUprobe daemon

**Tier 2 — Burst profiling (run these one at a time, under controlled load):**
- PyTorch Profiler (scheduled, 3–5 iterations)
- Nsight Systems (30-second capture)

**Tier 3 — Deep dive (run on isolated kernels):**
- Nsight Compute (target specific hot kernels identified in Tier 2)

**Tier 4 — Frontier/experimental:**
- parca-agent with CUPTI injection
- xpu-perf flamegraph generation

This tiered approach tells a compelling story for your readers: here's what you can run all the time, here's what you run periodically, here's what you pull out for deep investigation, and here's what's coming next.

---

## 7. Cloud Resource Requirements & Cost Estimates

Given your constraints (Dell XPS laptop, $100–200 budget, 10–15 hrs/week), here's a realistic plan.

### What You Need

You need a single Linux VM with an NVIDIA GPU that supports data center features (for DCGM) and has CUDA installed. A T4 instance is the sweet spot for this survey — it's the cheapest data center GPU, supports DCGM, and is powerful enough to run inference workloads like a quantized 7–8B parameter model.

For Nsight Compute's deep-dive experiments, you might optionally want one short session on an A100 to show the richer hardware counter data, but the T4 covers 90% of your needs.

### Provider Recommendations

For a solo hobby project on a tight budget, avoid the major clouds (AWS, GCP, Azure) where T4 pricing is $0.35–0.50/hr *before* the VM cost. Instead, use a marketplace or budget provider:

| Provider | GPU | Approx. Cost/hr | Notes |
|---|---|---|---|
| **Vast.ai** (community) | T4 | ~$0.10–0.15/hr | Cheapest option; variable availability; root access; bare-metal feel |
| **RunPod** | T4 | ~$0.20/hr | More reliable; good templates; community cloud option |
| **Lambda Labs** | A100 40GB | ~$1.29/hr | For your optional A100 session; straightforward pricing |
| **Google Cloud (spot)** | T4 | ~$0.09–0.12/hr per GPU | If you already have GCP credits; can be interrupted |

### Budget Breakdown

Assuming $150 total budget and 12 hrs/week over 4 weeks of experimentation:

| Phase | Hours | Instance | Cost/hr | Subtotal |
|---|---|---|---|---|
| Setup & Tier 1 testing (DCGM, nvidia-smi, GPUprobe) | 10 hrs | T4 on Vast.ai | $0.15 | $1.50 |
| Tier 2 testing (PyTorch Profiler, Nsight Systems) | 10 hrs | T4 on Vast.ai | $0.15 | $1.50 |
| Tier 3 testing (Nsight Compute deep dives) | 5 hrs | T4 on RunPod | $0.20 | $1.00 |
| Tier 4 / frontier tools (eBPF experiments) | 10 hrs | T4 on Vast.ai | $0.15 | $1.50 |
| A100 session (optional, for richer HW counters) | 3 hrs | A100 on Lambda | $1.29 | $3.87 |
| Buffer for false starts, re-runs, storage | — | — | — | $10.00 |
| **Total estimated** | **~38 hrs** | | | **~$20** |

You're looking at roughly $20–40 in actual GPU compute costs — well within your $100–200 budget. The remaining budget gives you room for re-runs, additional experiments, or a longer A100 session to compare T4 vs. A100 profiling fidelity.

### Setup Notes

- **Use a persistent Docker image or VM snapshot** with your tooling pre-installed (CUDA toolkit, Nsight CLI tools, Python environment with PyTorch, GPUprobe, Prometheus, Grafana). This avoids burning GPU-hours on setup.
- **Pre-build your workload offline** on your Dell XPS (the inference model, benchmark scripts, etc.) and upload it to the GPU instance.
- **For Vast.ai/RunPod,** you get root access and can install eBPF tools (bpftool, bcc, libbpf). Some managed Kubernetes-based providers restrict kernel access, which breaks eBPF — check before provisioning.
- **Capture everything.** Export all traces, logs, and metrics to local storage before tearing down instances. GPU hours are expensive to re-do.

---

## 8. Suggested Blog Post Structure

Given your 10–15 hrs/week, here's a realistic 4–5 week timeline:

**Week 1:** Set up your cloud environment, install all tools, get the baseline inference workload running. Write the "why observability matters" and "stack model" sections.

**Week 2:** Run Tier 1 and Tier 2 experiments. Collect screenshots, Grafana dashboards, PyTorch Profiler traces, Nsight Systems timelines. Write the tool-by-tool analysis for the established tools.

**Week 3:** Run Tier 3 experiments (Nsight Compute). Attempt GPUprobe and parca-agent experiments. Write the eBPF tools section.

**Week 4:** Optional A100 session. Synthesize the comparison matrix. Write the "what's coming next" frontier section covering eInfer, eGPU, and the broader eBPF-on-GPU movement.

**Week 5:** Edit, add visualizations, publish.

---

## 9. Key Takeaways

The GPU profiling landscape in 2025–2026 is stratified into three clear tiers:

**Production-safe, coarse-grained:** nvidia-smi and DCGM give you fleet-level health and utilization. They're table stakes for any GPU deployment but blind to application semantics.

**Development-time, fine-grained:** Nsight Systems, Nsight Compute, and PyTorch Profiler give you deep insight but at prohibitive cost for production. The recommended workflow is: PyTorch Profiler → Nsight Systems → Nsight Compute, narrowing from application to kernel.

**The emerging middle ground:** eBPF-based tools (GPUprobe, parca-agent, eInfer, ProfInfer) are attempting to deliver application-level insight at production-safe overhead. This is the most exciting space to watch — and the most compelling angle for your blog post. The fact that Meta deployed eBPF for fleet-wide GPU profiling, that Polar Signals shipped the first open-source always-on GPU profiler, and that research teams are pushing eBPF *onto the GPU itself* via PTX injection — all of this happened in 2024–2025 and represents a genuine paradigm shift.

---

*Last updated: March 2026. All overhead numbers are approximate and workload-dependent. Cloud pricing reflects market rates as of early 2026 and is subject to change.*

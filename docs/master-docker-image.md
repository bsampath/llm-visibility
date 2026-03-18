Build a single Docker image that contains everything you'll need across all 8 weeks. This avoids reinstalling tools on every GPU session.

 Create a Dockerfile based on nvcr.io/nvidia/pytorch:24.12-py3 (includes CUDA toolkit, PyTorch, Nsight CLI tools)
 Add to image: vLLM, GPUprobe dependencies (Rust toolchain, bpftool, libbpf-dev), Prometheus, Grafana, dcgm-exporter, python benchmarking scripts
 Add to image: stress-ng, htop, perf, bcc-tools, bpftrace for OS-level profiling
 Add a scripts/ directory with placeholder benchmark scripts
 Build and test the image locally (it will fail GPU operations but should build cleanly)
 Push to Docker Hub or GitHub Container Registry

FROM nvcr.io/nvidia/pytorch:24.12-py3

# ── System packages ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # GPUprobe / eBPF dependencies
    libbpf-dev \
    linux-tools-generic \
    linux-headers-generic \
    # OS-level profiling
    stress-ng \
    htop \
    bpfcc-tools \
    bpftrace \
    # General utilities
    curl \
    wget \
    git \
    build-essential \
    pkg-config \
    cmake \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# ── Rust toolchain (required for GPUprobe) ───────────────────────────────────
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# ── Python packages ──────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    vllm \
    # Benchmarking / observability
    prometheus-client \
    openai \
    aiohttp \
    requests \
    pandas \
    numpy \
    psutil \
    gputil \
    py-cpuinfo \
    tqdm \
    pynvml

# ── Prometheus ────────────────────────────────────────────────────────────────
ARG PROMETHEUS_VERSION=2.51.2
RUN wget -q "https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz" \
    -O /tmp/prometheus.tar.gz \
    && tar -xzf /tmp/prometheus.tar.gz -C /opt \
    && mv "/opt/prometheus-${PROMETHEUS_VERSION}.linux-amd64" /opt/prometheus \
    && ln -s /opt/prometheus/prometheus /usr/local/bin/prometheus \
    && ln -s /opt/prometheus/promtool /usr/local/bin/promtool \
    && rm /tmp/prometheus.tar.gz

# ── Grafana ───────────────────────────────────────────────────────────────────
ARG GRAFANA_VERSION=10.4.2
RUN wget -q "https://dl.grafana.com/oss/release/grafana-${GRAFANA_VERSION}.linux-amd64.tar.gz" \
    -O /tmp/grafana.tar.gz \
    && tar -xzf /tmp/grafana.tar.gz -C /opt \
    && mv "/opt/grafana-v${GRAFANA_VERSION}" /opt/grafana \
    && ln -s /opt/grafana/bin/grafana /usr/local/bin/grafana \
    && ln -s /opt/grafana/bin/grafana-server /usr/local/bin/grafana-server \
    && rm /tmp/grafana.tar.gz

# ── DCGM Exporter ─────────────────────────────────────────────────────────────
# NVIDIA does not publish standalone binaries for dcgm-exporter.
# Run it as a sidecar container on the GPU host:
#   docker run -d --gpus all --cap-add SYS_ADMIN -p 9400:9400 \
#     nvcr.io/nvidia/k8s/dcgm-exporter:4.5.2-4.8.1-distroless

# ── Benchmark scripts placeholder ─────────────────────────────────────────────
WORKDIR /workspace
RUN mkdir -p scripts

COPY scripts/ scripts/

# ── Default command ───────────────────────────────────────────────────────────
CMD ["/bin/bash"]

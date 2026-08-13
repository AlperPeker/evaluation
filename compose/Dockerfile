FROM debian:bookworm-slim

RUN apt-get update -y && apt-get install -y \
    ca-certificates \
    curl \
    wget \
    git \
    build-essential \
    libseccomp-dev \
    pkg-config \
    fuse3 \
    iproute2 \
    jq \
    python3 \
    file \
    && rm -rf /var/lib/apt/lists/*

# Go
RUN wget -O go.tar.gz https://golang.org/dl/go1.24.0.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go.tar.gz && rm go.tar.gz
ENV PATH="/usr/local/go/bin:${PATH}"

# containerd and runc
ARG CONTAINERD_VERSION=v2.1.5
ARG RUNC_VERSION=v1.3.3
ARG NERDCTL_VERSION=2.2.0
ARG BUILDKIT_VERSION=v0.13.2

RUN cd /tmp && \
    git clone -b ${CONTAINERD_VERSION} --depth 1 https://github.com/containerd/containerd && \
    cd containerd && make && make install PREFIX=/usr/local

RUN cd /tmp && \
    git clone -b ${RUNC_VERSION} --depth 1 https://github.com/opencontainers/runc && \
    cd runc && make && make install PREFIX=/usr/local

# stargz-snapshotter
COPY binaries/containerd-stargz-grpc /usr/local/bin/
COPY binaries/ctr-remote /usr/local/bin/
# RUN cd /tmp && \
#     git clone https://github.com/containerd/stargz-snapshotter.git && \
#     cd stargz-snapshotter && \
#     make containerd-stargz-grpc && \
#     make ctr-remote && \
#     cp out/containerd-stargz-grpc /usr/local/bin/ && \
#     cp out/ctr-remote /usr/local/bin/

RUN wget -O /tmp/buildkit.tar.gz \
      https://github.com/moby/buildkit/releases/download/${BUILDKIT_VERSION}/buildkit-${BUILDKIT_VERSION}.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf /tmp/buildkit.tar.gz && \
    rm /tmp/buildkit.tar.gz

RUN wget -O /tmp/nerdctl.tar.gz \
      https://github.com/containerd/nerdctl/releases/download/v${NERDCTL_VERSION}/nerdctl-${NERDCTL_VERSION}-linux-amd64.tar.gz && \
    tar -C /usr/local/bin -xzf /tmp/nerdctl.tar.gz && \
    rm /tmp/nerdctl.tar.gz

# configuration files
RUN mkdir -p /etc/containerd /etc/containerd-stargz-grpc
COPY containerd-config-eval.toml /etc/containerd/config.toml
COPY stargz-config.toml /etc/containerd-stargz-grpc/config.toml
RUN mkdir -p /etc/containerd/certs.d/registry:5000
COPY hosts.toml /etc/containerd/certs.d/registry:5000/hosts.toml

# startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# 2dfs from local binary
# TODO replace with remote install script
COPY binaries/tdfs /usr/local/bin/tdfs
RUN mkdir /2dfs-files
COPY sample-image/ /2dfs-files/

# evaluation script
# COPY *.py /
# RUN chmod +x /eval.py

# sample dockerfiles
# RUN mkdir -p /workspace/sample-image
# COPY sample-image/ /workspace/sample-image

RUN mkdir -p /tmp && chmod 1777 /tmp

WORKDIR /2dfs-files
CMD ["/start.sh", "python3", "/eval.py"]
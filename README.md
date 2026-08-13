# Benchmark setup

The benchmark runs on two cloudlab `c6525-25g` nodes ([spec](https://www.utah.cloudlab.us/portal/show-nodetype.php?type=c6525-25g&_gl=1*wfe0yn*_ga*MTg1OTgyNjU4MNzcxNDE4OTE3*_ga_6W2Y02FJX6*czE3NzM4NTAzNzgkbzcyJGcwJHQxNzczODUwNDIwJGoxOCRsMCRoMA)):
a registry node, and a combined builder/client node. This setup can also run on arbitrary server hardware. However, it is important to make sure that root folder `/mydata` exists.

## Cloudlab guide
- go to experiments -> start an experiment
- pick a profile. In general profile doesn't matter because the hardware is defined separately. We use `small-lan` in our experiments.

![first step](docs/cloudlab-first-step.png)

- specify node type, OS, etc
- important to select "temp filesystem max space". In this case temporaray filesystem size doesn't matter as Cloudlab will allocate all available storage on that node

![second step](docs/cloudlab-second-step.png)

## About `/mydata`

Cloudlab root disk is too small for benchmark artifacts, so the benchmark lives on the `/mydata`
mount point allocated in the cloudlab profile. Cloudlab can allocate up to 1TB into the mount point of user's choice.

- `/var/lib/containerd` and `/var/lib/containerd-stargz-grpc` are symlinked into `/mydata`
- `TMPDIR=/mydata/tmp`, buildkit root `/mydata/buildkit`
- 2dfs home dir - `/mydata/.2dfs`, `HF_HOME=/mydata/huggingface`
- registry storage in `/mydata/2dfs-registry-data`
- both repos in `/mydata/evaluation` and `/mydata/split-llm-simple`

## Setup steps

### get registry ip - `10.10.1.2` by default (cloudlab)

```bash
hostname -I | awk '{print $2}'
```

### registry setup (run before client)

```bash
curl -Lo "${HOME}/registry-node-setup.sh" \
	https://raw.githubusercontent.com/mitrafsnap/evaluation/main/node-setup/registry-node-setup.sh

chmod +x "${HOME}/registry-node-setup.sh"
sudo "${HOME}/registry-node-setup.sh"
```

### combined node setup (client + builder)

- install kernel (stargz FUSE passthrough needs >= 6.9)
- the script reboots the machine, reconnect and continue after reboot

```bash
cd /mydata
sudo chown $USER /mydata
git clone https://github.com/mitrafsnap/evaluation.git
cd evaluation/
sudo ./node-setup/install-hwe-kernel.sh
```

- node setup script
- use correct registry ip and stargz-snapshotter implementation

```bash
cd /mydata/evaluation/node-setup
sudo ./combined-node-setup.sh 10.10.1.2 https://github.com/mitrafsnap/stargz-snapshotter.git
```

- env setup
- prepare `HF_TOKEN` with access to models used in the benchmark
```bash
cd ..
python3 -m venv .venv-benchmark
source .venv-benchmark/bin/activate
pip install -r ./benchmark/requirements.txt
echo "HF_TOKEN=..." > ./benchmark/.env
```

- benchmark uses `shared/config.yaml` by default. Copy desired config and adjust registry variable if needed.

```bash
cp /mydata/evaluation/benchmark/shared/config-cloudlab.yaml /mydata/evaluation/benchmark/shared/config.yaml 
```

- clone split_llm (needs an ssh key)

```bash
cat > ~/.ssh/id_ed25519 <<'EOF'
-----BEGIN OPENSSH PRIVATE KEY-----
...
-----END OPENSSH PRIVATE KEY-----
EOF

chmod 600 ~/.ssh/id_ed25519
cd /mydata/
git clone git@gitlab.lrz.de:distributed-ml/split-llm-simple.git
```

- install dependencies, use specified venv name as it is later used in the benchmark 

```bash
cd /mydata/split-llm-simple
sudo apt install -y graphviz
python3 -m venv .venv-split-llm
source .venv-split-llm/bin/activate
pip install -r requirements.txt
```

- download CV model splits (optional)

```bash
cd /mydata/evaluation
./util/download-splits.sh
```

### Run the benchmark

`run-bg.sh` runs `run.py` in detached mode. Set `NTFY_TOPIC` for
push notifications on finish/failure.

```bash
cd /mydata/evaluation
source .venv-benchmark/bin/activate
cd benchmark
./run-bg.sh
```

Edit `run.py` to specify which benchmark phases you want to run.
Results are written to `build_performance/{results,charts}/` and
`pull_performance/{results,charts}/`. 
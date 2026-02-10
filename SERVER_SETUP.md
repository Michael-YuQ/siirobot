# AT-PC Training Server Setup Guide

Tested on: NGC PyTorch 24.05 + CUDA 12.4 + 4090 (48GB)

## 1. Clone repo (avoid disk quota issues)

The project directory `/inspire/hdd/...` may have disk quota limits. Clone to `/root/` instead:

```bash
cd /root
GIT_TEMPLATE_DIR=/dev/null git clone https://github.com/Michael-YuQ/siirobot.git
cd siirobot
```

`GIT_TEMPLATE_DIR=/dev/null` skips git template files that can trigger quota errors.

## 2. Install Miniconda + Python 3.8

IsaacGym Preview 4 only has .so binaries for Python 3.6/3.7/3.8. NGC images ship Python 3.10, so we need conda:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /root/miniconda
eval "$(/root/miniconda/bin/conda shell.bash hook)"
```

Accept Terms of Service if prompted:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

Create Python 3.8 environment:

```bash
conda create -n rl python=3.8 -y
conda activate rl
```

## 3. Install IsaacGym

IsaacGym is not on PyPI. Upload it to your file server first, then download:

```bash
cd /root
wget "http://111.170.6.103:10005/download?path=isaacgym/isaacgym.tar.gz" -O isaacgym.tar.gz
tar xf isaacgym.tar.gz
```

Fix Python version restriction in setup.py:

```bash
sed -i "s/python_requires='>=3.6,<3.9'/python_requires='>=3.6'/" /root/isaacgym/python/setup.py
```

Install:

```bash
pip install -e /root/isaacgym/python --no-deps
```

## 4. Install PyTorch

```bash
pip install torch==2.0.1 torchvision --index-url https://download.pytorch.org/whl/cu118
```

This is ~2GB, takes a few minutes.

## 5. Fix LD_LIBRARY_PATH

The system Python 3.10 torch can interfere. Force conda's torch to load first:

```bash
export LD_LIBRARY_PATH=/root/miniconda/envs/rl/lib/python3.8/site-packages/torch/lib:/root/miniconda/envs/rl/lib:
unset PYTHONPATH
```

## 6. Install rsl_rl

Latest rsl_rl has pyproject.toml issues with Python 3.8 setuptools. Use v1.0.2:

```bash
pip install git+https://github.com/leggedrobotics/rsl_rl.git@v1.0.2
```

## 7. Install numpy (compatible version)

numpy >= 1.24 removed `np.float` which IsaacGym still uses:

```bash
pip install numpy==1.23.5
```

## 8. Install project + remaining deps

```bash
pip install -e /root/siirobot --no-deps
pip install requests matplotlib scipy pyyaml imageio ninja tensorboard
```

## 9. Verify

```bash
python -c "import isaacgym; print('isaacgym OK')"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import rsl_rl; print('rsl_rl OK')"
```

All three should pass.

## 10. Run experiments

```bash
cd /root/siirobot

# Single experiment
python -m experiments.run_experiment --generator G1 --method atpc --seed 42

# All 36 experiments
python -m experiments.run_all
```

## One-liner setup script

Copy-paste this entire block after cloning and extracting isaacgym:

```bash
eval "$(/root/miniconda/bin/conda shell.bash hook)"
conda activate rl
export LD_LIBRARY_PATH=/root/miniconda/envs/rl/lib/python3.8/site-packages/torch/lib:/root/miniconda/envs/rl/lib:
unset PYTHONPATH
cd /root/siirobot
python -m experiments.run_all
```

## Troubleshooting

| Error | Fix |
|-------|-----|
| `Disk quota exceeded` on git clone | Clone to `/root/` instead of project dir |
| `No gym module found for Python (3.10)` | Use conda Python 3.8 environment |
| `libpython3.8.so.1.0: cannot open` | `export LD_LIBRARY_PATH=/root/miniconda/envs/rl/lib:$LD_LIBRARY_PATH` |
| `libtorch_python.so: undefined symbol` | `unset PYTHONPATH` and set LD_LIBRARY_PATH as above |
| `np.float` AttributeError | `pip install numpy==1.23.5` |
| `python_requires '<3.9'` on isaacgym | `sed -i` fix in step 3 |
| rsl_rl pyproject.toml error | Use `@v1.0.2` tag |
| `Object of type Tensor is not JSON serializable` | Already fixed in code |
| `deepcopy RuntimeError on non-leaf Tensors` | Already fixed in code |

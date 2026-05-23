# IK2221 Assignment 2- GPU Inferencing

## Setup

- Clone the repository: `git clone git@github.com:sophiaaa05/ik2221_project2.git`, then navigate into it
- Download the UV package manager: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Create a new virtual environment and activate it: `uv venv ./venv --python 3.12 && source ./venv/bin/activate`
- Install required packages: `uv pip install -r ./lmcache-vllm-extended/requirements.txt`
- Install LMCache and LMCache Server as editable packages in the venv: `uv pip install -e ./lmcache-vllm-extended && uv pip install -e ./LMCache && uv pip install -e ./lmcache-server`

To work with Git, you'll need to set up the VM's SSH key and add to your Github account; see [this article](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent). You may also have to update your git config to make commits appear from your correct account in the repo by running:

- git config --global user.name "your-github-username"
- git config --global user.email "<your-github-email@example.com>"

## Running the application

- Run the LMCache storage server: `python3 -m lmcache_server.server 127.0.0.1 65432 data/`
- Run the LMCache engine: `LMCACHE_CONFIG_FILE=lmcache-vllm-extended/configuration.yaml CUDA_VISIBLE_DEVICES=0 python lmcache-vllm-extended/lmcache_vllm/script.py serve Qwen/Qwen2.5-3B-Instruct  --gpu-memory-utilization 0.8 --dtype half --port 8000 --guided-decoding-backend lm-format-enforcer --max-model-len 8192`
- Run the frontend: `cd lmcache-vllm-extended/frontend && streamlit run frontend.py`

The frontend will be visible on `https://gpu1.eecs.kth.se/user/<kth-username>/vscode/proxy/8501/`.

## Running the benchmarks

> Note: Some Makefile paths may have to be adjusted based on your setup

- Set `max_local_cache_size` to the desired local KV cache size in `configuration.yaml`

- Clean cached KV blocks and/or staged RAG documents:

```bash
make clean
````

- Run the LMCache storage server:

```bash
make server
```

- Run the LMCache engine:

```bash
make engine
```

### Task 1 + 2

- Run the request generator, with the appropriate options set:

```bash
python lmcache-vllm-extended/request_generator.py
    # --mode single                                 Task 1.1
    # --mode repeat                                 Task 1.2
    # --mode diverse_sweep --max-contexts <x>       Task 1.3
    # --mode batch --batch-size <x>                 Task 2
    # --mode overlap --batch-size <x>               Task 2
```

### Task 3

- Run the task 3 evaluation with 14 documents:

```bash
make ndocs-14
```

See the `Makefile` for additional testing options and configurations.

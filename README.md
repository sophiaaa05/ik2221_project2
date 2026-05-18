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

## Development

- Run the LMCache storage server: `python3 -m lmcache_server.server 127.0.0.1 65432 kv-cache/`
- Run the LMCache engine: `LMCACHE_CONFIG_FILE=lmcache-vllm-extended/configuration.yaml CUDA_VISIBLE_DEVICES=0 python lmcache-vllm-extended/lmcache_vllm/script.py serve Qwen/Qwen2.5-1.5B-Instruct  --gpu-memory-utilization 0.8 --dtype half --port 8000 --guided-decoding-backend lm-format-enforcer`
- Run the frontend: `cd lmcache-vllm-extended/frontend && streamlit run frontend.py`

The frontend will be visible on `https://gpu1.eecs.kth.se/user/<kth-username>/vscode/proxy/8501/`.

LMCACHE_CONFIG_FILE=lmcache-vllm-extended/configuration.yaml CUDA_VISIBLE_DEVICES=0 python lmcache-vllm-extended/lmcache_vllm/script.py serve Qwen/Qwen2.5-3B-Instruct  --gpu-memory-utilization 1.0 --dtype half --port 8000 --guided-decoding-backend lm-format-enforcer --max-model-len 8192
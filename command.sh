cp -r /opt/conda/envs/vllm-23-env /root/private_data/env/vllm-23-env
ln -s /root/private_data/env/vllm-23-env /opt/conda/envs/vllm-23-env

conda create -n vllm-23-env python=3.11 -y \
  --override-channels \
  --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
  --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge


conda activate vllm-23-env
pip install vllm==0.23.0

sudo apt-get update && sudo apt-get install -y nginx
apt-get update && apt-get install -y curl

export MODEL_PATH=/root/private_data/models/Qwen/Qwen3.5-4B
export MODEL_NAME=Qwen3.5-4B
export TP_SIZE=1
export CONCURRENCY=64
export STOP_AFTER=1

bash run_pipeline.sh
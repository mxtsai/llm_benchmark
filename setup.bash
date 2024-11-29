# install wget
sudo apt-get update
sudo apt-get install wget nano tmux net-tools iputils-ping

# setup library path for cuda
export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64

# install miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

# init conda
source ~/miniconda3/bin/activate
conda init --all

# new conda env
conda create -n vllm python=3.10
# What did I do?
**Sophia is not an AI testbed**

Sophia is comprised of 24 NVIDIA DGX A100 nodes. Each DGX A100 node comprises eight NVIDIA A100 Tensor Core GPUs and two AMD Rome CPUs that provide 22 nodes with 320 GB of GPU memory and two nodes with 640 GB of GPU memory (8,320 GB in total) for training artificial intelligence (AI) datasets, while also enabling GPU-specific and -enhanced high-performance computing (HPC) applications for modeling and simulation.
A 15-terabyte solid-state drive offers up to 25 gigabits per second in bandwidth. The dedicated compute fabric comprises 20 Mellanox QM9700 HDR200 40-port switches wired in a fat-tree topology.


## Logging into Sophia:
```
ssh <username>@sophia.alcf.anl.gov
```
Then, type in the password from your CRYPTOCard/MobilePASS+ token. Once logged in, you land on one of the Sophia login nodes (sophia-login-01, sophia-login-02).

## Containers on Sophia: Apptainer

Sophia employs Apptainer (formerly known as Singularity) for container management. To set up Apptainer, run:
```
module use /soft/spack/base/0.7.1/install/modulefiles/Core/
module load apptainer
apptainer version #1.3.3
```
### Connect to outer internet:
```
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
```

## Create a virtual ENV:
```
mkdir -p venv_sophia
python -m venv venv_sophia --system-site-packages
source venv_sophia/bin/activate
```

### ENV for Autotrain
**I don't know if I am going to use HuggingFace API for training or whatever**
[Autotrain](https://huggingface.co/docs/autotrain/main/en/tasks/llm_finetuning), developed by Hugging Face, is a platform designed to simplify training cutting-edge models in various fields: NLP, LLM, CV, etc. Let's first create a virtual environment for Autotrain, built on top of the minimal system Python installation located at /usr/bin/python:
```
mkdir -p venv_autotrain
python -m venv venv_autotrain --system-site-packages
source venv_autotrain/bin/activate
pip3 install autotrain-advanced
```

**Note: If** Autotrain doesn't work properly, you may have to reinstall nvidia-ml-py.
```
pip3 uninstall nvidia-ml-py3 pynvml
pip3 install --force-reinstall nvidia-ml-py==11.450.51
```

### Config File for Fine-Tuning Local LLM
Here is an example to create a config file for supervised fine-tuning purposes:
```
task: llm-sft
base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
project_name: Llama-3-1-FT
log: wandb
backend: local
data:
  path: Path/to/the/training/dataset/folder
  train_split: train
  valid_split: null
  chat_template: null
  column_mapping:
    text_column: text
params:
  block_size: 1024
  model_max_length: 8192
  epochs: 800
  batch_size: 2
  lr: 1e-5
  peft: true
  quantization: null
  target_modules: all-linear
  padding: right
  optimizer: paged_adamw_8bit
  scheduler: cosine
  gradient_accumulation: 8
  mixed_precision: bf16
hub:
  username: ***
  token: hf_***
  push_to_hub: true
```

### Building from Docker or Argonne GitHub Container Registry
Containers on Sophia can be built by writing Dockerfiles on a local machine and then publishing the container to DockerHub, or by directly building them on an ALCF compute node by writing an Apptainer recipe file. If you prefer to use existing containers, you can pull them from various registries like DockerHub and run them on Sophia.
Since Docker requires root privileges, which users do not have on Sophia, existing Docker containers must be converted to Apptainer. To build a Docker-based container on Sophia, use the following as an example:
```
qsub -I -A <Project> -l select=1:ngpus=8:ncpus=256 -l walltime=01:00:00 -l filesystems=home:eagle -l singularity_fakeroot=True -q by-node -k doe
apptainer build --fakeroot pytorch:22.06-py3.sing docker://nvcr.io/nvidia/pytorch:22.06-py3
```
You can find the latest prebuilt NVIDIA PyTorch containers here. The TensorFlow containers are here (though note that LCF doesn't typically prebuild the TF-1 containers). You can search the full container registry here. For custom containers tailored for Sophia, visit ALCF's GitHub container registry.
Note: Currently, container build and executions are only supported on the Sophia compute nodes.

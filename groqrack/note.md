# Willium Arnold and Murali Emani is still working on how to use GroqRack

# What did I do:
ALCF's Groq system consists of a single GroqRackTM compute cluster that provides an extensible accelerator network consisting of 9 GroqNodeTM (groq-r01-gn-01 through groq-r01-gn-09) nodes with a rotational multi-node network topology. Each of these GroqNodes consists of 8 GroqCardTM accelerators in them with integrated chip-to-chip connections with a dragonfly multi-chip topology.
GroqCardTM accelerator is a dual-width, full-height, three-quarter length PCI-Express Gen4 x16 adapter that includes a single GroqChipTM processor with 230 MB of on-chip memory. Based on the proprietary Tensor Streaming Processor (TSP) architecture, the GroqChip processor is a low latency and high throughput single core SIMD compute engine capable of 750 TOPS (INT8) and 188 TFLOPS (FP16) @ 900 MHz that includes advanced vector and matrix mathematical acceleration units. The GroqChip processor is deterministic, providing predictable and repeatable performance.

## Log in to a Login node and GroqRack node
ssh into the device:
```
ssh ALCFUserID@groq.ai.alcf.anl.gov
ssh groq-r01-gn-0[1-9].ai.alcf.anl.gov
```

If you expect a loss of an internet connection for any reason, for long-running jobs we suggest logging into a specific node and using either screen or tmux to create persistent command line sessions.
```
man screen
# or
man tmux
```

## Set Virtual Environments
Create a groqflow conda environment and activate it
```
export PYTHON_VERSION=3.10.12
conda create -n groqflow python=$PYTHON_VERSION -y
conda activate groqflow
```

## Running a groqflow sample

Clone the GroqFlow github repo
```
cd ~/
git clone https://github.com/groq/groqflow.git
cd groqflow
```

Running GroqFlow conda environments and run an example
```
conda activate groqflow
cd ~/groqflow/proof_points/natural_language_processing/minilm
pip install -r requirements.txt
python minilmv2.py
```

When tried to install requirements:
```
Installing collected packages: typing-extensions, triton, tqdm, sympy, safetensors, regex, nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufile-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, MarkupSafe, idna, hf-xet, fsspec, filelock, charset_normalizer, certifi, requests, nvidia-cusparse-cu12, nvidia-cufft-cu12, nvidia-cudnn-cu12, jinja2, nvidia-cusolver-cu12, huggingface-hub, torch, tokenizers, transformers
  Attempting uninstall: typing-extensions
    Found existing installation: typing_extensions 4.9.0
    Uninstalling typing_extensions-4.9.0:
ERROR: Could not install packages due to an OSError: [Errno 13] Permission denied: 'METADATA'
Consider using the `--user` option or check the permissions.
```

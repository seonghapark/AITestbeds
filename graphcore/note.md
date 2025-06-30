# What did I do?

ssh into the machine:
```
ssh ALCFUserID@gc-login-01.ai.alcf.anl.gov
ssh gc-poplar-04.ai.alcf.anl.gov
```

## [Poplar SDK Setup](https://docs.alcf.anl.gov/ai-testbed/graphcore/virtual-environments/#poplar-sdk-setup)

The Poplar SDK is downloaded onto the graphcore systems at the /software/graphcore/poplar_sdk/ location. The default poplar version (3.3.0) is enabled automatically upon logging into a graphcore node. Check if Poplar is setup correctly:
```
conda deactivate
popc --version
```
One should see:
```
POPLAR version 3.3.0 (de1f8de2a7)
clang version 16.0.0 (2fce0648f3c328b23a6cbc664fc0dd0630122212)
```
**If the Poplar SDK is not enabled,** it can be enabled with
```
source /software/graphcore/poplar_sdk/3.3.0/enable
```

## Set miscellaneous env variable
```
export PYTHONPATH=$POPLAR_SDK_ROOT/python:$PYTHONPATH

mkdir ~/tmp
export TF_POPLAR_FLAGS=--executable_cache_path=~/tmp
export POPTORCH_CACHE_DIR=~/tmp

export POPART_LOG_LEVEL=WARN
export POPLAR_LOG_LEVEL=WARN
export POPLIBS_LOG_LEVEL=WARN

export PYTHONPATH=/software/graphcore/poplar_sdk/3.3.0/poplar-ubuntu_20_04-3.3.0+7857-b67b751185/python:$PYTHONPATH
```

## PopTorch Environment Setup
```
mkdir -p ~/venvs/graphcore
virtualenv ~/venvs/graphcore/poptorch33_env
source ~/venvs/graphcore/poptorch33_env/bin/activate

POPLAR_SDK_ROOT=/software/graphcore/poplar_sdk/3.3.0/
export POPLAR_SDK_ROOT=$POPLAR_SDK_ROOT
pip install $POPLAR_SDK_ROOT/poptorch-3.3.0+113432_960e9c294b_ubuntu_20_04-cp38-cp38-linux_x86_64.whl
```


## To Run Example Models/Programs
```
mkdir ~/graphcore
cd ~/graphcore
git clone https://github.com/graphcore/examples.git
cd examples
git checkout v3.3.0
```

### MNIST
Activate PopTorch Environment
```
source ~/venvs/graphcore/poptorch33_env/bin/activate
cd ~/graphcore/examples/tutorials/simple_applications/pytorch/mnist
pip install -r requirements.txt
```

I now really HATE version compatibility!!!
```
ERROR: Ignored the following yanked versions: 0.1.6, 0.1.7, 0.1.8, 0.1.9, 0.2.0, 0.2.1, 0.2.2, 0.2.2.post2, 0.2.2.post3
ERROR: Could not find a version that satisfies the requirement torchvision==0.15.2 (from versions: 0.17.0, 0.17.1, 0.17.2, 0.18.0, 0.18.1, 0.19.0, 0.19.1, 0.20.0, 0.20.1, 0.21.0, 0.22.0, 0.22.1)
ERROR: No matching distribution found for torchvision==0.15.2
```
So installed the oldest torchvision, and execute the command:
```
/opt/slurm/bin/srun --ipus=1 python mnist_poptorch.py
```
Then:

```
srun: job 40076 queued and waiting for resources
(few tens of seconds..)
srun: job 40076 has been allocated resources

A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.3.0 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

Traceback (most recent call last):  File "/home/seonghapark/graphcore/examples/tutorials/simple_applications/pytorch/mnist/mnist_poptorch.py", line 37, in <module>
    import torch
  File "/home/seonghapark/venvs/graphcore/poptorch33_env/lib/python3.12/site-packages/torch/__init__.py", line 1471, in <module>
    from .functional import *  # noqa: F403
  File "/home/seonghapark/venvs/graphcore/poptorch33_env/lib/python3.12/site-packages/torch/functional.py", line 9, in <module>
    import torch.nn.functional as F
  File "/home/seonghapark/venvs/graphcore/poptorch33_env/lib/python3.12/site-packages/torch/nn/__init__.py", line 1, in <module>
    from .modules import *  # noqa: F403
  File "/home/seonghapark/venvs/graphcore/poptorch33_env/lib/python3.12/site-packages/torch/nn/modules/__init__.py", line 35, in <module>
    from .transformer import TransformerEncoder, TransformerDecoder, \
  File "/home/seonghapark/venvs/graphcore/poptorch33_env/lib/python3.12/site-packages/torch/nn/modules/transformer.py", line 20, in <module>
    device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),
/home/seonghapark/venvs/graphcore/poptorch33_env/lib/python3.12/site-packages/torch/nn/modules/transformer.py:20: UserWarning: Failed to initialize NumPy: _ARRAY_API not found (Triggered internally at ../torch/csrc/utils/tensor_numpy.cpp:84.)
  device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),
Traceback (most recent call last):
  File "/home/seonghapark/graphcore/examples/tutorials/simple_applications/pytorch/mnist/mnist_poptorch.py", line 40, in <module>
    import poptorch
ModuleNotFoundError: No module named 'poptorch'
```

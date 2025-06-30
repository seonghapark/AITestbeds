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
/opt/slurm/bin/srun --ipus=1 python mnist_poptorch.py
```

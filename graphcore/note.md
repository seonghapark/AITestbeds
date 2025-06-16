# What did I do?

ssh into the machine:
```
ssh ALCFUserID@gc-login-01.ai.alcf.anl.gov
ssh gc-poplar-04.ai.alcf.anl.gov
```

## [Poplar SDK Setup](https://docs.alcf.anl.gov/ai-testbed/graphcore/virtual-environments/#poplar-sdk-setup)

The Poplar SDK is downloaded onto the graphcore systems at the /software/graphcore/poplar_sdk/ location. The default poplar version (3.3.0) is enabled automatically upon logging into a graphcore node. Check if Poplar is setup correctly:
```
popc --version
```
One should see:
```
POPLAR version 3.3.0 (de1f8de2a7)
clang version 16.0.0 (2fce0648f3c328b23a6cbc664fc0dd0630122212)
```
If the Poplar SDK is not enabled, it can be enabled with
```
source /software/graphcore/poplar_sdk/3.3.0/enable
```

### Set SDK env variable
```
POPLAR_SDK_ROOT=/software/graphcore/poplar_sdk/3.3.0/
export POPLAR_SDK_ROOT=$POPLAR_SDK_ROOT
```
Create a new virtual environment with this SDK and install popTorch and or other frameworks as needed.
```
virtualenv ~/Graphcore/workspace/poptorch33_env
source ~/Graphcore/workspace/poptorch33_env/bin/activate
```
Then it results:
```
created virtual environment CPython3.12.7.final.0-64 in 1650ms
  creator CPython3Posix(dest=/home/seonghapark/Graphcore/workspace/poptorch33_env, clear=False, no_vcs_ignore=False, global=False)
  seeder FromAppData(download=False, pip=bundle, wheel=latest, via=copy, app_data_dir=/home/seonghapark/.local/share/virtualenv)
    added seed packages: pip==25.1.1
  activators BashActivator,CShellActivator,FishActivator,NushellActivator,PowerShellActivator,PythonActivator
```

**First Error occurred** when continue setting env:
```
pip install $POPLAR_SDK_ROOT/poptorch-3.3.0+113432_960e9c294b_ubuntu_20_04-cp38-cp38-linux_x86_64.whl
```
Then it outputs:
```
ERROR: poptorch-3.3.0+113432_960e9c294b_ubuntu_20_04-cp38-cp38-linux_x86_64.whl is not a supported wheel on this platform.
```

**But continued**:
```
export PYTHONPATH=$POPLAR_SDK_ROOT/python:$PYTHONPATH

## and Miscellaneous Setup
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
```
Then:
```
created virtual environment CPython3.12.7.final.0-64 in 1847ms
  creator CPython3Posix(dest=/home/seonghapark/venvs/graphcore/poptorch33_env, clear=False, no_vcs_ignore=False, global=False)
  seeder FromAppData(download=False, pip=bundle, wheel=latest, via=copy, app_data_dir=/home/seonghapark/.local/share/virtualenv)
    added seed packages: pip==25.1.1
  activators BashActivator,CShellActivator,FishActivator,NushellActivator,PowerShellActivator,PythonActivator
```
There is CLIs to install PopTorch-3.3.0, but it seems that is already installed.

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
```

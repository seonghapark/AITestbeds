# What did I do:

## Set environment:
### What ALCF suggested to do:
[ALCF suggests](https://docs.alcf.anl.gov/ai-testbed/cerebras/customizing-environment/) to use and/or install cerebras_pytorch==2.4.0, but cerebras-modelzoo requires cerebras_pytorch==2.5.0, so that does not work.

### So what I really did:
```
mkdir ~/R_2.4.0
cd ~/R_2.4.0
/software/cerebras/python3.8/bin/python3.8 -m venv venv_cerebras_pt
source venv_cerebras_pt/bin/activate
pip install --upgrade pip
pip install --editable git+https://github.com/Cerebras/modelzoo#egg=cerebras_modelzoo 'murmurhash==1.0.10' 'thinc==8.2.2' 'cymem<2.0.10'
```

The last line uninstalled and installed things:
```
  Attempting uninstall: cerebras-appliance
    Found existing installation: cerebras-appliance 2.4.0
    Uninstalling cerebras-appliance-2.4.0:
      Successfully uninstalled cerebras-appliance-2.4.0
  Attempting uninstall: cerebras_pytorch
    Found existing installation: cerebras-pytorch 2.4.0
    Uninstalling cerebras-pytorch-2.4.0:
      Successfully uninstalled cerebras-pytorch-2.4.0
  Attempting uninstall: cerebras_modelzoo
    Found existing installation: cerebras-modelzoo 2.5.0
    Uninstalling cerebras-modelzoo-2.5.0:
      Successfully uninstalled cerebras-modelzoo-2.5.0
  Running setup.py develop for cerebras_modelzoo
Successfully installed cerebras-appliance-2.5.0 cerebras_modelzoo cerebras_pytorch-2.5.0
```

## Running a Pytorch sample
Refered [documents provided from ALCF](https://docs.alcf.anl.gov/ai-testbed/cerebras/running-a-model-or-program/#running-jobs-on-the-wafer):
```
mkdir ~/R_2.4.0
cd ~/R_2.4.0
git clone https://github.com/Cerebras/modelzoo.git
cd modelzoo
git tag
```

And the git repo's `tag`s are
```
R_1.6.0
R_1.6.1
R_1.7.0
R_1.7.1
Release_1.8.0
Release_1.9.1
Release_2.0.2
Release_2.0.3
Release_2.1.0
Release_2.1.1
Release_2.2.0
Release_2.2.1
Release_2.3.0
Release_2.3.1
Release_2.4.0
Release_2.4.3
Release_2.5.0
```

ALCF suggested to use `git checkout Release_2.4.0`, but I noticed the words `cerebras-pytorch==2.5.0` during ENV setting; which I guess I need to use `Release_2.5.0`.







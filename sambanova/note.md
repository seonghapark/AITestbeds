# What did I do?

The SambaNova DataScale SN30 system is architected around the next-generation Reconfigurable Dataflow Unit (RDU) processor for optimal dataflow processing and acceleration. The AI Testbed's SambaNova SN30 system consists of eight nodes in 4 full racks, each node featuring eight RDUs interconnected to enable model and data parallelism. SambaFlow, Sambanova's software stack, extracts, optimizes, and maps the dataflow graphs to the RDUs from standard machine learning frameworks like PyTorch.

## Log in to Login Node and Sambonova Node:
```
ssh ALCFUserID@sambanova.alcf.anl.gov
ssh sn30-r[1-4]-h[1-2]
```
where 'r' stands for the rack number, and 'h' stands for host. sn30-r1-h1 is the first host of the first rack.

## SDK setup
The required software environment (SambaFlow software stack and the associated environmental variables) for a SN30 node is set up automatically at login.
This is unlike the SN10 where the environment had to be set up by each user.


## Virtual Environments
To create a virtual environment, one can use the --system-site-packages flag:
```
python -m venv --system-site-packages my_env
source my_env/bin/activate
```
And there is nothing like `my_env/bin/deactive`. Then how to deactivate it? Just exit the kernel?

### Installing Packages
Install packages in the normal manner such as:
```
python3 -m pip install <package>
```
To install a different version of a package that is already installed in one's environment, one can use:
```
pip install --ignore-installed  ... # or -I
```

### Pre-Built Sample Venv
Each of the samples or application examples provided by SambaNova has its own pre-built virtual environment which can be readily used. They are present in the /opt/sambaflow/apps/ directory tree within each of the applications.
Note: Conda is not supported on the SambaNova system.

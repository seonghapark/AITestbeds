# Walking through basics:

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
conda deactivate
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


## Example Programs
You can use the link to the tutorials on the SambaNova GitHub site or the examples on the compute node (as explained below).
Find the tutorials on the SambaNova GitHub site. If you use those instructions, ensure that you still use the steps for accessing the SN compute node, setting the required environment and compiling and running the applications as described in this documentation.

Use the examples of well-known simple AI applications under the path: `/opt/sambaflow/apps/starters`, on all SambaNova compute nodes, as discussed on this page.
```
cd ~/
mkdir apps
cp -r /opt/sambaflow/apps/starters apps/starters
```
**Deactivate any active conda environment.** If you have conda installed and a conda environment is active, you will see something like (base) at the beginning of the command prompt. If so, you will need to deactivate it with conda deactivate. **Conda is not used on the SambaNova SN30 cluster.**

### Running LeNet example
```console
cd ~/apps/starters/lenet
# if it is first time training the model
srun python lenet.py compile -b=1 --pef-name="lenet" --output-folder="pef"
# if not, then run
srun python lenet.py run --pef="pef/lenet/lenet.pef"
```
You may check the status of your job with
```
csctl get jobs
```
For every new model, you need to compile it. The compiled artifact is usually cached if you run it multiple times, otherwise a fresh compilation is needed.
```
srun python lenet.py compile -b=1 --pef-name="lenet" --output-folder="pef"
```

### [SambaNova Model Zoo samples](https://docs.alcf.anl.gov/ai-testbed/sambanova/example-modelzoo-programs/):

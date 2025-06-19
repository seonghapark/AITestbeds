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
```
cd ~/apps/starters/lenet
pip install --upgrade pip
pip install fsspec==2024.6.1
pip install numpy==1.26.4
pip install pillow==10.4.0
pip install pandas==2.0.0
pip install torchvision
pip install samba

srun python lenet.py compile -b=1 --pef-name="lenet" --output-folder="pef"
srun python lenet.py run --pef="pef/lenet/lenet.pef"
```

Then it messaged:
```console
Traceback (most recent call last):
File "/home/seonghapark/sambanova_apps/starters/lenet/lenet.py", line 14, in <module>
import sambaflow.samba.utils as utils
ModuleNotFoundError: No module named 'sambaflow'
srun: error: sn30-r1-h1: task 0: Exited with exit code 1
srun: Terminating job step 55851.0
# And
Traceback (most recent call last):
File "/home/seonghapark/sambanova_apps/starters/lenet/lenet.py", line 14, in <module>
import sambaflow.samba.utils as utils
ModuleNotFoundError: No module named 'sambaflow'
srun: error: sn30-r1-h1: task 0: Exited with exit code 1
srun: Terminating job step 55851.0
```


### Alternatively to use Slurm sbatch,
```
mkdir -p pef/lenet
sbatch --output=pef/lenet/output.log submit-lenet-job.sh
```
And create submit-lenet-job.sh with the following contents:
```
#!/bin/sh

python lenet.py compile -b=1 --pef-name="lenet" --output-folder="pef"
python lenet.py run --pef="pef/lenet/lenet.pef"
```

Squeue will give you the queue status.
```
squeue
# One may also...
watch squeue
```
One may see the run log using: `cat pef/lenet/output.log`

### [SambaNova Model Zoo samples](https://docs.alcf.anl.gov/ai-testbed/sambanova/example-modelzoo-programs/):

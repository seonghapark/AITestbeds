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

GroqFlow is the simplest way to port applications running inference to groq.
```
cd ~/
git clone https://github.com/groq/groqflow.git
cd groqflow
export PYTHON_VERSION=3.10.12
cond
```
Then:
```
Retrieving notices: done
Channels:
 - defaults
Platform: linux-64
Collecting package metadata (repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /home/seonghapark/anaconda3/envs/groqflow
  added / updated specs:
    - python=3.10.12

The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    libxcb-1.17.0              |       h9b100fa_0         430 KB
    pthread-stubs-0.3          |       h0ce48e5_1           5 KB
    python-3.10.12             |       h955ad1f_0        26.8 MB
    setuptools-78.1.1          |  py310h06a4308_0         1.7 MB
    tk-8.6.14                  |       h993c535_1         3.4 MB
    wheel-0.45.1               |  py310h06a4308_0         115 KB
    xorg-libx11-1.8.12         |       h9b100fa_1         895 KB
    xorg-libxau-1.0.12         |       h9b100fa_0          13 KB
    xorg-libxdmcp-1.1.5        |       h9b100fa_0          19 KB
    xorg-xorgproto-2024.1      |       h5eee18b_1         580 KB
    ------------------------------------------------------------
                                           Total:        33.9 MB

The following NEW packages will be INSTALLED:

  _libgcc_mutex      pkgs/main/linux-64::_libgcc_mutex-0.1-main 
  _openmp_mutex      pkgs/main/linux-64::_openmp_mutex-5.1-1_gnu 
  bzip2              pkgs/main/linux-64::bzip2-1.0.8-h5eee18b_6 
  ca-certificates    pkgs/main/linux-64::ca-certificates-2025.2.25-h06a4308_0 
  ld_impl_linux-64   pkgs/main/linux-64::ld_impl_linux-64-2.40-h12ee557_0 
  libffi             pkgs/main/linux-64::libffi-3.4.4-h6a678d5_1 
  libgcc-ng          pkgs/main/linux-64::libgcc-ng-11.2.0-h1234567_1 
  libgomp            pkgs/main/linux-64::libgomp-11.2.0-h1234567_1 
  libstdcxx-ng       pkgs/main/linux-64::libstdcxx-ng-11.2.0-h1234567_1 
  libuuid            pkgs/main/linux-64::libuuid-1.41.5-h5eee18b_0 
  libxcb             pkgs/main/linux-64::libxcb-1.17.0-h9b100fa_0 
  ncurses            pkgs/main/linux-64::ncurses-6.4-h6a678d5_0 
  openssl            pkgs/main/linux-64::openssl-3.0.16-h5eee18b_0 
  pip                pkgs/main/noarch::pip-25.1-pyhc872135_2 
  pthread-stubs      pkgs/main/linux-64::pthread-stubs-0.3-h0ce48e5_1 
  python             pkgs/main/linux-64::python-3.10.12-h955ad1f_0 
  readline           pkgs/main/linux-64::readline-8.2-h5eee18b_0 
  setuptools         pkgs/main/linux-64::setuptools-78.1.1-py310h06a4308_0 
  sqlite             pkgs/main/linux-64::sqlite-3.45.3-h5eee18b_0 
  tk                 pkgs/main/linux-64::tk-8.6.14-h993c535_1 
  tzdata             pkgs/main/noarch::tzdata-2025b-h04d1e81_0 
  wheel              pkgs/main/linux-64::wheel-0.45.1-py310h06a4308_0 
  xorg-libx11        pkgs/main/linux-64::xorg-libx11-1.8.12-h9b100fa_1 
  xorg-libxau        pkgs/main/linux-64::xorg-libxau-1.0.12-h9b100fa_0 
  xorg-libxdmcp      pkgs/main/linux-64::xorg-libxdmcp-1.1.5-h9b100fa_0 
  xorg-xorgproto     pkgs/main/linux-64::xorg-xorgproto-2024.1-h5eee18b_1 
  xz                 pkgs/main/linux-64::xz-5.6.4-h5eee18b_1 
  zlib               pkgs/main/linux-64::zlib-1.2.13-h5eee18b_1 

Downloading and Extracting Packages:                                                                                                                                                                                                          
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#      
# To activate this environment, use
#      
#     $ conda activate groqflow
#      
# To deactivate an active environment, use
#
#     $ conda deactivate
```

And then:
```
conda activate groqflow
```

## Install groqflow into the groqflow conda environment
Execute the following commands to install groqflow into the activated groqflow conda environment
```
cd ~/groqflow
pip install -e .
# Technically this is pip install -e with no argument for -e.
# It provides a help string listing usages of the command
# (see http://docopt.org/ for examples of this syntax), and does nothing.
```
Then has an error:
```
(groqflow) seonghapark@gc-poplar-04:~/groqflow/proof_points/natural_language_processing/minilm$ pip3 install groq-devtools
ERROR: Could not find a version that satisfies the requirement groq-devtools (from versions: none)
ERROR: No matching distribution found for groq-devtools
(groqflow) seonghapark@gc-poplar-04:~/groqflow/proof_points/natural_language_processing/minilm$ cd ../../../
(groqflow) seonghapark@gc-poplar-04:~/groqflow$ pip install .
Processing /home/seonghapark/groqflow
  Installing build dependencies ... done
  Getting requirements to build wheel ... error
  error: subprocess-exited-with-error
  
  × Getting requirements to build wheel did not run successfully.
  │ exit code: 1
  ╰─> [15 lines of output]
      Traceback (most recent call last):
        File "/home/seonghapark/anaconda3/envs/groqflow/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 389, in <module>
          main()
        File "/home/seonghapark/anaconda3/envs/groqflow/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 373, in main
          json_out["return_val"] = hook(**hook_input["kwargs"])
        File "/home/seonghapark/anaconda3/envs/groqflow/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 143, in get_requires_for_build_wheel
          return hook(config_settings)
        File "/mnt/localdata/pip-build-env-42kfy4ul/overlay/lib/python3.10/site-packages/setuptools/build_meta.py", line 331, in get_requires_for_build_wheel
          return self._get_build_requires(config_settings, requirements=[])
        File "/mnt/localdata/pip-build-env-42kfy4ul/overlay/lib/python3.10/site-packages/setuptools/build_meta.py", line 301, in _get_build_requires
          self.run_setup()
        File "/mnt/localdata/pip-build-env-42kfy4ul/overlay/lib/python3.10/site-packages/setuptools/build_meta.py", line 317, in run_setup
          exec(code, locals())
        File "<string>", line 3, in <module>
      PermissionError: [Errno 13] Permission denied: 'groqflow/version.py'
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: subprocess-exited-with-error

× Getting requirements to build wheel did not run successfully.
│ exit code: 1
╰─> See above for output.

note: This error originates from a subprocess, and is likely not a problem with pip.
```



## Running a groqflow sample
Run a sample using PBS in batch mode (See Job Queueing and Submission for more information about the PBS job scheduler.)
Create a script `run_minilmv2.sh` with the following contents. It assumes that conda was installed in the default location. The conda initialize section can also be copied from your .bashrc if the conda installer was allowed to add it.
```
cd ~/groqflow/proof_points/natural_language_processing/minilm
pip install -r requirements.txt
python minilmv2.py
```
Then run the script as a batch job with PBS. This will reserve a full eight-card(chip) node.
```
qsub -l  select=1,place=excl run_minilmv2.sh
```
Note: the number of chips used by a model can be found in the compile cache dir for the model after it is compiled. E.g.
```
$ grep num_chips_used ~/.cache/groqflow/minilmv2/minilmv2_state.yaml
num_chips_used: 1
The groqflow proofpoints models use 1, 2 or 4 chips.
If your ~/.bashrc initializes conda, an alternative to copying the conda initilization script into your execution scripts is to comment out this section in your "~/.bashrc":

# If not running interactively, don't do anything
case $- in
    *i*) ;;
      *) return;;
esac
```
to
```
## If not running interactively, don't do anything
#case $- in
#    *i*) ;;
#      *) return;;
#esac
```
Then the execution script becomes:
```
#!/bin/bash
conda activate groqflow
cd ~/groqflow/proof_points/natural_language_processing/minilm
pip install -r requirements.txt
python minilmv2.py
Job status can be tracked with qstat:
```
$ qstat
Job id            Name             User              Time Use S Queue
----------------  ---------------- ----------------  -------- - -----
3084.groq-r01-co* run_minilmv2     user              0 R workq           
$ 
```
Output will by default go to two files with names like the following, where the suffix is the job id. One standard output for the job. The other is the standard error for the job.
```
$ ls -la run_minilmv2.sh.*
-rw------- 1 user users   448 Oct 16 18:40 run_minilmv2.sh.e3082
-rw------- 1 user users 50473 Oct 16 18:42 run_minilmv2.sh.o3082
```
***Run a sample using PBS in interactive mode***

An alternative is to use an interactive PBS job. This may be useful when debugging new or changed code. Here is an example that starts a 24 hour interactive job. It reserves a full eight-card(chip) node.
```
qsub -IV -l walltime=24:00:00 -l select=1,place=excl
```
Then activate your groqflow environment, and run python scripts with
```
conda activate groqflow
python scriptname.py
```

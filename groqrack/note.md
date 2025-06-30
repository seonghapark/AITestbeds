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

Running GroqFlow conda environments
```
conda activate groqflow
```

Run a sample using PBS in batch mode (See Job Queueing and Submission for more information about the PBS job scheduler.)
Create a script `run_minilmv2.sh` with the following contents. It assumes that conda was installed in the default location. The conda initialize section can also be copied from your .bashrc if the conda installer was allowed to add it.
```
#!/bin/bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$(${HOME}'/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
        . "${HOME}/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="${HOME}/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate groqflow
cd ~/groqflow/proof_points/natural_language_processing/minilm
pip install -r requirements.txt
python minilmv2.py
```

Then run the script as a batch job with PBS. This will reserve a full eight-card(chip) node.
```
qsub -l  select=1,place=excl run_minilmv2.sh
```

**Note:** the number of chips used by a model can be found in the compile cache dir for the model after it is compiled. E.g.
```
$ grep num_chips_used ~/.cache/groqflow/minilmv2/minilmv2_state.yaml
num_chips_used: 1
```

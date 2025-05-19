# What did I do to run LM at Sambanova

## I found this after installing Anaconda:
[Conda is not supported on the SambaNova system.](https://docs.alcf.anl.gov/ai-testbed/sambanova/virtual-environment/)
## Install Anaconda
Because I heard that ALCF does not allow to use docker container (maybe not), I installed conda:
```
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash ~/Anaconda3-2024.10-1-Linux-x86_64.sh
cd anaconda3/
source bin/activate
conda init --all
```

After initializing conda, below lines are added in the `.bashrc`:
```
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/seonghapark/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/seonghapark/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/seonghapark/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/seonghapark/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<
```

## Follow the ALCF git repo
And because I don't know where to start, I followed the git repository that Farah shared after cloned the repo:
```
git clone https://github.com/argonne-lcf/LLM-Inference-Bench.git
cd LLM-Inference-Bench/
conda create -n SN40L python=3.11
conda activate SN40L

git clone https://github.com/sambanova/ai-starter-kit.git
cd ai-starter-kit/benchmarking/
pip3 install -r requirements.txt 
```

I think the repo let me follow methods provided by sambanova, by cloning one of their git repo.

And tryied to create an `.env` file as the instruction in the git repo that Farah shared with parameters:
```
SAMBASTUDIO_BASE_URL="https://sjc3-e3.sambanova.net"
SAMBASTUDIO_BASE_URI="api/predict/generic"
SAMBASTUDIO_PROJECT_ID=<>
SAMBASTUDIO_ENDPOINT_ID=<>
SAMBASTUDIO_API_KEY=<>
```

and tryied to run
```bash run_synthetic_dataset.sh```,
but terminated with following error message:

```
]Traceback (most recent call last):
  File "/home/seonghapark/farah/LLM-Inference-Bench/Sambaflow/SN40L/ai-starter-kit/benchmarking/src/evaluator.py", line 352, in <module>
    main()
  File "/home/seonghapark/farah/LLM-Inference-Bench/Sambaflow/SN40L/ai-starter-kit/benchmarking/src/evaluator.py", line 257, in main
    synthetic_evaluator.run_benchmark(
  File "/home/seonghapark/farah/LLM-Inference-Bench/Sambaflow/SN40L/ai-starter-kit/benchmarking/src/performance_evaluation.py", line 913, in run_benchmark
    summary, individual_responses = self.get_token_throughput_latencies(
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/seonghapark/farah/LLM-Inference-Bench/Sambaflow/SN40L/ai-starter-kit/benchmarking/src/performance_evaluation.py", line 1138, in get_token_throughput_latencies
    raise Exception(
Exception: Unexpected error happened when executing requests:                
- Error while running LLM API requests. Check your model name, LLM API type, env variables and endpoint status.                

Additional messages:
- 401 Client Error: UNAUTHORIZED for url: https://api.sambanova.ai/v1/chat/completions
```

And so I followed README in the sambanova's repo saying add following in `.env`:
```
SAMBASTUDIO_URL="https://api-stage.sambanova.net/api/predict/nlp/12345678-9abc-def0-1234-56789abcdef0/456789ab-cdef-0123-4567-89abcdef0123"
SAMBASTUDIO_API_KEY="89abcdef-0123-4567-89ab-cdef01234567"
```
but receieved the same error mesasage.

### Answer from ALCF:
```
Hi, You are trying to run this code on SN30 which does not support it.
The github instructions were for a SN40L cloud instance.
We recently deployed the SN40L system on prem in the AI Testbed that will be made available to users soon.
I would suggest you wait for a few weeks and once you have access to this machine, give it a try then.
```

So move to other way to run models.
+) And based on the information on answer from ALCF, Farah performed her test on Sambanova Cloud.

## Follow SambaNova documentation [to run language example applications](https://docs.sambanova.ai/developer/latest/run-examples-language.html)
In this documentation, the first thing that they say is
```
To prepare your environment, you:
- **check your SambaFlow installation**
- make a copy of the tutorial files
- download the data files from the internet
```

This `sambaflow` is not install in the Sambanova, and users do not allow to install things, therefore, move to other way to run models.

## Trying things from scratch
I started from generating a new conda environment, `conda create -n test_env1`, and installed required libraries that Sambanova stated in their (repo)[https://github.com/sambanova/ai-starter-kit/blob/main/benchmarking/requirements.txt]. But because I am not sure if the global environment of the Sambanova machine of ALCF is the same as that of when the Sambanova poeple uploaded the information, I removed versions of the libs.

### From the installation of the libs:
The ALCF's Sambanova supports python3.13.8, which is not compatible with any version of sentencepiece (python version is too high).
Can use models using `Transformers`, but that is not stand-alone-way, but access to HuggingFace Cloud to process.


## Note:
- Usually, I cannot use ALCF machine on Mondays (sometimes Tuesdays too), because of regular maintenance of the system.
- (From Google AI Overview) Sambanova uses a custom AI chip called the **SN40L Reconfigurable Dataflow Unit (RDU)**, not a traditional GPU.
- Sambanova requies SambaFlow to run ML models: SambaFlow automatically extracts, optimizes, and executes the optimal dataflow graph of any of your models on Sambanova's RDUs. This enables you to achieve out-of-box performance, accuracy, scale, and easy of use.


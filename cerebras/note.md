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
  DEPRECATION: Legacy editable install of cerebras_modelzoo from git+https://github.com/Cerebras/modelzoo#egg=cerebras_modelzoo (setup.py develop) is deprecated. pip 25.1 will enforce this behaviour change. A possible replacement is to add a pyproject.toml or enable --use-pep517, and use setuptools >= 64. If the resulting installation is not behaving as expected, try using --config-settings editable_mode=compat. Please consult the setuptools documentation for more information. Discussion can be found at https://github.com/pypa/pip/issues/11457
```

Tried to install lower version of **`cerebras_modelzoo`**, but the system said **there is no other version**.

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

**So I stayed in the `main`** and when I install libs in requirements.txt, it said requirements are already satisfied.
```
pip install -r ~/R_2.4.0/modelzoo/requirements.txt
cd ~/R_2.4.0/modelzoo/src/cerebras/modelzoo/models/nlp/gpt3
cp /software/cerebras/dataset/OWT/Pytorch/111m_modified.yaml configs/Cerebras_GPT/111m_modified.yaml
```

### To run the sample:
```
export MODEL_DIR=model_dir_gpt3_111m
# the removal of the model_dir is only needed if sample has been previously run
if [ -d "$MODEL_DIR" ]; then rm -Rf $MODEL_DIR; fi
python run.py CSX --job_labels name=gpt3_111m --params configs/Cerebras_GPT/111m_modified.yaml --num_csx=1 --mode train --model_dir $MODEL_DIR --mount_dirs /home/ /software --python_paths /home/$(whoami)/R_2.4.0/modelzoo/src --compile_dir $(whoami) |& tee mytest.log
```

And then errors:
```
(venv_cerebras_pt) (base) [seonghapark@cer-login-01 gpt3]$ python run.py CSX --job_labels name=gpt3_111m --params configs/Cerebras_GPT/111m_modified.yaml --num_csx=1 --mode train --model_dir $MODEL_DIR --mount_dirs /home/ /software --python_paths /home/$(whoami)/R_2.4.0/modelzoo/src --compile_dir $(whoami) |& tee mytest.log
run.py:27: UserWarning: Running models using run.py is deprecated. Please switch to using the ModelZoo CLI. See https://training-docs.cerebras.ai/model-zoo/cli-overview for more details.
  warnings.warn(
The cache for model files in Transformers v4.22.0 has been updated. Migrating your old cache. This is a one-time only operation. You can interrupt this and resume the migration later on by calling `transformers.utils.move_cache()`.
0it [00:00, ?it/s]
/home/seonghapark/R_2.4.0/venv_cerebras_pt/lib/python3.8/site-packages/pydantic/_internal/_generate_schema.py:404: UserWarning: [<class 'int'>, <class 'int'>] is not a Python type (it may be an instance of an object), Pydantic will allow any object with no validation since we cannot even enforce that the input is an instance of the given type. To get rid of this error wrap the type with `pydantic.SkipValidation`.
  warn(
/home/seonghapark/R_2.4.0/venv_cerebras_pt/src/cerebras-modelzoo/src/cerebras/modelzoo/trainer/validate.py:758: UserWarning: Failed to validate params:
.
.
.
Traceback (most recent call last):
  File "run.py", line 32, in <module>
    run()
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/src/cerebras-modelzoo/src/cerebras/modelzoo/common/run_utils.py", line 65, in run
    main(
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/src/cerebras-modelzoo/src/cerebras/modelzoo/common/run_utils.py", line 122, in main
    return run_trainer(mode, params)
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/src/cerebras-modelzoo/src/cerebras/modelzoo/trainer/utils.py", line 179, in run_trainer
    configs = validate_trainer_params(params)
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/src/cerebras-modelzoo/src/cerebras/modelzoo/trainer/validate.py", line 749, in validate_trainer_params
    return construct_multi_phase_trainer_config(
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/lib/python3.8/site-packages/pydantic/type_adapter.py", line 142, in wrapped
    return func(self, *args, **kwargs)
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/lib/python3.8/site-packages/pydantic/type_adapter.py", line 373, in validate_python
    return self.validator.validate_python(object, strict=strict, from_attributes=from_attributes, context=context)
pydantic_core._pydantic_core.ValidationError: 2 validation errors for function-before[unpack_trainer(), tuple[TrainerConfig]]
0.init.model.config.mixed_precision
  Extra inputs are not permitted [type=extra_forbidden, input_value=True, input_type=bool]
    For further information visit https://errors.pydantic.dev/2.8/v/extra_forbidden
0.init.model.config.fp16_type
  Extra inputs are not permitted [type=extra_forbidden, input_value='bfloat16', input_type=str]
    For further information visit https://errors.pydantic.dev/2.8/v/extra_forbidden
```
Which seems the configuration file does not match with---

So changed branch to release_2.4.0 and tried to `pip install -r requirements.txt` and:
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
cerebras-modelzoo 2.5.0 requires cerebras_pytorch==2.5.0, but you have cerebras-pytorch 2.4.0 which is incompatible.
cerebras-modelzoo 2.5.0 requires tokenizers==0.20.1, but you have tokenizers 0.19.1 which is incompatible.
cerebras-modelzoo 2.5.0 requires transformers==4.45.2, but you have transformers 4.40.0 which is incompatible.
```

Guessing it is difficult to find solution of them, I started over from the begining with LLaMA example because I saw [this](https://github.com/Cerebras/modelzoo/blob/main/src/cerebras/modelzoo/tutorials/pretraining/model_config.yaml) in Cerebras's git repo.

### SO:
```
git checkout Release_2.5.0
pip install -r ~/R_2.4.0/modelzoo/requirements.txt
cd ~/R_2.4.0/modelzoo/src/cerebras/modelzoo/models/nlp/llama
cp /software/cerebras/dataset/OWT/Pytorch/params_llama_7b.yaml configs/params_llama_7b.yaml

```

Then an error:
```
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/src/cerebras-modelzoo/src/cerebras/modelzoo/trainer/validate.py", line 696, in check
    raise KeyError("Model configuration must have a 'name' key.")
KeyError: "Model configuration must have a 'name' key."
```

So added `name: "llama"` in the configuration file, and then:
```
2025-06-13 17:29:11,781 INFO:   Running command: run.py CSX --job_labels name=llama_7b --params configs/params_llama_7b.yaml --num_csx=1 --mode train --model_dir model_dir_llama_7b --mount_dirs /home/ /software --python_paths /home/seonghapark/R_2.4.0/modelzoo/src --compile_dir seonghapark
run.py:27: UserWarning: Running models using run.py is deprecated. Please switch to using the ModelZoo CLI. See https://training-docs.cerebras.ai/model-zoo/cli-overview for more details.
  warnings.warn(
.
.
.
```

Because the warning message said **Running models using run.py is deprecated. Please switch to using the ModelZoo CLI. See [https://training-docs.cerebras.ai/model-zoo/cli-overview](https://training-docs.cerebras.ai/rel-2.4.0/model-zoo/cli-overview) for more details.** I think this means that the models are in their cloud and users just use API for inference?

```
mkdir pretraining_tutorial
cp modelzoo/src/cerebras/modelzoo/tutorials/pretraining/* pretraining_tutorial
cszoo data_preprocess run --config pretraining_tutorial/train_data_config.yaml
cszoo data_preprocess run --config pretraining_tutorial/valid_data_config.yaml
```

Then, probably download dataset:
```
INFO:/home/seonghapark/R_2.4.0/venv_cerebras_pt/src/cerebras-modelzoo/src/cerebras/modelzoo/data_preparation/data_preprocessing/preprocess_data.py:
Finished writing data to pretraining_tutorial/train_data. Args & outputs can be found at pretraining_tutorial/train_data/data_params.json.

INFO:/home/seonghapark/R_2.4.0/venv_cerebras_pt/src/cerebras-modelzoo/src/cerebras/modelzoo/data_preparation/data_preprocessing/preprocess_data.py:
Finished writing data to pretraining_tutorial/valid_data. Args & outputs can be found at pretraining_tutorial/valid_data/data_params.json.
```

And failed to run `cszoo fit pretraining_tutorial/model_config.yaml`:
```
(venv_cerebras_pt) (base) [seonghapark@cer-login-01 R_2.4.0]$ cszoo fit pretraining_tutorial/model_config.yaml
/home/seonghapark/R_2.4.0/venv_cerebras_pt/lib/python3.8/site-packages/pydantic/_internal/_generate_schema.py:404: UserWarning: [<class 'int'>, <class 'int'>] is not a Python type (it may be an instance of an object), Pydantic will allow any object with no validation since we cannot even enforce that the input is an instance of the given type. To get rid of this error wrap the type with `pydantic.SkipValidation`.
  warn(
/home/seonghapark/R_2.4.0/venv_cerebras_pt/src/cerebras-modelzoo/src/cerebras/modelzoo/trainer/validate.py:758: UserWarning: Failed to validate params:
{'trainer': {'init': {'model_dir': 'pretraining_tutorial/model',
.
.
.
  warn(
Traceback (most recent call last):
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/bin/cszoo", line 33, in <module>
    sys.exit(load_entry_point('cerebras-modelzoo', 'console_scripts', 'cszoo')())
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/src/cerebras-modelzoo/src/cerebras/modelzoo/cli/main.py", line 299, in main
    ModelZooCLI()
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/src/cerebras-modelzoo/src/cerebras/modelzoo/cli/main.py", line 232, in __init__
    args.func(args)
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/src/cerebras-modelzoo/src/cerebras/modelzoo/cli/main.py", line 247, in run_trainer
    run_trainer(args.mode, params)
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/src/cerebras-modelzoo/src/cerebras/modelzoo/trainer/utils.py", line 179, in run_trainer
    configs = validate_trainer_params(params)
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/src/cerebras-modelzoo/src/cerebras/modelzoo/trainer/validate.py", line 749, in validate_trainer_params
    return construct_multi_phase_trainer_config(
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/lib/python3.8/site-packages/pydantic/type_adapter.py", line 142, in wrapped
    return func(self, *args, **kwargs)
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/lib/python3.8/site-packages/pydantic/type_adapter.py", line 373, in validate_python
    return self.validator.validate_python(object, strict=strict, from_attributes=from_attributes, context=context)
pydantic_core._pydantic_core.ValidationError: 8 validation errors for function-before[unpack_trainer(), tuple[TrainerConfig]]
0.fit.train_dataloader.GptHDF5MapDataProcessor.config.data_dir.function-before[validate_path(), str]
  Value error, Path train_data does not exist. [type=value_error, input_value='train_data', input_type=str]
    For further information visit https://errors.pydantic.dev/2.8/v/value_error
0.fit.train_dataloader.GptHDF5MapDataProcessor.config.data_dir.list[function-before[validate_path(), str]]
  Input should be a valid list [type=list_type, input_value='train_data', input_type=str]
    For further information visit https://errors.pydantic.dev/2.8/v/list_type
0.fit.val_dataloader.0.GptHDF5MapDataProcessor.config.data_dir.function-before[validate_path(), str]
  Value error, Path valid_data does not exist. [type=value_error, input_value='valid_data', input_type=str]
    For further information visit https://errors.pydantic.dev/2.8/v/value_error
0.fit.val_dataloader.0.GptHDF5MapDataProcessor.config.data_dir.list[function-before[validate_path(), str]]
  Input should be a valid list [type=list_type, input_value='valid_data', input_type=str]
    For further information visit https://errors.pydantic.dev/2.8/v/list_type
0.validate.val_dataloader.GptHDF5MapDataProcessor.config.data_dir.function-before[validate_path(), str]
  Value error, Path valid_data does not exist. [type=value_error, input_value='valid_data', input_type=str]
    For further information visit https://errors.pydantic.dev/2.8/v/value_error
0.validate.val_dataloader.GptHDF5MapDataProcessor.config.data_dir.list[function-before[validate_path(), str]]
  Input should be a valid list [type=list_type, input_value='valid_data', input_type=str]
    For further information visit https://errors.pydantic.dev/2.8/v/list_type
0.validate_all.val_dataloaders.0.GptHDF5MapDataProcessor.config.data_dir.function-before[validate_path(), str]
  Value error, Path valid_data does not exist. [type=value_error, input_value='valid_data', input_type=str]
    For further information visit https://errors.pydantic.dev/2.8/v/value_error
0.validate_all.val_dataloaders.0.GptHDF5MapDataProcessor.config.data_dir.list[function-before[validate_path(), str]]
  Input should be a valid list [type=list_type, input_value='valid_data', input_type=str]
    For further information visit https://errors.pydantic.dev/2.8/v/list_type
```

**Anyway**
When YH helped me and followed the examples, he also faced the final error message that I saw before that made me to try Release-2.4.0:
```
(venv_cerebras_pt) (base) [seonghapark@cer-login-01 gpt3]$ python run.py CSX --job_labels name=gpt3_111m --params configs/Cerebras_GPT/111m_modified.yaml --num_csx=1 --mode train --model_dir $MODEL_DIR --mount_dirs /home/ /software --python_paths /home/$(whoami)/R_2.4.0/modelzoo/src --compile_dir $(whoami) |& tee mytest.log
Traceback (most recent call last):
  File "run.py", line 25, in <module>
    run()
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/src/cerebras-modelzoo/src/cerebras/modelzoo/common/run_utils.py", line 65, in run
    main(
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/src/cerebras-modelzoo/src/cerebras/modelzoo/common/run_utils.py", line 104, in main
    from cerebras.modelzoo.trainer.restartable_trainer import RestartableTrainer
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/src/cerebras-modelzoo/src/cerebras/modelzoo/trainer/__init__.py", line 20, in <module>
    from . import callbacks, loggers, utils
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/src/cerebras-modelzoo/src/cerebras/modelzoo/trainer/callbacks/__init__.py", line 34, in <module>
    from .checkpoint import (
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/src/cerebras-modelzoo/src/cerebras/modelzoo/trainer/callbacks/checkpoint.py", line 34, in <module>
    from cerebras.appliance.storage import StorageWriter
ModuleNotFoundError: No module named 'cerebras.appliance.storage'
```
And explained me that this is low level error that I cannot handle, so sent an email to ALCF about this version imcompatibility.


# Based on the answer from support@alcf.anl.gov:
```console
mkdir ~/R_2.4.0
cd ~/R_2.4.0
git clone https://github.com/Cerebras/modelzoo.git
cd modelzoo
git tag
git checkout Release_2.4.0
```
Then build the virtual environment

```console
mkdir ~/R_2.4.0
cd ~/R_2.4.0
# Note: "deactivate" does not actually work in scripts.
conda deactivate
rm -r venv_cerebras_pt
/software/cerebras/python3.8/bin/python3.8 -m venv venv_cerebras_pt
source venv_cerebras_pt/bin/activate
pip install --upgrade pip
pip install cerebras_pytorch==2.4.0
pip install -r modelzoo/requirements.txt
pip install 'murmurhash==1.0.10' 'thinc==8.2.2' 'cymem<2.0.10'
```

And tried to run an example:
```
cd modelzoo
pip install -r ~/R_2.4.0/modelzoo/requirements.txt
cd ~/R_2.4.0/modelzoo/src/cerebras/modelzoo/models/nlp/gpt3
cp /software/cerebras/dataset/OWT/Pytorch/111m_modified.yaml configs/Cerebras_GPT/111m_modified.yaml
 
export MODEL_DIR=model_dir_gpt3_111m
if [ -d "$MODEL_DIR" ]; then rm -Rf $MODEL_DIR; fi
python run.py CSX --job_labels name=gpt3_111m --params configs/Cerebras_GPT/111m_modified.yaml --num_csx=1 --mode train --model_dir $MODEL_DIR --mount_dirs /home/ /software --python_paths /home/$(whoami)/R_2.4.0/modelzoo/src --compile_dir $(whoami) |& tee mytest.log
```
 
**Then has error saying:**
```
(venv_cerebras_pt) [seonghapark@cer-login-02 gpt3]$ python run.py CSX --job_labels name=gpt3_111m --params configs/Cerebras_GPT/111m_modified.yaml --num_csx=1 --mode train --model_dir $MODEL_DIR --mount_dirs /home/ /software --python_paths /home/$(whoami)/R_2.4.0/modelzoo/src --compile_dir $(whoami) |& tee mytest.log
Traceback (most recent call last):
  File "run.py", line 23, in <module>
    from cerebras.modelzoo.common.run_utils import run
ModuleNotFoundError: No module named 'cerebras.modelzoo'
```

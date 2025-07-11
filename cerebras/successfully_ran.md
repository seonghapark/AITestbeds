# The models successfully trained
# LLaMA2 7B FP16:
- 7B parameters,
- 13.5G when FP16,
- 28G when FP32

## Configuration
```
# LLaMa v2 model, 7B parameters, max_seq_length 4096
# Based on: https://arxiv.org/pdf/2307.09288.pdf

trainer:
  init:
    backend:
      backend_type: CSX
    model_dir: ./model_dir
    seed: 1
    model:
      name: "llama"
      # Embedding
      vocab_size: 32000
      hidden_size: 4096
      position_embedding_type: rotary
      rotary_dim: 128
      share_embedding_weights: false
      max_position_embeddings: 4096
      embedding_dropout_rate: 0.0
      # Decoder
      num_hidden_layers: 32
      dropout_rate: 0.0
      layer_norm_epsilon: 1.0e-05
      norm_type: rmsnorm
      # Decoder - Attention
      num_heads: 32
      attention_type: scaled_dot_product
      attention_dropout_rate: 0.0
      use_projection_bias_in_attention: false
      use_ffn_bias_in_attention: false
      # Decoder - ffn
      filter_size: 11008
      nonlinearity: swiglu
      use_ffn_bias: false
      # Task-specific
      use_bias_in_output: false
      loss_scaling: num_tokens
      loss_weight: 1.0
      # Cerebras parameters
      mixed_precision: true
      fp16_type: cbfloat16
    optimizer:
      AdamW:
        betas:
        - 0.9
        - 0.95
        eps: 1.0e-05
        correct_bias: true
        weight_decay: 0.1
    schedulers:
    - SequentialLR:
        schedulers:
        - LinearLR:
            initial_learning_rate: 0.0
            end_learning_rate: 0.0003
            total_iters: 2000
        - CosineDecayLR:
            initial_learning_rate: 0.0003
            end_learning_rate: 3.0e-05
            total_iters: 474837
    precision:
      # Cerebras parameters
      enabled: true
      fp16_type: cbfloat16
      loss_scaling_factor: dynamic
      max_gradient_norm: 1.0
    loop:
      max_steps: 200 # 476837 # Llama v2 7B was trained on 2T tokens. # steps = 2T / (1024 * 4096)
      eval_frequency: 10000
      eval_steps: 89
    checkpoint:
      steps: 200
      save_initial_checkpoint: true
    logging:
      # steps = 2T / (1024 * 4096)
      log_steps: 50
  fit:
    train_dataloader:
      data_processor: GptHDF5MapDataProcessor
      data_dir: /software/datasets/llama_data_32K/test/
      shuffle: false
      shuffle_seed: 1
      batch_size: 1024
      num_workers: 8
      prefetch_factor: 10
      persistent_workers: true
    val_dataloader: &val_dataloader
      data_processor: GptHDF5MapDataProcessor
      data_dir: /software/datasets/llama_data_32K/test/
      shuffle: false
      shuffle_seed: 1
      batch_size: 1024
      num_workers: 8
      prefetch_factor: 10
      persistent_workers: true
  validate:
    val_dataloader: *val_dataloader
  validate_all:
    val_dataloaders: *val_dataloader
```
## CLIs:
```
export PYTHONPATH=/home/$(whoami)/R_2.4.0/modelzoo/src
 
export MODEL_DIR=model_dir_llama2_7b
if [ -d "$MODEL_DIR" ]; then rm -Rf $MODEL_DIR; fi
python run.py CSX --job_labels name=llama2_7b --params configs/params_llama2_7b.yaml --num_csx=1 --mode train --model_dir $MODEL_DIR --mount_dirs /projects /home/ /software --python_paths /home/$(whoami)/R_2.4.0/modelzoo/src  --compile_dir $(whoami) |& tee mytest.log
```
Then
```
2025-07-11 15:24:55,954 INFO:   Checkpoint autoloading is enabled. Looking for latest checkpoint in "model_dir_llama2_7b" directory with the following naming convention: `checkpoint_(step)(_timestamp)?.mdl`.
2025-07-11 15:24:55,955 INFO:   No checkpoints were found in "model_dir_llama2_7b".                                                             
2025-07-11 15:24:55,955 INFO:   No checkpoint was provided. Using randomly initialized model parameters.                                        
2025-07-11 15:24:55,975 INFO:   Starting training loop 1, from global step 0 to 200                                                             
2025-07-11 15:24:57,096 INFO:   Saving checkpoint at step 0
2025-07-11 15:27:53,583 INFO:   Saved checkpoint model_dir_llama2_7b/checkpoint_0.mdl                                                           
2025-07-11 15:28:09,423 INFO:   Compiling the model. This may take a few minutes.                                                               
2025-07-11 15:28:11,856 INFO:   Initiating a new image build job against the cluster server.                                                    
2025-07-11 15:28:11,863 INFO:   Custom worker image build is disabled from server. Falling back to venv mounting.                               
2025-07-11 15:28:12,047 INFO:   Initiating a new compile wsjob against the cluster server.                                                      
2025-07-11 15:28:12,058 INFO:   Compile job id: wsjob-duhl7jck7hjhbtxdpxvk67, remote log path: /n1/wsjob/workdir/job-operator/wsjob-duhl7jck7hjhbtxdpxvk67
2025-07-11 15:28:22,066 INFO:   Poll ingress status: Waiting for job running, current job status: Queueing, msg: job queueing to be scheduled. Job queue status: current job is top of queue but likely blocked by running jobs, 1 compile job(s) running using 50Gi memory. For more information, please run 'csctl get jobs'
```
And then after few hours of waiting until previous job is terminated:
```

2025-07-11 15:58:22,764 INFO:   Poll ingress status: Waiting for job running, current job status: Queueing, msg: job queueing to be scheduled. Job queue status: current job is top of queue but likely blocked by running jobs, 1 compile job(s) running using 50Gi memory. For more information, please run 'csctl get jobs'.
2025-07-11 16:28:23,465 INFO:   Poll ingress status: Waiting for job running, current job status: Queueing, msg: job queueing to be scheduled. Job queue status: current job is top of queue but likely blocked by running jobs, 1 compile job(s) running using 50Gi memory. For more information, please run 'csctl get jobs'.
2025-07-11 16:58:24,185 INFO:   Poll ingress status: Waiting for job running, current job status: Queueing, msg: job queueing to be scheduled. Job queue status: current job is top of queue but likely blocked by running jobs, 1 compile job(s) running using 50Gi memory. For more information, please run 'csctl get jobs'.
2025-07-11 17:28:24,945 INFO:   Poll ingress status: Waiting for job running, current job status: Queueing, msg: job queueing to be scheduled. Job queue status: current job is top of queue but likely blocked by running jobs, 1 compile job(s) running using 50Gi memory. For more information, please run 'csctl get jobs'.
2025-07-11 17:47:35,525 INFO:   Poll ingress status: Waiting for job running, current job status: Initializing, msg: job initializing with config generation/image pulling/....
2025-07-11 17:47:45,452 INFO:   Poll ingress status: Waiting for job ingress readiness.
2025-07-11 17:47:55,464 INFO:   Poll ingress success: Job ingress ready, dashboard: https://grafana.cerebras1.lab.alcf.anl.gov/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-duhl7jck7hjhbtxdpxvk67&from=1752255455000&to=now
2025-07-11 17:47:56,996 INFO:   Found existing cached compile with hash: "cs_16980171559998961274"
2025-07-11 17:48:04,241 INFO:   Compile artifacts successfully written to remote compile directory. Compile hash is: cs_16980171559998961274
2025-07-11 17:48:11,313 INFO:   Compile was successful!
2025-07-11 17:48:11,314 INFO:   Waiting for weight initialization to complete
2025-07-11 17:48:11,315 INFO:   Programming Cerebras Wafer Scale Cluster for execution. This may take a few minutes.
2025-07-11 17:48:11,484 INFO:   Initiating a new execute wsjob against the cluster server.
2025-07-11 17:48:11,510 INFO:   Execute job id: wsjob-7javbqdggsmrb44obfaqft, remote log path: /n1/wsjob/workdir/job-operator/wsjob-7javbqdggsmrb44obfaqft
2025-07-11 17:48:21,548 INFO:   Poll ingress status: Waiting for job running, current job status: Initializing, msg: job initializing with config generation/image pulling/....
2025-07-11 17:48:31,551 INFO:   Poll ingress status: Waiting for all Chief pods to be running, current running: 0/1.
2025-07-11 17:48:41,560 INFO:   Poll ingress status: Waiting for all Weight pods to be running, current running: 0/20.
2025-07-11 17:48:51,566 INFO:   Poll ingress status: Waiting for all Activation pods to be running, current running: 0/59.
2025-07-11 17:49:11,589 INFO:   Poll ingress status: Waiting for all Chief pods to be running, current running: 0/1.
2025-07-11 17:49:21,613 INFO:   Poll ingress status: Waiting for all Activation pods to be running, current running: 0/59.
2025-07-11 17:49:41,621 INFO:   Poll ingress status: Waiting for all Weight pods to be running, current running: 12/20.
2025-07-11 17:49:51,631 INFO:   Poll ingress status: Waiting for all Activation pods to be running, current running: 57/59.
2025-07-11 17:50:01,658 INFO:   Poll ingress success: Job ingress ready, dashboard: https://grafana.cerebras1.lab.alcf.anl.gov/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-7javbqdggsmrb44obfaqft&from=1752255503000&to=now
2025-07-11 17:50:01,905 INFO:   Preparing to execute using 1 CSX
2025-07-11 17:50:39,946 INFO:   About to send initial weights
/home/seonghapark/R_2.4.0/venv_cerebras_pt/lib/python3.8/site-packages/pydantic/_internal/_generate_schema.py:404: UserWarning: [<class 'int'>, <class 'int'>] is not a Python type (it may be an instance of an object), Pydantic will allow any object with no validation since we cannot even enforce that the input is an instance of the given type. To get rid of this error wrap the type with `pydantic.SkipValidation`.
  warn(
../../../../../cerebras/modelzoo/config/base_config.py:114: FutureWarning: Found deprecated fields for LlamaModelConfig: ['fp16_type', 'mixed_precision']
Support for passing these fields in will be removed in the future.
  warn(
2025-07-11 17:51:46,872 INFO:   Finished sending initial weights
2025-07-11 17:51:46,873 INFO:   Finalizing appliance staging for the run
2025-07-11 17:51:46,896 INFO:   Waiting for device programming to complete
2025-07-11 17:53:31,008 INFO:   Device programming is complete
2025-07-11 17:53:32,561 INFO:   Using network type: ROCE
2025-07-11 17:53:32,562 INFO:   Waiting for input workers to prime the data pipeline and begin streaming ...
2025-07-11 17:53:32,583 INFO:   Input workers have begun streaming input data
2025-07-11 17:53:33,940 INFO:   Appliance staging is complete
2025-07-11 17:53:33,940 INFO:   Beginning appliance run
E0711 18:04:02.654472867  609793 hpack_parser.cc:1234]       Error parsing metadata: error=invalid value key=content-type value=text/html
E0711 18:04:03.657033328  620785 hpack_parser.cc:1234]       Error parsing metadata: error=invalid value key=content-type value=text/html
E0711 18:04:05.657732039  608834 hpack_parser.cc:1234]       Error parsing metadata: error=invalid value key=content-type value=text/html
E0711 18:04:09.415502427  609793 hpack_parser.cc:1234]       Error parsing metadata: error=invalid value key=content-type value=text/html
E0711 18:04:15.954994608  608834 hpack_parser.cc:1234]       Error parsing metadata: error=invalid value key=content-type value=text/html
2025-07-11 18:04:19,669 ERROR:   Initiating shutdown sequence due to Appliance error: Ran into error while receiving tensor output_model_wise_params_norm for runtime iteration 10
2025-07-11 18:04:19,670 INFO:   Trying to fetch failure info from the cluster for job wsjob-7javbqdggsmrb44obfaqft. This may take up to 60 seconds.
2025-07-11 18:04:19,787 INFO:   wsjob-7javbqdggsmrb44obfaqft dashboard: https://grafana.cerebras1.lab.alcf.anl.gov/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-7javbqdggsmrb44obfaqft&from=1752255503000&to=1752257629000
2025-07-11 18:04:19,787 ERROR:   Job wsjob-7javbqdggsmrb44obfaqft failed due to: job has failed because total 1 replica(s) failed, first failed pods: [wsjob-7javbqdggsmrb44obfaqft-weight-3]
2025-07-11 18:04:19,787 WARNING:   Event 2025-07-11 18:03:47 +0000 UTC reason=Error object=wsjob-7javbqdggsmrb44obfaqft-weight-3 message='SUSPEND:              0
        RETIRE:               0

              total        used        free      shared     buffers       cache   available
Mem:           125G         94G        1.5G        4.0G          0B         29G         25G
Low:           125G        124G        1.5G
High:            0B          0B          0B
Swap:            0B          0B          0B
Total:         125G         94G        1.5G
[Pool Allocator Stats]
Total size of UNBACKED: 35176 MiB
Total size of BACKED_UNALLOCATED: 14468 MiB
Total size of BACKED_ALLOCATED: 8303 MiB
Total size of PEAK BACKED_ALLOCATED: 8319 MiB
Total BACKED: 22771 MiB
Total size of PEAK BACKED: 22771 MiB
Num growths: 0
Num shuffles: 0
Number of free page fragments: 284
Num pool segments: 1


terminate called after throwing an instance of 'std::runtime_error'
  what():  HIOStream::frame failed. bytes=-296, expected=466944
2025-07-11 18:03:47,814:INFO wsjob-7javbqdggsmrb44obfaqft-weight-3 main subprocess terminated by signal 6 (SIGABRT): Aborted, time: 2025-07-11T18:03:47.814734+00:00'
2025-07-11 18:04:19,788 WARNING:   Event 2025-07-11 18:03:48 +0000 UTC reason=Error object=wsjob-7javbqdggsmrb44obfaqft message='Error pod wsjob-7javbqdggsmrb44obfaqft-weight-15 container ws exitCode: 250 terminated reason/message: Error'
2025-07-11 18:04:19,788 WARNING:   Event 2025-07-11 18:03:48 +0000 UTC reason=Error object=wsjob-7javbqdggsmrb44obfaqft message='Error pod wsjob-7javbqdggsmrb44obfaqft-weight-3 container ws exitCode: 250 terminated reason/message: Error'
2025-07-11 18:04:19,788 WARNING:   Event 2025-07-11 18:03:49 +0000 UTC reason=Error object=wsjob-7javbqdggsmrb44obfaqft-weight-3 message='Pod: job-operator.wsjob-7javbqdggsmrb44obfaqft-weight-3 exited with code 250'
E0711 18:04:20.256250640  609792 hpack_parser.cc:1234]       Error parsing metadata: error=invalid value key=content-type value=text/html
2025-07-11 18:04:23,132 INFO:   Processed 11264 training sample(s) in 9567.159426912 seconds.
Traceback (most recent call last):
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/lib/python3.8/site-packages/cerebras/appliance/appliance_client.py", line 1108, in _recv_output_stream
    for response in output_stream:
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/lib/python3.8/site-packages/cerebras/appliance/utils/interceptors.py", line 34, in intercept_unary_stream
    raise PicklableRpcError.from_grpc_error(e) from None
cerebras.appliance.errors.PicklableRpcError: gRPC Error:
  Status Code: StatusCode.INTERNAL
  Details:
COUT030 18:04:18 GMT

Deadline Exceeded


  Metadata:
    date: Fri, 11 Jul 2025 18:04:19 GMT
    content-length: 0
    strict-transport-security: max-age=15724800; includeSubDomains


The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "run.py", line 27, in <module>
    run()
  File "../../../../../cerebras/modelzoo/common/run_utils.py", line 65, in run
    main(
  File "../../../../../cerebras/modelzoo/common/run_utils.py", line 119, in main
    return run_trainer(mode, params)
  File "../../../../../cerebras/modelzoo/trainer/utils.py", line 149, in run_trainer
    run_trainer(mode, config)
  File "../../../../../cerebras/modelzoo/trainer/utils.py", line 138, in run_trainer
    trainer.fit(train_dataloader, val_dataloader, config.fit.ckpt_path)
  File "../../../../../cerebras/modelzoo/trainer/trainer.py", line 721, in fit
    self._run_train(train_dataloader, loop, loop_idx)
  File "../../../../../cerebras/modelzoo/trainer/trainer.py", line 762, in _run_train
    for batch_idx, batch in enumerate(self.executor):
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/lib/python3.8/site-packages/cerebras/pytorch/utils/data/data_executor.py", line 303, in __iter__
    self.backend.on_batch_end()
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/lib/python3.8/site-packages/cerebras/pytorch/backend/ltc_backend.py", line 957, in on_batch_end
    self.run_step_closures()
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/lib/python3.8/site-packages/cerebras/pytorch/backend/ltc_backend.py", line 1222, in run_step_closures
    cpu_args, cpu_kwargs = torch.utils._pytree.tree_map(
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/lib/python3.8/site-packages/torch/utils/_pytree.py", line 948, in tree_map
    return treespec.unflatten(map(func, *flat_args))
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/lib/python3.8/site-packages/torch/utils/_pytree.py", line 787, in unflatten
    leaves = list(leaves)
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/lib/python3.8/site-packages/cerebras/pytorch/backend/ltc_backend.py", line 1224, in <lambda>
    self._get_cpu_tensor(arg)
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/lib/python3.8/site-packages/cerebras/pytorch/backend/ltc_backend.py", line 1139, in _get_cpu_tensor
    return cerebras_pytorch_lib.get_appliance_data(arg).tensor
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/lib/python3.8/site-packages/cerebras/pytorch/backend/ltc_backend.py", line 671, in get_tensor
    tensor = self.appliance.receive_output(iteration, name)
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/lib/python3.8/site-packages/cerebras/pytorch/core/appliance.py", line 164, in receive_output
    out = super().receive_output(iteration, name)
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/lib/python3.8/site-packages/cerebras/appliance/appliance_manager.py", line 754, in receive_output
    return self.grpc_client.recv_output(iteration, name)
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/lib/python3.8/site-packages/cerebras/appliance/appliance_client.py", line 649, in recv_output
    return _recv_output_stream(
  File "/home/seonghapark/R_2.4.0/venv_cerebras_pt/lib/python3.8/site-packages/cerebras/appliance/appliance_client.py", line 1130, in _recv_output_stream
    raise ApplianceUnknownError(
cerebras.appliance.errors.ApplianceUnknownError: Ran into error while receiving tensor output_model_wise_params_norm for runtime iteration 10

```

## Resources used while training (CPU, GPU, memory, power, throughput, training time)
## Performance while trainig (loss, accuracy)


# Vision Transformer Base FP16
- ViT-Base FP16: a patch size of 16x16, a hidden dimension of 768, 12 layers, and 12 attention heads,
- 86.8M parameters

And additional:
- ViT-Base/16: Often pre-trained on ImageNet-21k, with 86 million parameters. 
- ViT-Large/16: Trained on JFT-300M, with 307 million parameters. 
- ViT-22B: A very large model with 22 billion parameters. 
- DeiT (Data-efficient Image Transformers): These models are distilled versions of ViT, with variants like DeiT-tiny, DeiT-small, and DeiT-base. 
- BEiT (BERT pre-training of Image Transformers): These models use a self-supervised method inspired by BERT.

And..
- ViT-Base: 12 layers, 768 hidden size, 3072 MLP size.
- ViT-Large: 24 layers, 1024 hidden size, 4096 MLP size.
- ViT-Huge: 32 layers, 1280 hidden size, 5120 MLP size.
- ViT-22B: 22 billion parameters, according to Google Research.

## Configuration
```
trainer:
  init:
    seed: 1
    backend:
      cluster_config:
        num_workers_per_csx: 2
      backend_type: CSX
    model:
      name: vision_transformer
      num_classes: 1000
      position_embedding_type: learned
      embedding_dropout_rate: 0.1
      hidden_size: 768
      num_hidden_layers: 12
      layer_norm_epsilon: 1.0e-06
      num_heads: 12
      attention_type: scaled_dot_product
      attention_softmax_fp32: true
      dropout_rate: 0.1
      nonlinearity: gelu
      pooler_nonlinearity: tanh
      attention_dropout_rate: 0.1
      use_projection_bias_in_attention: true
      use_ffn_bias_in_attention: true
      filter_size: 3072
      use_ffn_bias: true
      initializer_range: 0.02
      norm_first: true
      image_size:
      - 224
      - 224
      num_channels: 3
      patch_size:
      - 16
      - 16
      use_conv_patchified_embedding: true
      use_encoder_pooler_layer: false
      prepend_cls_token: true
      mixed_precision: true
      fp16_type: bfloat16
    optimizer:
      Adam:
        betas:
        - 0.9
        - 0.999
        eps: 1.0e-08
        weight_decay: 0.0003
        correct_bias: true
    schedulers:
    - SequentialLR:
        schedulers:
        - LinearLR:
            initial_learning_rate: 0.0
            end_learning_rate: 0.0005
            total_iters: 14371
        - CosineDecayLR:
            initial_learning_rate: 0.0005
            end_learning_rate: 0.0
            total_iters: 117854
    precision:
      enabled: true
      fp16_type: bfloat16
      loss_scaling_factor: dynamic
      max_gradient_norm: 1.0
    loop:
      max_steps: 200 
      eval_frequency: 1000
    checkpoint:
      steps: 100
    logging:
      log_steps: 10
    callbacks:
    - ScopedTrainFlags:
        csx.performance.micro_batch_size: 2850
    - ScopedValidateFlags:
        csx.performance.micro_batch_size: 2850
  fit:
    train_dataloader:
      data_processor: ImageNet1KProcessor
      data_dir: /software/datasets/imagenet/ #./computer_vision/datasets/imagenet/imagenet1k_ilsvrc2012
      num_classes: 1000
      batch_size: 2850
      image_size:
      - 224
      - 224
      shuffle: true
      shuffle_seed: 42
      split: train
      transforms:
      - name: resize
        size:
        - 256
        - 256
      - name: random_resized_crop
        size:
        - 224
        - 224
        scale:
        - 0.08
        - 1.0
        ratio:
        - 0.75
        - 1.33
        interpolation: bilinear
      - name: random_horizontal_flip
        p: 0.5
      - name: to_tensor
      - name: normalize
        mean:
        - 0.5
        - 0.5
        - 0.5
        std:
        - 0.5
        - 0.5
        - 0.5
      num_workers: 2
      prefetch_factor: 2
      persistent_workers: true
      use_worker_cache: true
    val_dataloader: &id001
      data_processor: ImageNet1KProcessor
      data_dir: /software/datasets/imagenet/ #./computer_vision/datasets/imagenet/imagenet1k_ilsvrc2012
      num_classes: 1000
      batch_size: 2850
      image_size:
      - 224
      - 224
      shuffle: false
      shuffle_seed: 42
      split: val
      transforms:
      - name: resize
        size:
        - 224
        - 224
      - name: to_tensor
      - name: normalize
        mean:
        - 0.5
        - 0.5
        - 0.5
        std:
        - 0.5
        - 0.5
        - 0.5
      num_workers: 2
      prefetch_factor: 2
      persistent_workers: true
      use_worker_cache: true
  validate:
    val_dataloader: *id001
  validate_all:
    val_dataloaders: *id001
```
## CLIs:
```
export PYTHONPATH=/home/$(whoami)/R_2.4.0/modelzoo/src

export MODEL_DIR=model_dir_vt
if [ -d "$MODEL_DIR" ]; then rm -Rf $MODEL_DIR; fi

python run.py CSX --job_labels name=vision_transformer --params configs/params_vit_base_patch_16_imagenet_1k.yaml --num_csx=1 --mode train --model_dir $MODEL_DIR --mount_dirs /home/$(whoami)/ /software --python_paths /home/$(whoami)/R_2.4.0/modelzoo/src --compile_dir /$(whoami) |& tee mytest.log
```
Then
```
2025-07-11 16:03:20,229 INFO:   No need to use DLS for loss when half dtype is bfloat16. Disabling gradient scaling.                                            
2025-07-11 16:03:20,407 INFO:   Checkpoint autoloading is enabled. Looking for latest checkpoint in "model_dir_vt" directory with the following naming convention: `checkpoint_(step)(_timestamp)?.mdl`.
2025-07-11 16:03:20,409 INFO:   No checkpoints were found in "model_dir_vt".
2025-07-11 16:03:20,409 INFO:   No checkpoint was provided. Using randomly initialized model parameters.                                                        
2025-07-11 16:03:20,409 INFO:   Effective batch size is 2850.
2025-07-11 16:03:20,412 INFO:   The following sequence is used to transform data:                                                                               
Compose(
    Resize(size=[256, 256], interpolation=bilinear, max_size=None, antialias=None)                                                                              
    RandomResizedCrop(size=[224, 224], scale=(0.08, 1.0), ratio=(0.75, 1.33), interpolation=bilinear, antialias=True)                                           
    RandomHorizontalFlip(p=0.5)
    ToTensor()
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    LambdaWithParam(args=(torch.bfloat16,), kwargs={})
)
2025-07-11 16:03:39,742 INFO:   Starting training loop 1, from global step 0 to 200                                                                             
2025-07-11 16:03:40,073 WARNING:   Passing an absolute path as the compile directory may lead to undesirably long paths as the directory is used on the server side, not on the client side. Please consider passing in a relative directory instead.                                                                            
2025-07-11 16:04:09,694 INFO:   Compiling the model. This may take a few minutes.                                                                               
2025-07-11 16:04:12,663 INFO:   Initiating a new image build job against the cluster server.                                                                    
2025-07-11 16:04:12,670 INFO:   Custom worker image build is disabled from server. Falling back to venv mounting.                                               
2025-07-11 16:04:12,673 WARNING:   Passing an absolute path as the compile directory may lead to undesirably long paths as the directory is used on the server side, not on the client side. Please consider passing in a relative directory instead.                                                                            
2025-07-11 16:04:13,075 INFO:   Initiating a new compile wsjob against the cluster server.                                                                      
2025-07-11 16:04:13,086 INFO:   Compile job id: wsjob-ij4dsj8vhnv9utzu6jrdtr, remote log path: /n1/wsjob/workdir/job-operator/wsjob-ij4dsj8vhnv9utzu6jrdtr      
2025-07-11 16:04:23,093 INFO:   Poll ingress status: Waiting for job running, current job status: Queueing, msg: job queueing to be scheduled. Job queue status:
2 compile job(s) queued before current job. For more information, please run 'csctl get jobs'.
```
After few hours of waiting until previous job is terminated:
```

2025-07-11 16:21:03,446 INFO:   Poll ingress status: Waiting for job running, current job status: Queueing, msg: job queueing to be scheduled. Job queue status: 1 compile job(s) queued before current job. For more information, please run 'csctl get jobs'.
2025-07-11 16:51:04,100 INFO:   Poll ingress status: Waiting for job running, current job status: Queueing, msg: job queueing to be scheduled. Job queue status: 1 compile job(s) queued before current job. For more information, please run 'csctl get jobs'.
2025-07-11 17:21:04,756 INFO:   Poll ingress status: Waiting for job running, current job status: Queueing, msg: job queueing to be scheduled. Job queue status: 1 compile job(s) queued before current job. For more information, please run 'csctl get jobs'.
2025-07-11 17:47:35,437 INFO:   Poll ingress status: Waiting for job running, current job status: Queueing, msg: job queueing to be scheduled. Job queue status: current job is top of queue but likely blocked by running jobs, 1 compile job(s) running using 50Gi memory. For more information, please run 'csctl get jobs'.
2025-07-11 17:48:15,655 INFO:   Poll ingress status: Waiting for job running, current job status: Scheduled, msg: job scheduled and waiting to be initialized.
2025-07-11 17:48:25,486 INFO:   Poll ingress status: Waiting for all Coordinator pods to be running, current running: 0/1.
2025-07-11 17:48:35,492 INFO:   Poll ingress status: Waiting for job ingress readiness.
2025-07-11 17:48:55,509 INFO:   Poll ingress success: Job ingress ready, dashboard: https://grafana.cerebras1.lab.alcf.anl.gov/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-ij4dsj8vhnv9utzu6jrdtr&from=1752255504000&to=now
2025-07-11 17:48:56,173 INFO:   Found existing cached compile with hash: "cs_9892963798577744835"
2025-07-11 17:49:01,120 INFO:   Compile artifacts successfully written to remote compile directory. Compile hash is: cs_9892963798577744835
2025-07-11 17:49:09,845 INFO:   Compile was successful!
2025-07-11 17:49:09,846 INFO:   Waiting for weight initialization to complete
2025-07-11 17:49:09,846 INFO:   Programming Cerebras Wafer Scale Cluster for execution. This may take a few minutes.
2025-07-11 17:49:10,226 INFO:   Initiating a new execute wsjob against the cluster server.
2025-07-11 17:49:10,252 INFO:   Execute job id: wsjob-nephnzm89xrzwsjjqqsftd, remote log path: /n1/wsjob/workdir/job-operator/wsjob-nephnzm89xrzwsjjqqsftd
2025-07-11 17:49:20,264 INFO:   Poll ingress status: Waiting for job running, current job status: Queueing, msg: job queueing to be scheduled. Job queue status: current job is top of queue but likely blocked by running jobs, 1 execute job(s) running using 1 system(s) and 1 nodegroup(s)[1 pop and 0 depop]. For more information, please run
'csctl get jobs'.
2025-07-11 18:04:31,295 INFO:   Poll ingress status: Waiting for job running, current job status: Scheduled, msg: job scheduled and waiting to be initialized.
2025-07-11 18:04:41,113 INFO:   Poll ingress status: Waiting for all Chief pods to be running, current running: 0/1.
2025-07-11 18:05:01,143 INFO:   Poll ingress status: Waiting for all Weight pods to be running, current running: 0/24.
2025-07-11 18:05:11,141 INFO:   Poll ingress status: Waiting for all Activation pods to be running, current running: 0/48.
2025-07-11 18:05:41,177 INFO:   Poll ingress status: Waiting for all Weight pods to be running, current running: 2/24.
2025-07-11 18:05:51,183 INFO:   Poll ingress status: Waiting for all Weight pods to be running, current running: 20/24.
2025-07-11 18:06:01,205 INFO:   Poll ingress success: Job ingress ready, dashboard: https://grafana.cerebras1.lab.alcf.anl.gov/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-nephnzm89xrzwsjjqqsftd&from=1752256476000&to=now
2025-07-11 18:06:01,392 INFO:   Preparing to execute using 1 CSX
2025-07-11 18:06:40,335 INFO:   About to send initial weights
/home/seonghapark/R_2.4.0/venv_cerebras_pt/lib/python3.8/site-packages/pydantic/_internal/_generate_schema.py:404: UserWarning: [<class 'int'>, <class 'int'>] is not a Python type (it may be an instance of an object), Pydantic will allow any object with no validation since we cannot even enforce that the input is an instance of the given
type. To get rid of this error wrap the type with `pydantic.SkipValidation`.
  warn(
../../../../../cerebras/modelzoo/config/base_config.py:114: FutureWarning: Found deprecated fields for VisionTransformerModelConfig: ['fp16_type', 'mixed_precision']
Support for passing these fields in will be removed in the future.
  warn(
../../../../../cerebras/modelzoo/trainer/validate.py:117: UserWarning: Adam got 1 unexpected and unused parameters: ['correct_bias'].
Please ensure that you specified the correct parameters:
Adam(params=[], lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0003, amsgrad=False)
Passing in unused parameters is deprecated behaviour and support for it will be removed in a future release.
  warn(
../../../../../cerebras/modelzoo/config/base_config.py:114: FutureWarning: Found deprecated fields for ImageNet1KProcessorConfig: ['num_classes']
Support for passing these fields in will be removed in the future.
  warn(
2025-07-11 18:06:49,651 INFO:   Finished sending initial weights
2025-07-11 18:06:49,652 INFO:   Finalizing appliance staging for the run
2025-07-11 18:07:09,595 INFO:   Waiting for device programming to complete
2025-07-11 18:09:16,255 INFO:   Device programming is complete
2025-07-11 18:09:17,134 INFO:   Using network type: ROCE
2025-07-11 18:09:17,135 INFO:   Waiting for input workers to prime the data pipeline and begin streaming ...
2025-07-11 18:10:18,040 INFO:   Input workers have begun streaming input data
2025-07-11 18:10:19,468 INFO:   Appliance staging is complete
2025-07-11 18:10:19,468 INFO:   Beginning appliance run
2025-07-11 18:10:57,392 INFO:   | Train Device=CSX, Step=10, Loss=7.01949, Rate=753.24 samples/sec, GlobalRate=753.24 samples/sec
2025-07-11 18:11:37,060 INFO:   | Train Device=CSX, Step=20, Loss=7.03425, Rate=732.38 samples/sec, GlobalRate=735.44 samples/sec
2025-07-11 18:12:30,250 INFO:   | Train Device=CSX, Step=30, Loss=6.99695, Rate=614.44 samples/sec, GlobalRate=654.20 samples/sec
2025-07-11 18:13:09,187 INFO:   | Train Device=CSX, Step=40, Loss=6.99425, Rate=684.94 samples/sec, GlobalRate=672.04 samples/sec
2025-07-11 18:13:58,801 INFO:   | Train Device=CSX, Step=50, Loss=6.97235, Rate=618.64 samples/sec, GlobalRate=649.96 samples/sec
2025-07-11 18:14:40,385 INFO:   | Train Device=CSX, Step=60, Loss=6.94855, Rate=658.67 samples/sec, GlobalRate=655.60 samples/sec
2025-07-11 18:15:28,712 INFO:   | Train Device=CSX, Step=70, Loss=6.93150, Rate=617.31 samples/sec, GlobalRate=645.31 samples/sec
2025-07-11 18:16:09,574 INFO:   | Train Device=CSX, Step=80, Loss=6.90237, Rate=665.40 samples/sec, GlobalRate=651.40 samples/sec
2025-07-11 18:16:55,981 INFO:   | Train Device=CSX, Step=90, Loss=6.87574, Rate=634.64 samples/sec, GlobalRate=647.03 samples/sec
2025-07-11 18:17:33,907 INFO:   | Train Device=CSX, Step=100, Loss=6.88158, Rate=704.73 samples/sec, GlobalRate=656.15 samples/sec
2025-07-11 18:17:33,910 INFO:   Saving checkpoint at step 100
2025-07-11 18:17:54,424 INFO:   Saved checkpoint model_dir_vt/checkpoint_100.mdl
2025-07-11 18:18:20,062 INFO:   | Train Device=CSX, Step=110, Loss=6.83740, Rate=652.38 samples/sec, GlobalRate=652.44 samples/sec
2025-07-11 18:18:58,726 INFO:   | Train Device=CSX, Step=120, Loss=6.83568, Rate=703.23 samples/sec, GlobalRate=658.74 samples/sec
2025-07-11 18:19:44,313 INFO:   | Train Device=CSX, Step=130, Loss=6.82083, Rate=656.40 samples/sec, GlobalRate=656.03 samples/sec
2025-07-11 18:20:22,072 INFO:   | Train Device=CSX, Step=140, Loss=6.79728, Rate=715.43 samples/sec, GlobalRate=662.22 samples/sec
2025-07-11 18:21:09,154 INFO:   | Train Device=CSX, Step=150, Loss=6.81465, Rate=649.36 samples/sec, GlobalRate=658.10 samples/sec
2025-07-11 18:21:48,269 INFO:   | Train Device=CSX, Step=160, Loss=6.79875, Rate=696.92 samples/sec, GlobalRate=662.10 samples/sec
2025-07-11 18:22:37,343 INFO:   | Train Device=CSX, Step=170, Loss=6.78806, Rate=627.22 samples/sec, GlobalRate=656.69 samples/sec
2025-07-11 18:23:16,739 INFO:   | Train Device=CSX, Step=180, Loss=6.76104, Rate=684.94 samples/sec, GlobalRate=660.08 samples/sec
2025-07-11 18:24:08,021 INFO:   | Train Device=CSX, Step=190, Loss=6.75545, Rate=607.43 samples/sec, GlobalRate=653.62 samples/sec
2025-07-11 18:24:46,556 INFO:   | Train Device=CSX, Step=200, Loss=6.72398, Rate=686.73 samples/sec, GlobalRate=657.44 samples/sec
2025-07-11 18:24:46,559 INFO:   Saving checkpoint at step 200
2025-07-11 18:25:06,987 INFO:   Saved checkpoint model_dir_vt/checkpoint_200.mdl
2025-07-11 18:25:27,397 INFO:   Training completed successfully!
2025-07-11 18:25:27,522 INFO:   Processed 570000 training sample(s) in 8507.781435492 seconds.
8542.796370267868
```
## Resources used while training (CPU, GPU, memory, power, throughput, training time)
## Performance while trainig (loss, accuracy)


# Name (number of parameters)
## Configuration for the training (input dataset, epochs, batch size, optimizer, and so on)
## CLIs for training them
## Resource used while training (CPU, GPU, memory, power, throughput, training time)
## Performance while training (loss, accuracy)

# The models successfully used for inference
# Name (number of parameters)
## Configuration for the inference (size of input, and what else?)
## CLIs for inference
## Resource used while training (CPU, GPU, memory, power, throughput, inference time)
## Performance while training (loss, accuracy)

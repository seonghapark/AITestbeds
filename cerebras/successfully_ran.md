# The models successfully trained
# LLaMA2:
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

## Resources used while training (CPU, GPU, memory, power, throughput, training time)
## Performance while trainig (loss, accuracy)


# Vision Transformer
- ViT-Base FP16: a patch size of 16x16, a hidden dimension of 768, 12 layers, and 12 attention heads,
- 86.8M parameters
Additiona:
- ViT-Base/16: Often pre-trained on ImageNet-21k, with 86 million parameters. 
- ViT-Large/16: Trained on JFT-300M, with 307 million parameters. 
- ViT-22B: A very large model with 22 billion parameters. 
- DeiT (Data-efficient Image Transformers): These models are distilled versions of ViT, with variants like DeiT-tiny, DeiT-small, and DeiT-base. 
- BEiT (BERT pre-training of Image Transformers): These models use a self-supervised method inspired by BERT. 

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

# The models successfully trained
## LLaMA
## Configuration
## CLIs:
```
cp  /software/cerebras/dataset/vision_transformer/params_vit_base_patch_16_imagenet_1k.yaml configs/params_vit_base_patch_16_imagenet_1k.yaml

export MODEL_DIR=model_dir_vt
if [ -d "$MODEL_DIR" ]; then rm -Rf $MODEL_DIR; fi

python run.py CSX --job_labels name=vision_transformer --params configs/params_vit_base_patch_16_imagenet_1k.yaml --num_csx=1 --mode train --model_dir $MODEL_DIR --mount_dirs /home/$(whoami)/ /software --python_paths /home/$(whoami)/R_2.4.0/modelzoo/src --compile_dir /$(whoami) |& tee mytest.log
```
## Resources used while training (CPU, GPU, memory, power, throughput, training time)
## Performance while trainig (loss, accuracy)


## Name (number of parameters)
## Configuration for the training (input dataset, epochs, batch size, optimizer, and so on)
## CLIs for training them
## Resource used while training (CPU, GPU, memory, power, throughput, training time)
## Performance while training (loss, accuracy)

# The models successfully used for inference
## Name (number of parameters)
## Configuration for the inference (size of input, and what else?)
## CLIs for inference
## Resource used while training (CPU, GPU, memory, power, throughput, inference time)
## Performance while training (loss, accuracy)

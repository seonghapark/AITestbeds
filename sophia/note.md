# What did I do?
**Sophia is not an AI testbed**

Sophia is comprised of 24 NVIDIA DGX A100 nodes. Each DGX A100 node comprises eight NVIDIA A100 Tensor Core GPUs and two AMD Rome CPUs that provide 22 nodes with 320 GB of GPU memory and two nodes with 640 GB of GPU memory (8,320 GB in total) for training artificial intelligence (AI) datasets, while also enabling GPU-specific and -enhanced high-performance computing (HPC) applications for modeling and simulation.
A 15-terabyte solid-state drive offers up to 25 gigabits per second in bandwidth. The dedicated compute fabric comprises 20 Mellanox QM9700 HDR200 40-port switches wired in a fat-tree topology.


## Logging into Sophia:
```
ssh <username>@sophia.alcf.anl.gov
```
Then, type in the password from your CRYPTOCard/MobilePASS+ token. Once logged in, you land on one of the Sophia login nodes (sophia-login-01, sophia-login-02).

## Containers on Sophia: Apptainer

Sophia employs Apptainer (formerly known as Singularity) for container management. To set up Apptainer, run:
```
module use /soft/spack/base/0.7.1/install/modulefiles/Core/
module load apptainer
apptainer version #1.3.3
```

### Building from Docker or Argonne GitHub Container Registry
Containers on Sophia can be built by writing Dockerfiles on a local machine and then publishing the container to DockerHub, or by directly building them on an ALCF compute node by writing an Apptainer recipe file. If you prefer to use existing containers, you can pull them from various registries like DockerHub and run them on Sophia.
Since Docker requires root privileges, which users do not have on Sophia, existing Docker containers must be converted to Apptainer. To build a Docker-based container on Sophia, use the following as an example:
```
qsub -I -A <Project> -l select=1:ngpus=8:ncpus=256 -l walltime=01:00:00 -l filesystems=home:eagle -l singularity_fakeroot=True -q by-node -k doe
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
module use /soft/spack/base/0.7.1/install/modulefiles/Core/
module load apptainer
apptainer build --fakeroot pytorch:22.06-py3.sing docker://nvcr.io/nvidia/pytorch:22.06-py3
```
You can find the latest prebuilt NVIDIA PyTorch containers here. The TensorFlow containers are here (though note that LCF doesn't typically prebuild the TF-1 containers). You can search the full container registry here. For custom containers tailored for Sophia, visit ALCF's GitHub container registry.
Note: Currently, container build and executions are only supported on the Sophia compute nodes.

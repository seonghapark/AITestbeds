# What did I do?
**Sophia is not an AI testbed**

Sophia is comprised of 24 NVIDIA DGX A100 nodes. Each DGX A100 node comprises eight NVIDIA A100 Tensor Core GPUs and two AMD Rome CPUs that provide 22 nodes with 320 GB of GPU memory and two nodes with 640 GB of GPU memory (8,320 GB in total) for training artificial intelligence (AI) datasets, while also enabling GPU-specific and -enhanced high-performance computing (HPC) applications for modeling and simulation.
A 15-terabyte solid-state drive offers up to 25 gigabits per second in bandwidth. The dedicated compute fabric comprises 20 Mellanox QM9700 HDR200 40-port switches wired in a fat-tree topology.


## Logging into Sophia:
```
ssh <username>@sophia.alcf.anl.gov
```
Then, type in the password from your CRYPTOCard/MobilePASS+ token. Once logged in, you land on one of the Sophia login nodes (sophia-login-01, sophia-login-02).



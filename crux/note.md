# What did I do?
**Crux is not an AI testbed.**

Crux is an HPE Cray EX Liquid Cooled system with a peak performance of 1.18 PF, comprised of 64 compute blades connected via Slingshot. Each blade has 4 compute nodes for a total of 256 nodes in the system. Each compute node has dual AMD EPYC 7742 64-Core Processors. Each CPU core supports up to two hyperthreads for a total of 256 threads possible per node. Each CPU has 128 GB of DDR4 memory for a total of 256 GB per node.

## Logging Into Crux
To log into Crux:
```
ssh <username>@crux.alcf.anl.gov
Then, type in the password from your CRYPTOCard/MobilePASS+ token. Once logged in, you land on one of the Crux login nodes (crux-login-01, crux-login-02).
```


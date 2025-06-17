# What did I do:
ALCF's Groq system consists of a single GroqRackTM compute cluster that provides an extensible accelerator network consisting of 9 GroqNodeTM (groq-r01-gn-01 through groq-r01-gn-09) nodes with a rotational multi-node network topology. Each of these GroqNodes consists of 8 GroqCardTM accelerators in them with integrated chip-to-chip connections with a dragonfly multi-chip topology.
GroqCardTM accelerator is a dual-width, full-height, three-quarter length PCI-Express Gen4 x16 adapter that includes a single GroqChipTM processor with 230 MB of on-chip memory. Based on the proprietary Tensor Streaming Processor (TSP) architecture, the GroqChip processor is a low latency and high throughput single core SIMD compute engine capable of 750 TOPS (INT8) and 188 TFLOPS (FP16) @ 900 MHz that includes advanced vector and matrix mathematical acceleration units. The GroqChip processor is deterministic, providing predictable and repeatable performance.

## Log in to a Login node and GroqRack node
ssh into the device:
```
ssh ALCFUserID@gc-login-02.ai.alcf.anl.gov
ssh gc-poplar-04.ai.alcf.anl.gov
```

If you expect a loss of an internet connection for any reason, for long-running jobs we suggest logging into a specific node and using either screen or tmux to create persistent command line sessions.
```
man screen
# or
man tmux
```

## Running jobs on Groq nodes using GroqFlow
GroqFlow is the simplest way to port applications running inference to groq.
```
cd ~/
git clone https://github.com/groq/groqflow.git
cd groqflow
export PYTHON_VERSION=3.10.12
conda create -n groqflow python=$PYTHON_VERSION -y
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


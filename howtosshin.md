# How to ssh into AI Testbed Machines:
https://docs.alcf.anl.gov/ai-testbed/graphcore/getting-started/

## Log in to Graphcore:
Connection to a Graphcore node is a two-step process.
The first step is to ssh from a local machine to the login node.
The second step is to log in to a Graphcore node from the login node.

### Log in to Login Node:
```
ssh ALCFUserID@gc-login-01.ai.alcf.anl.gov
# or
ssh ALCFUserID@gc-login-02.ai.alcf.anl.gov
```

### Log in to a Graphcore Node:
```
ssh gc-poplar-02.ai.alcf.anl.gov
# or
ssh gc-poplar-03.ai.alcf.anl.gov
# or
ssh gc-poplar-04.ai.alcf.anl.gov
```

## Log in to Cerebras:
Either ssh to `cerebras.ai.alcf.anl.gov`, which randomly resolves to one of cer-login-0[1-3].ai.alcf.anl.gov,
or ssh to a specific node, `cer-login-01.ai.alcf.anl.gov`, `cer-login-02.ai.alcf.anl.gov`, `cer-login-03.ai.alcf.anl.gov`.


## Log in to Groq:
Connection to a Groq node is a two-step process.
The first step is to ssh from a local machine to the login node.
The second step is to log in to a Groq node from the login node.

### Log in to Login Node:
```
ssh ALCFUserID@groq.ai.alcf.anl.gov
```
This randomly selects one of the login nodes, namely `groq-login-01.ai.alcf.anl.gov` or `groq-login-02.ai.alcf.anl.gov`.
You can alternatively ssh to the specific login nodes directly.

### Log in to a GroqRack node:
Once you are on a login node, optionally ssh to one of the GroqRack nodes, which are numbered 1-9.
```
ssh groq-r01-gn-01.ai.alcf.anl.gov
# or
ssh groq-r01-gn-09.ai.alcf.anl.gov
# or any node with hostname of form groq-r01-gn-0[1-9].ai.alcf.anl.gov
```

## Log in to Grux:

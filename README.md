DPmixGGM
========

This folder contains source codes for the "GPU-powered Stochastic Shotgun Search for Dirichlet proces mixtures of Gaussian Graphical Models" 
by Chiranjit Mukherjee and Abel Rodriguez. Implementation and testing are done using CUDA SDK 4.0 on a platform of compute capability 1.3. 
When compiling on a different platform please set "ROOTDIR" and "CUFILES_sm_**" compatiable with the installed CUDA SDK version and compute 
capability of the platform.

The "DPmixGGM_SSS_main.cu" file contains tuning parameters for the algorithm, as elaborated below:
1. Run the SSS or the MCMC by enabling either of the macros SSS and MCMC while disabling the other.
2. Run GPU/CPU versions of the SSS by enabling / disabling the macro CUDA.
3. Specify maximum number of mixture components that the model should accommodate (for pre-allocation of memory).
4. Set SSS runtime parameters C, D, R, S, M, g, h, f, t.
5. Set SSS number of chain parameters. User needs to provide at least one initial point.
6. Set MCMC runtime parameters.
7. Set hyperparameters of for the prior on (mu, K | G) with N0, DELTA0.

Complie source codes using the "make" command and run with "release/DPmixGGM_SSS.exe f9_n150_p50 3" command. Here 3 refers to maximum number of 
GPUs to engage for SSS when CUDA is enabled.

The program expects an input-data file (e.g. f9_n150_p50) in the DATA/ folder and at least one initialization point (e.g. f9_n150_p50_init1). 
The input-data file should specify n and p in the first row and then provide n rows of length p. The initial point data-file should specify n, p
and L of the initial model configuration in the first row and xi-indices of the initial point in the second row. Subsequent L rows specify 
G_l (l=1:L).

A list of highest-score models is stored in folder RES/.

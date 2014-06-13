// Chiranjit Mukherjee  (chiranjit@soe.ucsc.edu)

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define SSS										// If uncommented, runs the Stochastic Shotgun Search
//#define MCMC									// If uncommented, runs the Markov chain Monte Carlo
												// Note: Need at lease one of the above two uncommented
//#define CUDA									// If uncommented, runs CUDA kernels on GPU
#define MAXL 10									// Maximum number of mixture components that can be accommodated

#ifdef SSS
// SSS- runtime parameters
#define maxLocalWastedIterations (n+p)			// In paper, C
#define climbDownStepSize 10					// In paper, D
#define maxLocalJumpCount 10					// In paper, R
#define MAXNGLOBALJUMP 2						// In paper, S / (C * R)
// SSS- parameters for lists of models saved
#define sizeOfFeatureSelectionList 20			// In paper, M
#define sizeOfBestList 100						// Number of highest-score models to keep track of
// SSS- 
#define LOCALMOVE_SFACTOR 0.001
#define GLOBALJUMP_SFACTOR 0.01
#define G_TO_XI int(L*p*(p-1)/(2*n))			// In paper, g
#define XI_TO_SM 10								// In paper, h
#define LOOKFORWARD 5							// In paper, f
#define RGMS_T 2								// In paper, t
// number of chains parameters
#define N_INIT 3								// Number of points of initial models provided by the user in folder DATA/
#define TRY_EACH_INIT 1							// Number of times to restart from each given initial point
#define N_RANDOM_RESTART 3						// Number of times to restart from random random initial points
#define N_MODES_LIST_RESTART 3					// Number of times to start from
#define maxNmodes ((TRY_EACH_INIT*N_INIT+N_RANDOM_RESTART+N_MODES_LIST_RESTART)+1)
#endif

#ifdef MCMC
// MCMC- runtime parameters
#define BURNIN 20000							// Burn-in
#define N_ITR  100000							// Number of iterations to run after burn-in
#ifdef CUDA
#undef CUDA
#endif
#endif

#define PI             3.1415926
#define log_2          0.693147180559945
#define log_pi_over_4  0.286182471462350
#define log_2_pi       1.837877066409345
#define NEG_INF        -999999.0
#define myBool bool
#define myInt short								// Using short interger
#define myIntFactor 2
#define intFactor 4
//#define Real double
#define Real float								// Using floating-point
#define ISFLOAT 10
using namespace std;

#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf.h>
#define GSL_INTEGRATION_GRIDSIZE 1000
gsl_integration_workspace * w; gsl_function F;

#include <gsl/gsl_randist.h>
#define RANDOMSEED calendar_time

// Define hyperparameters for the prior distribution of (mu, K | G)
#define N0 0.01									
#define DELTA0 3
#define JEFFREYS_PRIOR
gsl_rng *rnd;
  
#ifdef CUDA
  #include <cuda.h>
  #include <cutil.h>
  #include <cuda_runtime_api.h>
  #include <cuda_runtime.h>
  #include <device_launch_parameters.h>
  #define BLOCKSIZ 32
  #define SYNC __syncthreads()
typedef struct {
  cudaStream_t delete_stream; cudaStream_t add_stream;  
  myInt* d_in_delete; myInt* d_in_add; myInt* d_which_delete; myInt* d_which_add;
  myInt* h_in_delete; myInt* h_in_add; myInt* which_delete; myInt* which_add;
  int n_add, n_delete;
} MGPUstuff;
#else
typedef struct {
} MGPUstuff;
#endif
MGPUstuff* device; int n_devices;
  
// Include source files
#include "utilities.cpp"
#ifndef GRAPH_CPP
#include "graph.cpp"
#endif
#ifndef GWISH_CPP
#include "gwish.cpp"
#endif
#ifndef DPMIXGGM_CPP
#include "DPmixGGM.cpp"
#endif
#ifndef LISTS_CPP
#include "DPmixGGM_Lists.cpp"
#endif
#ifndef SSSMOVES_CPP
#include "DPmixGGM_SSSmoves.cpp"
#endif
#ifdef MCMC
#include "DPmixGGM_MCMCmoves.cpp"
#endif

//////////////////////////////////////////////////////////////// START OF MAIN ///////////////////////////////////////////////////////////////  

int main (int argc, char *argv[])
{
  // declarations and initialisations
  int i,j,l,q,r,t; int L = 2; long int k; Real score; char initID[] = {'1','2','3'}; clock_t start, now; double cpu_time;
  
  // Initializing gsl random variate generators and integration tools
  const gsl_rng_type *T; time_t calendar_time; gsl_rng_env_setup(); T = gsl_rng_default; rnd = gsl_rng_alloc (T); calendar_time = time(NULL); gsl_rng_set(rnd,RANDOMSEED);
  #ifdef SSS
  unsigned long int seedset[maxNmodes]; for(i=0; i<maxNmodes; i++) { seedset[i] = gsl_rng_get (rnd); }
  #endif
  
  w = gsl_integration_workspace_alloc (GSL_INTEGRATION_GRIDSIZE);
  
  // DATA INPUT
  char datafile[50] = ""; strcpy(datafile,"DATA/"); strcat(datafile,argv[1]); strcat(datafile,".txt"); ifstream data(datafile);
  int n, p; data >> n; data >> p; printf("%d %d\n",n,p); Real *X = new Real[n*p]; for(i=0; i<n; i++) { for(j=0; j<p; j++) { data >> X[p*i+j]; } }; data.close();
    
  // more declarations and initialisations
  int ee = p*(p-1)/2;
  
////////////////////////////////////////////////////////////// START OF SSS ///////////////////////////////////////////////////////////////  
#ifdef SSS
  
  // OUTPUT FILES
  char outfile[100] = ""; strcpy(outfile,"RES/");
  #ifndef CUDA  
  strcat(outfile,argv[1]); strcat(outfile,"_modes_CPU.txt"); ofstream outmodes(outfile); outmodes << n << " " << p << endl;
  #ifndef CUDA
  strcpy(outfile,"RES/"); strcat(outfile,argv[1]); strcat(outfile,"_best_CPU.txt"); ofstream outbest(outfile); outbest << n << " " << p << endl;
  #endif
  #else
  strcat(outfile,argv[1]); strcat(outfile,"_modes_GPU.txt"); ofstream outmodes(outfile); outmodes << n << " " << p << endl;
  #ifndef CUDA
  strcpy(outfile,"RES/"); strcat(outfile,argv[1]); strcat(outfile,"_best_GPU.txt"); ofstream outbest(outfile); outbest << n << " " << p << endl;
  #endif
  #endif
  
  // Initialisations
  State initstates[N_INIT+N_RANDOM_RESTART]; int* initstateID = new int[maxNmodes];
  for(i=0; i<N_INIT; i++)
  {	strcpy(datafile,"DATA/"); strcat(datafile,argv[1]); strcat(datafile,"_init"); strncat(datafile,&initID[i],1); strcat(datafile,".txt");
	ifstream initfile(datafile); initstates[i] = new DPmixGGM(X,L,n,p,0.1,initfile); initfile.close(); initstateID[i] = i;
  }
  
  State state           = new DPmixGGM(initstates[0]);
  State localBestState  = new DPmixGGM(state);
  State globalBestState = new DPmixGGM(state);
  List  featureList = new DPmixGGMlist (sizeOfFeatureSelectionList, n, p);
  List  modesList   = new DPmixGGMlist (maxNmodes, n, p);
#ifdef CUDA
  List  bestList    = (List) NULL;
#else
  List  bestList    = new DPmixGGMlist (sizeOfBestList, n, p);
#endif  
  
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  #ifdef CUDA
  myInt req_GPU = atoi(argv[2]); cudaGetDeviceCount(&n_devices); n_devices = ((n_devices <= req_GPU) ? n_devices : req_GPU); device = new MGPUstuff[n_devices]; 
  size_t size_temp; cudaError_t e1;
  for(r=0; r<n_devices; r++) {				cudaSetDevice(r);  
  cudaStreamCreate(&(device[r].delete_stream));		cudaStreamCreate(&(device[r].add_stream));  
  size_temp = sizeof(myInt)*(3+p+p*p+2*ee);		e1 = cudaMalloc((void**) &(device[r].d_in_delete), size_temp); 	  if(e1 != cudaSuccess) { cout << "Error." << endl; exit(0); }
  size_temp = sizeof(myInt)*(3+4*p+2*p*p+2*ee);		e1 = cudaMalloc((void**) &(device[r].d_in_add), size_temp); 	  if(e1 != cudaSuccess) { cout << "Error." << endl; exit(0); }  
  size_temp = sizeof(myInt)*ee;				e1 = cudaMalloc((void**) &(device[r].d_which_delete), size_temp); if(e1 != cudaSuccess) { cout << "Error." << endl; exit(0); }
  size_temp = sizeof(myInt)*ee;				e1 = cudaMalloc((void**) &(device[r].d_which_add), size_temp);	  if(e1 != cudaSuccess) { cout << "Error." << endl; exit(0); }  
  device[r].h_in_delete = new myInt[3+p+p*p+2*ee]; device[r].h_in_add = new myInt[4+4*p+2*p*p+2*ee]; device[r].which_delete = new myInt[ee]; device[r].which_add = new myInt[ee];
  }
  #endif
  
  // more declarations and initialisations
  bool globalMoveFlag = 0; myInt nmodes = 1; Real localBestScore = NEG_INF, globalBestScore = NEG_INF; gsl_rng_set(rnd,seedset[nmodes-1]);
  int wastedIterations = 0; int localJumpCount = 0, globalJumpCount = 0; int num_cases; long int num_allModels = 0; 
  
  // initial xi scan
  num_cases += updateAllXis (1, state, bestList); L = state->L; score = state->plp; for(l=0; l<L; l++) { score += state->pll[l]; }
  printf("%ld %d %.4f %.4f %.4f %d %.4f %d %ld\n",k,state->L,score,localBestScore,globalBestScore,nmodes,cpu_time,num_cases,num_allModels);
     
  // start the stopwatch
  start = clock(); k = 0; 
  while(nmodes<=maxNmodes)
  {   k++; num_cases = 0;

      // LOCAL MOVES ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
      if((k%G_TO_XI)) { num_cases += updateOneEdgeInEveryG (state->L, NULL, 0, state->graphlist, state->pll, NULL, state, bestList); }
      else
      { j = k/G_TO_XI;
	if(j%XI_TO_SM) { if(state->L>1) { num_cases += updateAllXis (1, state, bestList); num_cases += Merge (state, bestList, LOOKFORWARD, 0); } }
	else           { num_cases += splitMerge(state, featureList, bestList, LOOKFORWARD, LOCALMOVE_SFACTOR, 0, 1, RGMS_T); }
      }
      // LOCAL MOVES ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

      // MODE BREAK MOVES //////////////////////////////////////////////////////////////////////////////////////////////////////////
      if ((wastedIterations > maxLocalWastedIterations) && (localJumpCount < maxLocalJumpCount))
      {   wastedIterations = 0; localJumpCount++; state->CopyState(localBestState);  
	  
	  // local graph jump
	  for(i=0; i<localJumpCount; i++)
	  {	num_cases += updateOneEdgeInEveryG (state->L, NULL, (i+1)*climbDownStepSize, state->graphlist, state->pll, NULL, state, bestList);
	  }
      }
      // MODE BREAK MOVES //////////////////////////////////////////////////////////////////////////////////////////////////////////
      
      // GLOBAL JUMP MOVES /////////////////////////////////////////////////////////////////////////////////////////////////////////
      if(wastedIterations > maxLocalJumpCount*maxLocalWastedIterations)
      { if(globalJumpCount==MAXNGLOBALJUMP) { globalMoveFlag = 1; }
	else
	{ wastedIterations = 0; globalJumpCount++; state->CopyState(globalBestState); localBestScore = NEG_INF;  
	  num_cases += globalJumpAllG (1, 1, LOOKFORWARD, GLOBALJUMP_SFACTOR, state, featureList, bestList);	// larger graph jump	  
	  state->plp = state->partitionlogPrior (state->L,state->xi,state->alpha);
	  for(l=0; l<state->L; l++) { state->pll[l] = state->cluster_k_loglikelihood (l,state->xi,state->graphlist[l]); }	 
	}	 
      }
      // GLOBAL JUMP MOVES /////////////////////////////////////////////////////////////////////////////////////////////////////////
      
      // SEARCH RESTART ////////////////////////////////////////////////////////////////////////////////////////////////////////////
      if(globalMoveFlag)
      { globalMoveFlag = 0; modesList->UpdateList(globalBestState); nmodes++; gsl_rng_set(rnd,seedset[nmodes-1]); start = clock(); k = 0; 
	localBestScore = NEG_INF; globalBestScore = NEG_INF; featureList->FlushList(state);
	
  #ifndef CUDA
	if(nmodes>maxNmodes)   { break; }
  #else
	if(nmodes>maxNmodes-1) { break; }
  #endif
  
	if (nmodes <= TRY_EACH_INIT*N_INIT)	// analyse prescribed starting points
	{	delete state; delete localBestState; delete globalBestState;
		strcpy(datafile,"DATA/"); strcat(datafile,argv[1]); strcat(datafile,"_init"); strncat(datafile,&initID[(nmodes-1)%N_INIT],1); strcat(datafile,".txt");
		ifstream initfile(datafile); state = new DPmixGGM(X,L,n,p,0.1,initfile); initfile.close();
		localBestState = new DPmixGGM(state); globalBestState = new DPmixGGM(state);
		num_cases += updateAllXis (1, state, bestList); L = state->L; score = state->plp; for(l=0; l<L; l++) { score += state->pll[l]; }
		
	}
	else if (nmodes <= (TRY_EACH_INIT*N_INIT+N_RANDOM_RESTART) && (N_RANDOM_RESTART>0))	// analyse renadom starting points
	{ 	randomRestart (rand_myInt(MAXL-1)+2, state, 0.1); initstates[N_INIT-1+nmodes-TRY_EACH_INIT*N_INIT] = new DPmixGGM(state);
		initstateID[nmodes] = N_INIT-1+nmodes-TRY_EACH_INIT*N_INIT;
		num_cases += updateAllXis (1, state, bestList); L = state->L; score = state->plp; for(l=0; l<L; l++) { score += state->pll[l]; }
	}
	else if (nmodes <= (TRY_EACH_INIT*N_INIT+N_RANDOM_RESTART+N_MODES_LIST_RESTART))
	{ 	int maxI; Real maxScore = NEG_INF;
		for(i=0; i<(nmodes-1); i++) { if(modesList->score_list[i]>maxScore) { maxScore = modesList->score_list[i]; maxI = i; } };
		
		//state->CopyState(initstates[1]);
		state->CopyState(initstates[initstateID[maxI]]);		
		localBestState->CopyState(state); globalBestState->CopyState(state);
		num_cases += updateAllXis (1, state, bestList); L = state->L; score = state->plp; for(l=0; l<L; l++) { score += state->pll[l]; }
	}
#ifndef CUDA
	else
	{ 	bestList = new DPmixGGMlist (sizeOfBestList, n, p); int maxI; Real maxScore = NEG_INF;
		for(i=0; i<(nmodes-1); i++) { if(modesList->score_list[i]>maxScore) { maxScore = modesList->score_list[i]; maxI = i; } }
		gsl_rng_set(rnd,seedset[maxI]);
	
		//state->CopyState(initstates[1]);
		state->CopyState(initstates[initstateID[maxI]]);
		localBestState->CopyState(state); globalBestState->CopyState(state);
		num_cases += updateAllXis (1, state, bestList); L = state->L; score = state->plp; for(l=0; l<L; l++) { score += state->pll[l]; }
	}
#endif
      }      
      // SEARCH RESTART ////////////////////////////////////////////////////////////////////////////////////////////////////////////
      
      // SCORE RECORDING ///////////////////////////////////////////////////////////////////////////////////////////////////////////
      L = state->L; score = state->plp; for(l=0; l<L; l++) { score += state->pll[l]; }
      now = clock(); cpu_time = ((double) (now-start))/CLOCKS_PER_SEC; num_allModels += num_cases;
      printf("%ld %d %.4f %.4f %.4f %d %.4f %d %ld\n",k,state->L,score,localBestScore,globalBestScore,nmodes,cpu_time,num_cases,num_allModels);
      
      if(score > localBestScore) { localBestScore = score; wastedIterations = 0; localJumpCount = 0; localBestState->CopyState(state); } else { wastedIterations++; }
      if(score > globalBestScore) { globalBestScore = score; wastedIterations = 0; globalJumpCount = 0; globalBestState->CopyState(state); featureList->UpdateList(state); } else { wastedIterations++; }
      // SCORE RECORDING ///////////////////////////////////////////////////////////////////////////////////////////////////////////
  }
  
  // writing the lists  
  modesList->WriteList (outmodes);
  #ifndef CUDA
  bestList->WriteList (outbest);
  #endif
      
  // cleanups
  #ifdef CUDA
  for(r=0; r<n_devices; r++)
  {	cudaSetDevice(r); cudaFree(device[r].d_in_add); cudaFree(device[r].d_in_delete); cudaFree(device[r].d_which_add); cudaFree(device[r].d_which_delete);
	delete[] device[r].h_in_delete; delete[] device[r].h_in_add; delete[] device[r].which_delete; delete[] device[r].which_add;
  }
  delete[] device;
  #endif
  
  outmodes.close(); 
  #ifndef CUDA
  outbest.close();
  #endif
#endif

////////////////////////////////////////////////////////////// END OF SSS ///////////////////////////////////////////////////////////////
  
////////////////////////////////////////////////////////////// START OF MCMC ///////////////////////////////////////////////////////////////
#ifdef MCMC
  
  // OUTPUT FILES
  char outfile[100] = ""; strcpy(outfile,"RES/");
  strcat(outfile,argv[1]); strcat(outfile,"_MAP"); strcat(outfile,argv[2]); strcat(outfile,".txt"); ofstream outMAP(outfile);
  strcpy(outfile,"RES/"); strcat(outfile,argv[1]); strcat(outfile,"_MCMCall"); strcat(outfile,argv[2]); strcat(outfile,".txt"); ofstream outMCMCall(outfile);
  outMAP << n << " " << p << endl; outMCMCall << n << " " << p << endl;  
  
  // more declarations and initialisations
  int *cluster_mat = new int[n*n]; for(i=0; i<n*n; i++) { cluster_mat[i] = 0; }; int* edge_mat = new int[n*ee]; for(i=0; i<n*ee; i++) { edge_mat[i] = 0; }
  List MAPList = new DPmixGGMlist (1, n, p); Real lastBestScore = NEG_INF; State state;
  
  // start the stopwatch
  start = clock();
  
  // Initializations  
  strcpy(datafile,"DATA/"); strcat(datafile,argv[1]); strcat(datafile,"_init"); strncat(datafile,argv[2],1); strcat(datafile,".txt");
  ifstream initfile(datafile); state = new DPmixGGM(X,L,n,p,0.1,initfile); initfile.close(); 
    
  for(k=0; k<(BURNIN+N_ITR); k++)
  {	MCMCUpdateXi(state); MCMCUpdateG (state);
	
	score = state->partitionlogPrior(state->L, state->xi, state->alpha); for(l=0; l<state->L; l++) { score += state->pll[l]; }
	now = clock(); cpu_time = ((double) (now-start))/CLOCKS_PER_SEC; printf("%ld %d %.4f %.4f\n",k,state->L,score,cpu_time);
  
	if(k>=BURNIN)
	{	for(i=0; i<n; i++) { q = state->xi[i]; for(j=0; j<n; j++) { r = state->xi[j]; cluster_mat[i*n+j] += (q==r); } }
		for(i=0; i<n; i++) { t = 0; for(q=0; q<p-1; q++) { for(r=q+1; r<p; r++) { edge_mat[i*ee+t] += state->graphlist[state->xi[i]]->Edge[q][r]; t++; } } }		
		if(score>lastBestScore) { MAPList->UpdateList(state); }
	}
  }
    
  MAPList->WriteList (outMAP);
  for(i=0; i<n*n; i++ ) { outMCMCall << Real(cluster_mat[i])/Real(N_ITR) << " "; }; outMCMCall << endl;
  for(i=0; i<n; i++) { for(j=0; j<ee; j++) { outMCMCall << Real(edge_mat[i*ee+j])/Real(N_ITR) << " "; }; outMCMCall << endl; }
  
  delete[] cluster_mat; delete[] edge_mat; outMAP.close(); outMCMCall.close();
  
#endif
////////////////////////////////////////////////////////////// END OF MCMC ///////////////////////////////////////////////////////////////
  
  // cleanups
  gsl_rng_free (rnd); delete[] X; gsl_integration_workspace_free (w);
    
}
// log determinant
__device__ void d_log_det(myInt p, Real* A, Real *result)
{
  myInt i,j,k; Real temp; *result = 0; __shared__ Real br; __shared__ bool flag; flag = 0;  
  myInt tid = threadIdx.x; myInt bdim = blockDim.x; myInt ii, jj;
  
  for (i=0; i<p; i++)
  { ii = i*(i+1)/2;
  
    if(tid==0)
    { jj = ii; temp = A[jj+i];
      for(k=i-1; k>=0; k--) { temp = temp - A[jj+k]*A[ii+k]; };
      if (temp <= 0.0) { *result = NEG_INF; flag = 1; } else { br = sqrt (temp); }
      A[jj+i] = temp / br;
    }
    
    if(flag) return;
    
    for (j=(i+1+tid); j<p; j+=bdim)
    { jj = j*(j+1)/2; temp = A[jj+i];
      for(k=i-1; k>=0; k--) { temp = temp - A[jj+k]*A[ii+k]; }
      A[jj+i] = temp / br;
    }
  }
  
  #ifdef ISFLOAT
  if(tid==0) { for(i=0; i<p; i++) { *result += logf(A[i*(i+1)/2+i]); }; *result = 2* (*result); }
  #else
  if(tid==0) { for(i=0; i<p; i++) { *result += log(A[i*(i+1)/2+i]);  }; *result = 2* (*result); }
  #endif
}

// computes the normalizing constant of a G-Wishart distribution for a full p-dimensional graph with parameters delta and D 
__device__ Real d_gwish_nc_complete(Real delta, myInt p, Real *D, myBool flag)
{
  Real c,a,g,d; myInt i; Real dblP = p; c = 0.0; a = 0.0; g = 0.0; d = 0.0;  
  myInt tid = threadIdx.x;
  
  if(flag) { d_log_det(p, D, &d); }
  
  if(tid==0) {  
  a = (delta + dblP - 1) / 2.0; d = a * d;
  c = dblP * a * log_2; g = dblP * (dblP - 1) * log_pi_over_4;  
  
  #ifdef ISFLOAT
	for(i=0; i<p; i++) { g += lgammaf(a - (Real) i / 2.0); }
  #else
	for(i=0; i<p; i++) { g += lgamma(a - (Real) i / 2.0);  }
  #endif	
  }
  
  return (-d + c + g);	// return value for thread 0 is correct and subsequently used
}


// utility function for making submatrices
__device__ void d_make_sub_mat_dbl(myInt p, myInt p_sub, myInt *sub, Real *A, Real *B)
{
  myInt i,j, ii, s_i, s_j; int tid = threadIdx.x; int bdim = blockDim.x;
  
  for(i=tid; i<p_sub; i+=bdim)
  {	ii = i*(i+1)/2; s_i = sub[i];
	for(j=0; j<=i; j++) { s_j = sub[j]; B[ii+j] = ((s_i >= s_j) ? A[s_i*(s_i+1)/2+s_j] : A[s_j*(s_j+1)/2+s_i]); }    
  }
  
}


__device__ Real CliqueScore (myInt p, Real r_nsub, myInt CliqueDimens, myInt* Clique, Real* D_prior, Real* D_post, Real* sub_D)
{	
  Real score; Real delta = 3.0;
  
#ifdef IDENTITY_DPRIOR
  score = - d_gwish_nc_complete(delta, CliqueDimens, sub_D, 0);
#else
  d_make_sub_mat_dbl(p, CliqueDimens, Clique, D_prior, sub_D); score = - d_gwish_nc_complete(delta,          CliqueDimens, sub_D, 1);
#endif
  d_make_sub_mat_dbl(p, CliqueDimens, Clique, D_post,  sub_D); score +=  d_gwish_nc_complete((delta+r_nsub), CliqueDimens, sub_D, 1);
  
  return (score);
}

// shared memory demand: p myInt + p*
__global__ void GGScore (myInt* d_in_delete, myInt* d_in_add, myInt* d_in_score_bi, Real* d_in_score_br, Real* d_out_score)
{
  int bid = blockIdx.x; myInt tid = threadIdx.x;
  __shared__ myInt n, a, b, baseCliqueDimens, CliqueType, CliqueID; __shared__ Real r_nsub, score;
  __shared__ myInt* Cliques; __shared__ myInt* Separators; __shared__ Real* D_post; __shared__ Real* D_prior;
  
  if(tid==0)
  {	n = (int) *d_in_delete; r_nsub = (Real) d_in_score_bi[0]; a = d_in_score_bi[1+bid*5]; b = d_in_score_bi[1+bid*5+1];
	baseCliqueDimens = d_in_score_bi[1+bid*5+2]; CliqueType = d_in_score_bi[1+bid*5+3]; CliqueID = d_in_score_bi[1+bid*5+4]; // CliqueType = 1 for CanDelete, else CanAdd
	Cliques = d_in_delete + (2 + *(d_in_delete+1)); Separators = d_in_add + (1 + *(d_in_add));
	score = d_in_score_br[0]; D_post = d_in_score_br+1; D_prior = D_post + n*(n+1)/2;
  }; SYNC;
  
  //if(tid==0) { d_out_score[bid] = (Real) baseCliqueDimens; }; SYNC; return;
  //if(tid==0) { d_out_score[bid] = (Real) bid; }; SYNC; return;
  
  extern __shared__ Real shmem[]; Real* sub_D = shmem; myInt* Clique = (myInt*) (sub_D + (baseCliqueDimens+2)*(baseCliqueDimens+3)/2);  
  int i, j, k; Real tempscore;
  
  if(tid==0)
  {	if(CliqueType)  { k = 0; for(i=0; i<(baseCliqueDimens+2); i++) { j = Cliques[CliqueID*n+i]; if((j!=a)&&(j!=b)) { Clique[k] = j; k++; } } }
	else		{ k = CliqueID*n; for(i=0; i<baseCliqueDimens; i++) { Clique[i] = Separators[k+i]; } }
  }; SYNC;
  
  // baseClique
  tempscore = CliqueScore (n, r_nsub, baseCliqueDimens, Clique, D_prior, D_post, sub_D); SYNC;
  if(tid==0) { if(CliqueType==1) { score -= tempscore; } else { score += tempscore; }; Clique[baseCliqueDimens] = b; }; SYNC;
  
  // baseClique + b
  tempscore = CliqueScore (n, r_nsub, baseCliqueDimens+1, Clique, D_prior, D_post, sub_D); SYNC;
  if(tid==0) { if(CliqueType==1) { score += tempscore; } else { score -= tempscore; }; Clique[baseCliqueDimens] = a; }; SYNC;
  
  // baseClique + a
  tempscore = CliqueScore (n, r_nsub, baseCliqueDimens+1, Clique, D_prior, D_post, sub_D); SYNC;
  if(tid==0) { if(CliqueType==1) { score += tempscore; } else { score -= tempscore; }; Clique[baseCliqueDimens+1] = b; }; SYNC;
  
  // baseClique + a + b
  tempscore = CliqueScore (n, r_nsub, baseCliqueDimens+2, Clique, D_prior, D_post, sub_D); SYNC;
  if(tid==0) { if(CliqueType==1) { score -= tempscore; } else { score += tempscore; }; d_out_score[bid] = score; }; SYNC;
    
  return;
}
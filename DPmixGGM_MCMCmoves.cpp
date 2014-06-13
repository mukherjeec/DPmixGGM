#define MCMCMOVES_CPP
#ifndef GRAPH_CPP
#include "graph.cpp"
#endif
#ifndef GWISH_CPP
#include "gwish.cpp"
#endif
#ifndef DPMIXGGM_CPP
#include "DPmixGGM.cpp"
#endif

//------ Update the cluster parameter (xi) --------------
void MCMCUpdateXi(State a)
{
  // Making a local copy of DPmixGGM class
  myInt n = a->n; myInt p = a->p; myInt *xi = a->xi; myInt L = a->L; LPGraph* graphlist = a->graphlist; Real plp = a->plp; Real *pll = a->pll; Real alpha = a->alpha; 
  
  int i, j, k, l; LPGraph *graphlist_new; LPGraph *tempgraphlist; LPGraph newgraph;
  int xi_old, xi_new; int others; Real *qs; bool temp; double maxq, sumq;
  
  //cout << "L before loop start = " << L << endl;   

  //----------- Main Loop: Loop through every observation ----------
  for(i=0; i<n; i++)
  {   //cout << i << " "; fflush(stdout);
      xi_old = xi[i]; xi[i] = -1;
      
      //----------------- Check to see if we need to drop a group -------------
      others = 0; for(j=0; j<n; j++) { if(xi[j] == xi_old) { others = 1; break; } }
      
      if(!others)
      {	  //cout << endl << "first check.\n"; fflush(stdout); 
	  graphlist_new = new LPGraph[L-1];
	  for(l = 0; l < L; l++) 
	  {   if(l < xi_old) { graphlist_new[l] = graphlist[l]; }
	      if(l > xi_old) { graphlist_new[l - 1] = graphlist[l]; }
	  }
	  tempgraphlist = graphlist; graphlist = graphlist_new; delete tempgraphlist[xi_old]; delete[] tempgraphlist; a->graphlist = graphlist;
	  for(j=0; j<n; j++) { if(xi[j] > xi_old) xi[j] = xi[j] - 1; }; L--; a->L = L;
      }
      
      //----------- Predictive Distributions, Existing Clusters ------------------
      qs = new Real[L+1]; for(l=0; l<L; l++) { qs[l] = a->predictiveDistribution (i,l,xi,graphlist[l]); }

      //----------- Predictive Distributions, New Cluster ------------------------
      newgraph = new Graph; newgraph->InitGraph(p);
      for(k=0; k<p-1; k++) { for(l=(k+1); l<p; l++) { temp = (gsl_ran_flat(rnd, 0.0, 1.0) < 0.5); newgraph->Edge[k][l] = temp; newgraph->Edge[l][k] = temp; } }
      TurnFillInGraph(newgraph); if(!newgraph->IsDecomposable()) { printf("HELP\n"); }
      qs[L] = a->predictiveDistribution (i, L, xi, newgraph);
      
      //cout << "newgraph->nVertices = " << newgraph->nVertices << " qs[L] = " << qs[L] << endl; fflush(stdout);

      //------------ Now that we've scored everything, make a proposal ----------
      maxq = qs[0]; for(l = 0; l < L + 1; l++) if(qs[l] > maxq) maxq = qs[l];
      sumq = 0; for(l = 0; l < L+1; l++) sumq += exp(qs[l] - maxq); for(l = 0; l < L+1; l++) qs[l] = exp(qs[l] - maxq) / sumq;
      xi_new = rand_int_weighted(L+1, qs); //cout << "xi_new = " << xi_new << endl; fflush(stdout);

      //------ Put observation in new cluster and make sure things are consistent----------
      xi[i] = xi_new;
      if(xi[i] == L)//If we created a new cluster, we have to do a little work
      {	  graphlist_new = new LPGraph[L+1]; for(l=0; l<L; l++) { graphlist_new[l] = graphlist[l]; }
	  graphlist_new[L] = newgraph; tempgraphlist = graphlist; graphlist = graphlist_new; delete[] tempgraphlist; a->graphlist = graphlist;
	  L++; a->L = L; delete[] pll; pll = new Real[L]; //cout << "Updated.\n"; fflush(stdout);
      }
      else { delete newgraph; }
      delete[] qs;
  }

}


//Function that determines which neighbors of a graph are decomposable the vector which has length the number of possible edges
int FindDecomposableNeighbors (LPGraph graph, int *which)
{
  int p = graph->nVertices; int ee = p*(p-1)/2; int i,j; int k = 0; int num = 0; //cout << "p = " << p << endl; fflush(stdout);
  for(i=0; i<p-1; i++)
  {	for(j=(i+1); j<p; j++)
	{	//cout << "(" << i << "," << j << ") "; fflush(stdout); //exit(0);
		if(graph->Edge[i][j]) {	if(graph->CanDeleteEdge(i,j) != -1) { which[k] = 1; } else { which[k] = 0; } } else { which[k] = graph->CanAddEdge(i,j); }
		if(which[k]) { num++; }; k++; //cout << num << " "; fflush(stdout);
	}
  } 
  return(num);
  
}

//This code updates the graph G for each cluster
void MCMCUpdateG (State a)
{
  // Making a local copy of DPmixGGM class
  myInt n = a->n; myInt p = a->p; myInt *xi = a->xi; myInt L = a->L; LPGraph* graphlist = a->graphlist; Real plp = a->plp; Real *pll = a->pll; Real alpha = a->alpha;
  
  //cout << "L = " << L << endl; fflush(stdout);
  //for(myInt l=0; l<L; l++) { cout << "l = " << l << " graphlist[l]->nVertices = " << graphlist[l]->nVertices << endl; fflush(stdout); }
    
  //---------- Set-up -----------------
  int i,j,k,l,ii; int n_dec, n_dec_new; int which_change; int ee = p*(p-1) / 2;
  int *which_dec = new int[ee]; LPGraph newgraph, tempgraph; double numerator, denominator; int n_sub; 
  Real n0 = N0; Real *mu0 = new Real[p]; for(i=0; i<p; i++) mu0[i] = 0; Real *xbar = new Real[p]; Real *mu_bar = new Real[p];
  Real *D_prior = new Real[p*(p+1)/2]; Real *D_post = new Real[p*(p+1)/2];
  
  //--------------------------------------

  //----- Loop through each cluster and update G ----------------
  for(l=0; l<L; l++)
  {   //--------------- Collect Cluster Information -------------
      n_sub = 0; for(i=0; i<n; i++) { if(xi[i]==l) n_sub++; }; make_sub_means_and_cov(a->X, xi, l, p, n, n_sub, xbar, D_post);
      for(i=0; i<p; i++) mu_bar[i] = (n_sub * xbar[i] + n0 * mu0[i]) / (n_sub + n0);
      for(i=0; i<p*(p+1)/2; i++) D_prior[i] = 0; for(i=0; i<p; i++) D_prior[i*(i+1)/2+i] = 1; for(i=0; i<p; i++) D_post[i*(i+1)/2+i] += 1;      
      for(i=0; i<p; i++) { ii = i*(i+1)/2; for(j=0; j<=i; j++) { D_post[ii+j] += -(n_sub+n0)*mu_bar[i]*mu_bar[j] + n_sub*xbar[i]*xbar[j] + n0*mu0[i]*mu0[j]; } }
      
      //cout << "l = " << l << " graphlist[l]->nVertices = " << graphlist[l]->nVertices << endl; fflush(stdout);
      //cout << "l = " << l << " graphlist[l]->IsDecomposable() = " << graphlist[l]->IsDecomposable() << endl; fflush(stdout);

      //---------------  Propose a Neighbor ---------------------
      n_dec = FindDecomposableNeighbors(graphlist[l], which_dec); //cout << "n_dec = " << n_dec << endl;fflush(stdout);
      which_change = sample_from(which_dec,ee,n_dec); newgraph = new Graph(graphlist[l]); //cout << "which_change = " << which_change << endl; fflush(stdout);
      newgraph->FlipEdge(which_change); if(!newgraph->IsDecomposable()) { printf("HELP\n"); };
      //cout << "l = " << l << " newgraph->IsDecomposable() = " << newgraph->IsDecomposable() << endl; fflush(stdout);
      n_dec_new = FindDecomposableNeighbors (newgraph, which_dec); //cout << "n_dec_new = " << n_dec_new << endl;fflush(stdout);

      //-------------- Compute a ratio ----------------------------
      numerator = -log(n_dec_new) +  j_g_decomposable(newgraph, D_prior, D_post, DELTA0, n_sub, 0); //HERE;
      denominator = -log(n_dec)   +  j_g_decomposable(graphlist[l], D_prior, D_post, DELTA0, n_sub, 0);
      
      //cout << "numerator = " << numerator << " denominator = " << denominator << " (numerator-denominator) = " << (numerator-denominator) << endl; fflush(stdout);

      //------- Do we update or stay the same ---------------------
      if(log(gsl_ran_flat(rnd, 0.0, 1.0)) < (numerator-denominator))
      {	//cout << "*" << endl; fflush(stdout);
	tempgraph = graphlist[l]; graphlist[l] = newgraph; delete tempgraph;
	pll[l] = numerator + log(n_dec_new) - (Real(n_sub*p)/2) * log_2_pi + Real(p)/2 * log(n0 / (n_sub + n0));
      }
      else
      { delete newgraph;
	pll[l] = denominator + log(n_dec) - (Real(n_sub*p)/2) * log_2_pi + Real(p)/2 * log(n0 / (n_sub + n0));
      }
      
      //cout << "end.\n"; fflush(stdout);
  }

  delete[] D_prior; delete[] D_post; delete[] which_dec;

}
/* KMeansRexCore.cpp
A fast, easy-to-read implementation of the K-Means clustering algorithm.
  allowing customized initialization (random samples or plus plus)
  and vectorized execution via the Eigen matrix template library.

Intended to be compiled as a shared library libkmeansrex.so
 which can then be utilized from high-level interactive environments,
  such as Matlab or Python.

Contains:
  Utility Fcns: 
    discrete_rand : sampling discrete r.v.
    select_without_replacement : sample discrete w/out replacement


  Cluster Location Mu Initialization:
    sampleRowsRandom : sample at random (w/out replacement)
    sampleRowsPlusPlus : sample via K-Means++ (Arthur et al)
              see http://en.wikipedia.org/wiki/K-means%2B%2B

  K-Means Algorithm (aka Lloyd's Algorithm)
    run_lloyd : executes lloyd for spec'd number of iterations

  External "C" function interfaces (for calling from Python)
    NOTE: These take only pointers to float arrays, not Eigen array types

    RunKMeans          : compute cluster centers and assignments via lloyd
    SampleRowsPlusPlus : get just a plusplus initialization

Dependencies:
  mersenneTwister2002.c : random number generator

Author: Mike Hughes (www.michaelchughes.com)
Date:   2 April 2013
*/

extern "C" {
  void RunKMeans(double *X_IN,  int N,  int D, int K, int Niter, int seed, char* initname, double *Mu_OUT, double *Z_OUT);
  void SampleRowsPlusPlus(double *X_IN,  int N,  int D, int K, int seed, double *Mu_OUT);
}

#include <iostream>
#include "mersenneTwister2002.c"
#include "Eigen/Dense"

using namespace Eigen;
using namespace std;

/*  DEFINE Custom Type Names to make code more readable
    ExtMat :  2-dim matrix/array externally defined (e.g. in Matlab or Python)
*/
typedef Map<ArrayXXd> ExtMat;
typedef ArrayXXd Mat;
typedef ArrayXd Vec;
typedef Array<bool,Dynamic,1> BoolVec;
typedef ArrayXd::Index Idx;


// ====================================================== Utility Functions
void set_seed( int seed ) {
  init_genrand( seed );
}

int discrete_rand( Vec &p ) {
    double total = p.sum();
    int K = (int) p.size();
    
    double r = total*genrand_double();
    double cursum = p(0);
    int newk = 0;
    while ( r >= cursum && newk < K-1) {
        newk++;
        cursum += p[newk];
    }
    if ( newk < 0 || newk >= K ) {
        cerr << "Badness. Chose illegal discrete value." << endl;
        return -1;
    }
    return newk;
}

void select_without_replacement( int N, int K, Vec &chosenIDs) {
    Vec p = Vec::Ones(N);
    for (int kk =0; kk<K; kk++) {
      int choice;
      int doKeep = false;
      while ( doKeep==false) {
      
        doKeep=true;
        choice = discrete_rand( p );
      
        for (int previd=0; previd<kk; previd++) {
          if (chosenIDs[previd] == choice ) {
            doKeep = false;
            break;
          }
        }      
      }      
      chosenIDs[kk] = choice;     
    }
}

// http://stackoverflow.com/questions/13290395/how-to-remove-a-certain-row-or-column-while-using-eigen-library-c
void removeRow(Mat &matrix, unsigned int rowToRemove)
{
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();

    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

    matrix.conservativeResize(numRows,numCols);
}

// http://stackoverflow.com/questions/13290395/how-to-remove-a-certain-row-or-column-while-using-eigen-library-c
void removeColumn(Mat &matrix, unsigned int colToRemove)
{
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols()-1;

    if( colToRemove < numCols )
        matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);

    matrix.conservativeResize(numRows,numCols);
}

// ======================================================= Init Cluster Locs Mu

void sampleRowsRandom( ExtMat &X, ExtMat &Mu ) {
    int N = X.rows();
    int K = Mu.rows();
    Vec ChosenIDs = Vec::Zero(K);
    select_without_replacement( N, K, ChosenIDs );
		for (int kk=0; kk<K; kk++) {
		  Mu.row( kk ) = X.row( ChosenIDs[kk] );
		}
}

void sampleRowsPlusPlus( ExtMat &X, ExtMat &Mu ) {
    int N = X.rows();
    int K = Mu.rows();
    Vec ChosenIDs = Vec::Ones(K);
    int choice = discrete_rand( ChosenIDs );
    Mu.row(0) = X.row( choice );
    ChosenIDs[0] = choice;
    Vec minDist(N);
    Vec curDist(N);
    for (int kk=1; kk<K; kk++) {
      curDist = ( X.rowwise() - Mu.row(kk-1) ).square().rowwise().sum().sqrt();
      if (kk==1) {
        minDist = curDist;
      } else {
        minDist = curDist.min( minDist );
      }      
      choice = discrete_rand( minDist );
      ChosenIDs[kk] = choice;
      Mu.row(kk) = X.row( choice );
    }       
}

void sampleRowsMAPA( ExtMat &U, ExtMat &seeds ) {
	int N = U.rows();
	int D = seeds.rows();
	float tol = 1e-8;
	int k = 0;
	Mat U1 = U;
	Idx ind_m;
	float max = (U.rowwise() - U.colwise().mean()).square().rowwise().sum().maxCoeff(&ind_m);
	seeds.row(0) = U1.row(ind_m);
	removeRow(U1,ind_m);

// 	BoolVec is_far_enough_away = ((U.rowwise() - seeds.row(0)).square().rowwise().sum() > tol);
// 	for(int ii=is_far_enough_away.size()-1; ii >= 0; ii--)
// 	{
// 		if (!is_far_enough_away(ii))
// 		{
// 			removeRow(U1,ii);
// 		}
// 	}
	while(k < D && U1.rows() > 0)
	{
		Mat seeds_row_arranged = seeds.topRows(k+1).transpose();
		seeds_row_arranged.resize(1,(k+1)*seeds.cols());
		Mat copies_of_U_row_arranged = U1.replicate(1,k+1);
		max = (copies_of_U_row_arranged.rowwise() - seeds_row_arranged.row(0)).square().rowwise().sum().maxCoeff(&ind_m);

		k += 1;
		seeds.row(k) = U1.row(ind_m);
		removeRow(U1,ind_m);

// 		is_far_enough_away = ((U1.rowwise() - seeds.row(k)).square().rowwise().sum() > tol);
// 		for(int ii=is_far_enough_away.size()-1; ii >= 0; ii--)
// 		{
// 			if (!is_far_enough_away(ii))
// 			{
// 				removeRow(U1,ii);
// 			}
// 		}
	}
}

void init_Mu( ExtMat &X, ExtMat &Mu, char* initname ) {		  
	  if ( string( initname ) == "random" ) {
    		sampleRowsRandom( X, Mu );
	  } else if ( string( initname ) == "plusplus" ) {
  			sampleRowsPlusPlus( X, Mu );
	  } else if ( string( initname ) == "mapa" ) {
  			sampleRowsMAPA( X, Mu );
	  }
}

// ======================================================= Update Cluster Assignments Z
void pairwise_distance( ExtMat &X, ExtMat &Mu, Mat &Dist ) {
  int N = X.rows();
  int D = X.cols();
  int K = Mu.rows();

  // For small dims D, for loop is noticeably faster than fully vectorized.
  // Odd but true.  So we do fastest thing 
  if ( D <= 16 ) {
    for (int kk=0; kk<K; kk++) {
      Dist.col(kk) = ( X.rowwise() - Mu.row(kk) ).square().rowwise().sum();
    }    
  } else {
    Dist = -2*( X.matrix() * Mu.transpose().matrix() );
    Dist.rowwise() += Mu.square().rowwise().sum().transpose().row(0);
  }
}

double assignClosest( ExtMat &X, ExtMat &Mu, ExtMat &Z, Mat &Dist) {
  double totalDist = 0;
  int minRowID;

  pairwise_distance( X, Mu, Dist );

  for (int nn=0; nn<X.rows(); nn++) {
    totalDist += Dist.row(nn).minCoeff( &minRowID );
    Z(nn,0) = minRowID;
  }
  return totalDist;
}

// ======================================================= Update Cluster Locations Mu
void calc_Mu( ExtMat &X, ExtMat &Mu, ExtMat &Z) {
  Mu = Mat::Zero( Mu.rows(), Mu.cols() );
  Vec NperCluster = Vec::Zero( Mu.rows() );
  
  for (int nn=0; nn<X.rows(); nn++) {
    Mu.row( (int) Z(nn,0) ) += X.row( nn );
    NperCluster[ (int) Z(nn,0)] += 1;
  }  
  Mu.colwise() /= NperCluster;
}

// ======================================================= Overall Lloyd Algorithm
void run_lloyd( ExtMat &X, ExtMat &Mu, ExtMat &Z, int Niter )  {
  double prevDist,totalDist = 0;

  Mat Dist = Mat::Zero( X.rows(), Mu.rows() );  

  for (int iter=0; iter<Niter; iter++) {
    
    totalDist = assignClosest( X, Mu, Z, Dist );
    calc_Mu( X, Mu, Z );
    if ( prevDist == totalDist ) {
      break;
    }
    prevDist = totalDist;
  }
}

// =================================================================================
// =================================================================================
// ===========================  EXTERNALLY CALLABLE FUNCTIONS ======================
// =================================================================================
// =================================================================================


void RunKMeans(double *X_IN,  int N,  int D, int K, int Niter, \
               int seed, char* initname, double *Mu_OUT, double *Z_OUT) {
  set_seed( seed );

  ExtMat X  ( X_IN, N, D);
  ExtMat Mu ( Mu_OUT, K, D);
  ExtMat Z  ( Z_OUT, N, 1);

  init_Mu( X, Mu, initname);
  run_lloyd( X, Mu, Z, Niter );
}


void SampleRowsPlusPlus(double *X_IN,  int N,  int D, int K, int seed, double *Mu_OUT) {
  set_seed( seed );

  ExtMat X  ( X_IN, N, D);
  ExtMat Mu ( Mu_OUT, K, D);

  sampleRowsPlusPlus( X, Mu);
}

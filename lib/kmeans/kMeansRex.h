#ifndef KMEANSREX_H
#define KMEANSREX_H

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

/* Modified to work as a C++ header file
   Eric E Monson, Duke University, 4/21/2014
*/


#include <iostream>
// mersenneTwister2002.c reworked as C++ header file class
#include "mersenneTwister2002.h"
#include "Eigen/Dense"

#include <stdlib.h>		/* NULL */
#include <time.h>			/* time */
#include <stdio.h>		/* srand */

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


namespace KMeans {

class KMeansRex {

	public: 
	
		KMeansRex()
		{
			set_seed((unsigned int)time(NULL));
		};
		
	private:
	
  	KMeans::MersenneTwister twist;
  	
		// ====================================================== Utility Functions
		void set_seed( int seed ) {
			twist.InitGenrand( seed );
		}

		int discrete_rand( Vec &p ) {
				double total = p.sum();
				int K = (int) p.size();
		
				double r = total * twist.GenrandDouble();
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

		void removeCloseRows(Mat &U1, ExtMat &seeds, int k, double tol)
		{
			BoolVec is_far_enough_away = ((U1.rowwise() - seeds.row(k)).square().rowwise().sum() > tol);
			// looping through in reverse so don't disrupt indices as removing rows
			for(int ii = (is_far_enough_away.size()-1); ii >= 0; ii--) 
			{
				if (!is_far_enough_away(ii)) 
				{
					removeRow(U1,ii);
				}
			}

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
			int dim = U.cols();
			double tol = 1e-8;
			Mat U1 = U;
			Idx ind_m;
			Vec sq_dist_sum;
			seeds.setZero();
	
			float max = (U.rowwise() - U.colwise().mean()).square().rowwise().sum().maxCoeff(&ind_m);
			seeds.row(0) = U1.row(ind_m);
			removeCloseRows(U1, seeds, 0, tol);
	
			for(int k = 1; k < D; k++)
			{
				if ( U1.rows() == 0 ) { break; }
		
				sq_dist_sum = ArrayXd::Zero(U1.rows());
				for(int ii=0; ii < k; ii++)
				{
					sq_dist_sum += (U1.rowwise() - seeds.row(ii)).square().rowwise().sum();
				}
				max = sq_dist_sum.maxCoeff(&ind_m);
		
				seeds.row(k) = U1.row(ind_m);
				removeCloseRows(U1, seeds, k, tol);
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



}; // class def

} // namespace KMeans

#endif
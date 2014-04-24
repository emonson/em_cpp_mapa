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


namespace KMeans {

class KMeansRex {

	public: 
	
		KMeansRex(const ArrayXXd &Xin, unsigned int Kin)
		{
			set_seed((unsigned int)time(NULL));
			X = Xin;
			N = Xin.rows();			// number of points
			D = Xin.cols();			// dimensionality of points
			K = Kin;
			Seeds.resize(K,D);
			Centers.resize(K,D);
			Z.resize(N,1);
	
			// only using these preset values for now
			method = "mapa";
			init_Mu();
			Centers = Seeds;
			run_lloyd();
		};

		ArrayXXd GetSeeds()
		{
			return Seeds;
		};

		ArrayXXd GetClusterAssignments()
		{
			return Z;
		};

		ArrayXXd GetCenters()
		{
			return Centers;
		};
		
	private:
		
		ArrayXXd X;
		ArrayXXd Seeds;
		ArrayXXd Centers;
		ArrayXXd Z;
		ArrayXXd Dist;
		unsigned int K;
		unsigned int N;
		unsigned int D;
		int Niter;
		string method;
		
		// Random number generator object
  	KMeans::MersenneTwister twist;
  	
		// Actual calculation of clusters
		void run_lloyd()  {
			
			double prevDist,totalDist = 0;
			Centers = Seeds;

			Dist = ArrayXXd::Zero(N,K);  

			for (int iter=0; iter<Niter; iter++) {
		
				totalDist = assignClosest();
				calc_Mu();
				if ( prevDist == totalDist ) {
					break;
				}
				prevDist = totalDist;
			}
		};

		// Switch for which seed generator to use
		void init_Mu() {		  
				if ( method == "random" ) {
						sampleRowsRandom();
				} else if ( method == "plusplus" ) {
						sampleRowsPlusPlus();
				} else if ( method == "mapa" ) {
						sampleRowsMAPA();
				}
		};

		// ====================================================== Utility Functions
		void set_seed( int seed ) {
			twist.InitGenrand( seed );
		};

		int discrete_rand( ArrayXd &p ) {
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
		};

		void select_without_replacement( int N, int K, ArrayXd &chosenIDs) {
				ArrayXd p = ArrayXd::Ones(N);
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
		};

		// http://stackoverflow.com/questions/13290395/how-to-remove-a-certain-row-or-column-while-using-eigen-library-c
		void removeRow(ArrayXXd &matrix, unsigned int rowToRemove)
		{
				unsigned int numRows = matrix.rows()-1;
				unsigned int numCols = matrix.cols();

				if( rowToRemove < numRows )
						matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

				matrix.conservativeResize(numRows,numCols);
		};

		void removeCloseRows(ArrayXXd &U1, ArrayXXd &seeds, int k, double tol)
		{
			Array<bool,Dynamic,1> is_far_enough_away = ((U1.rowwise() - seeds.row(k)).square().rowwise().sum() > tol);
			// looping through in reverse so don't disrupt indices as removing rows
			for(int ii = (is_far_enough_away.size()-1); ii >= 0; ii--) 
			{
				if (!is_far_enough_away(ii)) 
				{
					removeRow(U1,ii);
				}
			}

		};

		// ======================================================= Init Cluster Locs Mu

		void sampleRowsRandom() 
		{
				ArrayXd ChosenIDs = ArrayXd::Zero(K);
				select_without_replacement( N, K, ChosenIDs );
				for (int kk=0; kk<K; kk++) 
				{
					Seeds.row( kk ) = X.row( ChosenIDs[kk] );
				}
		};

		void sampleRowsPlusPlus() 
		{
				ArrayXd ChosenIDs = ArrayXd::Ones(K);
				int choice = discrete_rand( ChosenIDs );
				Seeds.row(0) = X.row( choice );
				ChosenIDs[0] = choice;
				ArrayXd minDist(N);
				ArrayXd curDist(N);
				for (int kk=1; kk<K; kk++) 
				{
					curDist = ( X.rowwise() - Seeds.row(kk-1) ).square().rowwise().sum().sqrt();
					if (kk==1) 
					{
						minDist = curDist;
					} else {
						minDist = curDist.min( minDist );
					}      
					choice = discrete_rand( minDist );
					ChosenIDs[kk] = choice;
					Seeds.row(kk) = X.row( choice );
				}       
		};

		void sampleRowsMAPA() 
		{
			double tol = 1e-8;
			ArrayXXd U1 = X;
			ArrayXd::Index ind_m;
			ArrayXd sq_dist_sum;
			Seeds = ArrayXXd::Zero(K,D);
	
			float max = (X.rowwise() - X.colwise().mean()).square().rowwise().sum().maxCoeff(&ind_m);
			Seeds.row(0) = U1.row(ind_m);
			removeCloseRows(U1, Seeds, 0, tol);
	
			for(int k = 1; k < D; k++)
			{
				if ( U1.rows() == 0 ) { break; }
		
				sq_dist_sum = ArrayXd::Zero(U1.rows());
				for(int ii=0; ii < k; ii++)
				{
					sq_dist_sum += (U1.rowwise() - Seeds.row(ii)).square().rowwise().sum();
				}
				max = sq_dist_sum.maxCoeff(&ind_m);
		
				Seeds.row(k) = U1.row(ind_m);
				removeCloseRows(U1, Seeds, k, tol);
			}
		};

		// ======================================================= Update Cluster Assignments Z
		void pairwise_distance() 
		{
			// For small dims D, for loop is noticeably faster than fully vectorized.
			// Odd but true.  So we do fastest thing 
			if ( D <= 16 ) {
				for (int kk=0; kk<K; kk++) {
					Dist.col(kk) = ( X.rowwise() - Centers.row(kk) ).square().rowwise().sum();
				}    
			} else {
				Dist = -2*( X.matrix() * Centers.transpose().matrix() );
				Dist.rowwise() += Centers.square().rowwise().sum().transpose().row(0);
			}
		};

		double assignClosest() {
			double totalDist = 0;
			int minRowID;

			pairwise_distance();

			for (int nn=0; nn<N; nn++) {
				totalDist += Dist.row(nn).minCoeff( &minRowID );
				Z(nn,0) = minRowID;
			}
			return totalDist;
		};

		// ======================================================= Update Cluster Locations Mu
		void calc_Mu() {
			Centers = ArrayXXd::Zero(K,D);
			ArrayXd NperCluster = ArrayXd::Zero(K);
	
			for (int nn=0; nn < N; nn++) {
				Centers.row( (int) Z(nn,0) ) += X.row( nn );
				NperCluster[ (int) Z(nn,0)] += 1;
			}  
			Centers.colwise() /= NperCluster;
		};



}; // class def

} // namespace KMeans

#endif
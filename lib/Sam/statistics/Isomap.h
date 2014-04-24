#ifndef ISOMAP_H
#define ISOMAP_H

#include "GraphAlgorithms.h"
#include "MetricMDS.h"
#include "Neighborhood.h"

#include <math.h>

template <typename TPrecision>
class Isomap{
  public:
    
    
    Isomap(Neighborhood<TPrecision> *n, int nd) : nb(n), ndims(nd){
    };




    DenseMatrix<TPrecision> embed(Matrix<TPrecision> &data ){
      SparseMatrix<TPrecision> adj = nb->generateNeighborhood(data);
      DenseMatrix<TPrecision> result =  embedAdj(adj);
      adj.deallocate();
      return result;
    };




    DenseMatrix<TPrecision> embedAdj(SparseMatrix<TPrecision> &adj ){
      DenseMatrix<TPrecision> dists(adj.N(), adj.N());
      GraphAlgorithms<TPrecision>::all_pairs_dijkstra(adj, dists);

      for(unsigned int i=0; i<dists.N(); i++){
        dists(i, i) = 0;
        for(unsigned int j=i+1; j<dists.N(); j++){
            TPrecision m = std::min(dists(i, j), dists(j, i));
            dists(i, j) = m;
            dists(j, i) = m;
        }
      }

      DenseMatrix<TPrecision> em = mds.embed(dists, ndims);
      dists.deallocate();
      return em;
    };


    
  
  private:
    Neighborhood<TPrecision> *nb;
    MetricMDS<TPrecision> mds; 
    int ndims;

};



#endif

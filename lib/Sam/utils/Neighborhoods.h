#ifndef NEIGHBORHOODS
#define NEIGHBORHOODS


#include "Random.h"
#include "Linalg.h"
#include "DenseMatrix.h"
#include "DenseVector.h"
#include "RandomSVD.h"
#include "SampleStatistics.h"
#include "SquaredEuclideanMetric.h"

#include <map>
#include <set>
#include <list>

template <typename TPrecision>
class RSTNode{

  public:

    RSTNode<TPrecision>(DenseMatrix<TPrecision> &X, DenseVector<int> ind, int n
        = 0, double eps = 0, int d=10) 
      : v( X.M() ), index(ind) {
      
      static Random<TPrecision> rand;
      //for(int i=0; i< v.N(); i++){
      //  v(i) = rand.Normal();
      //}
      DenseMatrix<TPrecision> Xi = createMatrix(X);
      

      //Use range for random direction sampling
      /* 
      DenseMatrix<TPrecision> Q = RandomRange<TPrecision>::FindRange(Xi, d, 0, true);
      Linalg<TPrecision>::Zero(v);
      for(int i=0; i<Q.N(); i++){
        Linalg<TPrecision>::AddScale(v, rand.Normal(), Q, i, v);
      }
      Q.deallocate();
      */
      //Use SVD for random direction sampling
      
      RandomSVD<TPrecision> svd(Xi, d, true);
     
      Linalg<TPrecision>::Zero(v);
      for(int i=0; i<svd.U.N(); i++){
        Linalg<TPrecision>::AddScale(v, rand.Normal()*svd.S(i), svd.U, i, v);
      }
      //Linalg<TPrecision>::ExtractColumn(svd.U, 0, v);
     
      //Linalg<TPrecision>::Normalize(v);

      DenseVector<TPrecision> s = split(Xi);

      bool stop = false;
      if(n > 0){
        stop = Xi.N() < n;
      }
      else{
        std::cout << sqrt(svd.S(0)) << " : " << eps << std::endl;
        stop =  ( sqrt(svd.S(0))  <  eps ); 
      }
      svd.deallocate();
      
        
     if( !stop ){
        
        std::list<int> m1;
        std::list<int> m2;
        
        for(int i=0; i<s.N(); i++){
          if(s(i) > 0){
            m1.push_back( index(i) );
          }
          else{
            m2.push_back( index(i) );
          }
        }

        DenseVector<int> ind1 = Linalg<int>::ToVector(m1);
        DenseVector<int> ind2 = Linalg<int>::ToVector(m2);
        
        Xi.deallocate();
        s.deallocate();

        c1 = new RSTNode<TPrecision>(X, ind1, n, eps, d);
        c2 = new RSTNode<TPrecision>(X, ind2, n, eps, d);
      
      }
      else{
        Xi.deallocate();
        s.deallocate();
        c1 = NULL;
        c2 = NULL;
      }
    };


    TPrecision split(DenseVector<TPrecision> x){
      TPrecision d = Linalg<TPrecision>::Dot(v, x);
      return d - m;
    };


    RSTNode<TPrecision> *findLeaf(DenseVector<TPrecision> x){
      if(c1 == NULL){
        return this;
      }
      else{
        if(split(x) > 0){
          return c1->findLeaf(x);
        }
        else{
          return c2->findLeaf(x);
        }
      }
    };


    RSTNode<TPrecision> *c1;
    RSTNode<TPrecision> *c2;

    DenseVector<int> index;

  private:

    DenseVector<TPrecision> v;
    TPrecision m;


    DenseMatrix<TPrecision> createMatrix(DenseMatrix<TPrecision> &X){
      DenseMatrix<TPrecision> M(X.M(), index.N());
      
      for(int i=0; i < index.N(); i++){
        Linalg<TPrecision>::SetColumn(M, i, X, index(i));
      }

      return M;
    };



    DenseVector<TPrecision> split(DenseMatrix<TPrecision> &X){
      DenseVector<TPrecision> p = Linalg<TPrecision>::Multiply(X, v, true);
      //TPrecision s = sqrt( SampleStatistics<TPrecision>::Variance(p, m) );
      //static Random<TPrecision> rand;
      //s = ( rand.Uniform()*1.0*s - s ) / 2.0;
      //m += s;

      m = Linalg<TPrecision>::Sum(p)/p.N();
      
      for(int i=0; i< p.N(); i++){
        p(i) -= m;//s;
      }

      return p;
    };

};






template <typename TPrecision>
class Neighborhoods{

  private:
    int K;
    int D;
    DenseMatrix<TPrecision> X;
    RSTNode<TPrecision> **root;

    SquaredEuclideanMetric<TPrecision> l22;

  public:

    Neighborhoods<TPrecision>(DenseMatrix<TPrecision> Xin, int k, int d=10) : K(k), D(d){
      //build forest
      X = Xin;
    }


    void buildTree(int n){
      root = new RSTNode<TPrecision>*[K];
      DenseVector<int> rIndex(X.N());
      for(int i=0; i< rIndex.N(); i++){
        rIndex(i) = i;
      }
      for(int i=0; i<K; i++){
        root[i]=new RSTNode<TPrecision>(X, rIndex, n, 0, D);
      }
    };


    void buildTree(double eps){
      root = new RSTNode<TPrecision>*[K];
      DenseVector<int> rIndex(X.N());
      for(int i=0; i< rIndex.N(); i++){
        rIndex(i) = i;
      }
      for(int i=0; i<K; i++){
        root[i]=new RSTNode<TPrecision>(X, rIndex, 0, eps, D);
      }
    };   

  



    //return neighborhoods unordered 
    DenseVector<int> neighborhood(DenseVector<TPrecision> x){
      std::map<int, int> pts;
      for(int  i=0; i<K; i++){
        RSTNode<TPrecision> *leaf = root[i]->findLeaf(x);
        for(int j=0; j<leaf->index.N(); j++){
          int id = leaf->index(j);
          int count = pts[id];
          pts[id] = count - 1;
        }
      }

      int n = pts.size();
      DenseVector<int> Xn(n);
      int index = 0;
      for(std::map<int, int>::iterator it=pts.begin(); it !=pts.end(); ++it,  ++index){
        Xn(index) = it->first;
      } 

      return Xn;
    };




    //return nieghborhoods order by number of overlaps in the leafs of the
    //different trees
    DenseVector<int> neighborhoodOverlap(DenseVector<TPrecision> x){
      std::map<int, int> pts;
      for(int  i=0; i<K; i++){
        RSTNode<TPrecision> *leaf = root[i]->findLeaf(x);
        for(int j=0; j<leaf->index.N(); j++){
          int id = leaf->index(j);
          int count = pts[id];
          pts[id] = count - 1;
        }
      }

      std::set< std::pair<int, int> > sorted;
      for(std::map<int, int>::iterator it=pts.begin(); it !=pts.end(); ++it){
        std::pair<int, int> p(it->second, it->first);
        sorted.insert(p);
      } 

      int n = sorted.size();
      DenseVector<int> Xn(n);
      
      std::set< std::pair<int, int> >::iterator sit = sorted.begin();
      for(int i=0; i<n; i++, ++sit){
        Xn(i) = sit->second;
      }
      return Xn;
    };





    //Return neighborhood sorted 
    DenseVector<int> neighborhoodSorted(DenseVector<TPrecision> x){
      typename std::map<int, TPrecision> pts;
      for(int  i=0; i<K; i++){
        RSTNode<TPrecision> *leaf = root[i]->findLeaf(x);
        for(int j=0; j<leaf->index.N(); j++){
          int id = leaf->index(j);
          if(pts.find(id) == pts.end() ){
            pts[id] = l22.distance(X, id, x);
          }
        }
      }

      std::list< std::pair<TPrecision, int> > sorted;
      for(typename std::map<int, TPrecision>::iterator it=pts.begin(); it !=pts.end(); ++it){
        std::pair<TPrecision, int> p((*it).second, (*it).first);
        sorted.push_back(p);
      } 
      sorted.sort();
     
      int n = sorted.size();
      DenseVector<int> Xn(n);
      
      typename std::list< std::pair<TPrecision, int> >::iterator sit =  sorted.begin();
      for(int i=0; i<n; i++, ++sit){
        Xn(i) = (*sit).second;
      }
      return Xn;
    };


};


#endif

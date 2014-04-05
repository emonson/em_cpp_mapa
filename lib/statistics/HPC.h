#ifndef HPC_H
#define HPC_H


#include "Geometry.h"
#include "Matrix.h"
#include "EuclideanMetric.h"
#include "SquaredEuclideanMetric.h"
#include "Linalg.h"
#include "LinalgIO.h"
#include "GaussianKernel.h"
#include "KernelDensity.h"
#include "Random.h"

#include <stdlib.h>
#include <limits>
#include <math.h>


template <typename TPrecision>
class HPC{

  private:    
    DenseMatrix<TPrecision> Y;
    DenseMatrix<TPrecision> X;

    DenseMatrix<int> KNNX;
    DenseMatrix<TPrecision> KNNXD;


    unsigned int knnSigma;

    EuclideanMetric<TPrecision> l2metric;
    SquaredEuclideanMetric<TPrecision> sl2metric;


    GaussianKernel<TPrecision> kernelX;



    TPrecision sX;
    TPrecision xCutoff;

    bool verbose;


  public:

    void cleanup(){      
      KNNX.deallocate();
      KNNXD.deallocate();
      Y.deallocate();
      X.deallocate();

    };

    //Create KernelMap 
    HPC(DenseMatrix<TPrecision> Ydata, DenseMatrix<TPrecision> Xinit, unsigned int nnSigma) :
      Y(Ydata), X(Xinit), knnSigma(nnSigma){

        verbose = true;//false;
        init();

        //initial kernel bandwidth
        TPrecision sigma = 0;
        sX = 0;
        for(unsigned int i=0; i < X.N(); i++){
          sigma += sqrt( KNNXD(knnSigma-1, i) );
        }
        sigma *= alpha/X.N();
        sX = sigma/alpha;


        if(verbose){
          std::cout << "sigmaX: " << sigma << std::endl;
          std::cout << "scale: " << sX << std::endl;
        }

        kernelX.setKernelParam(sigma); 


        xCutoff = 3*kernelX.getKernelParam();
        xCutoff = xCutoff*xCutoff;
      };



    HPC(DenseMatrix<TPrecision> Ydata, DenseMatrix<TPrecision> Xopt, 
        unsigned int nnSigma, double sigmaX): Y(Ydata),
    X(Xopt), knnSigma(nnSigma){

      verbose = true;//false;
      init();
      kernelX.setKernelParam(sigmaX);
      sX = kernelX.getKernelParam();   
      xCutoff = 3*kernelX.getKernelParam();
      xCutoff = xCutoff*xCutoff;

      if(verbose){
        std::cout << "knnSigma: " << knnSigma<<std::endl;
        std::cout << "sigmaX: " << kernelX.getKernelParam() <<std::endl;
      }
    }; 







    //evalue objective function, squared error
    TPrecision mse(){
      TPrecision dummy = 0;
      return mse(dummy);
    }

    //evalue objective function, squared error
    TPrecision mse(TPrecision &o){
      o=0;
      TPrecision e = 0;

      //Jacobian of g(x)
      DenseMatrix<TPrecision> J(Y.M(), X.M());

      //Temp vars.
      DenseVector<TPrecision> gfy(Y.M());
      DenseVector<TPrecision> diff(Y.M());
      DenseVector<TPrecision> pDot(X.M());

      for(unsigned int i=0; i < Y.N(); i++){
        g(i, gfy, J);
        e += sl2metric.distance(Y, i, gfy);

        Linalg<TPrecision>::Subtract(gfy, Y, i, diff);  

        for(unsigned int n=0; n< J.N(); n++){
          TPrecision norm = 0;
          for(unsigned int j=0; j<J.M(); j++){
            norm += J(j, n) * J(j, n);
          }
          if(norm == 0){
            norm = 0.0001;
          }
          norm = sqrt(norm);

          for(unsigned int j=0; j<J.M(); j++){
            J(j, n) /= norm;
          }
        }

        Linalg<TPrecision>::Normalize(diff);
        Linalg<TPrecision>::Multiply(J, diff, pDot, true);

        for(unsigned int n=0; n< pDot.N(); n++){
          o += acos(sqrt(pDot(n)*pDot(n)));
        }  
      }
      o = o/(X.M()*X.N())/ M_PI * 180;

      pDot.deallocate();
      gfy.deallocate();
      diff.deallocate();
      J.deallocate();


      return e/Y.N();
    };


    TPrecision mse(int index){
      TPrecision e = 0;


      //Temp vars.
      DenseVector<TPrecision> gfy(Y.M());

      for(unsigned int i=0; i < knnSigma; i++){
        unsigned int nn = KNNX(i, index);
        g(nn, gfy);
        e += sl2metric.distance(Y, nn, gfy); 
      }

      gfy.deallocate();
      return e/knnSigma;
    };







    void minimize(int nIter = 100){
      DenseVector<TPrecision> y(Y.M());
      DenseVector<TPrecision> yout(Y.M());
      DenseVector<TPrecision> x(X.M());
      DenseMatrix<TPrecision> Xtmp(X.M(), X.N());
      DenseVector<TPrecision> zmin(X.M());
      DenseVector<TPrecision> zmax(X.M());
      boundary(zmin, zmax);

      TPrecision msep = std::numeric_limits<TPrecision>::max();
      for(int i=0; i<nIter; i++){
        computeYp();          
        TPrecision mse = 0;
        for(unsigned int i=0; i<Y.N(); i++){
          Linalg<TPrecision>::ExtractColumn(Y, i, y);
          mse += project(y, x, yout, sX/10.f, zmin, zmax);
          Linalg<TPrecision>::SetColumn(Xtmp, i, x);
        }
        mse /= Y.N();
        std::cout << "MSE: " << mse << std::endl;
        if(mse > msep){ break; }

        Linalg<TPrecision>::Copy(Xtmp, X);
        msep = mse;

      }

    };





    DenseMatrix<TPrecision> parametrize(DenseMatrix<TPrecision> &Ydata){
      computeYp();
      DenseVector<TPrecision> y(Ydata.M());
      DenseVector<TPrecision> yout(Ydata.M());
      DenseVector<TPrecision> x(X.M());     
      DenseVector<TPrecision> zmin(X.M());
      DenseVector<TPrecision> zmax(X.M());
      boundary(zmin, zmax);
      DenseMatrix<TPrecision> result(X.M(), Ydata.N());
      for(unsigned int i=0; i<Ydata.N(); i++){
        Linalg<TPrecision>::ExtractColumn(Ydata, i, y);
        project(y, x, yout, sX/10.f, zmin, zmax);
        Linalg<TPrecision>::SetColumn(result, i, x); 
      }
      return result;
    };

    //------g linear 
    //g(x_index) - reconstruction mapping
    void g(unsigned int index, Vector<TPrecision> &out){
      DenseMatrix<TPrecision> sol = LeastSquares(index);

      for(unsigned int i=0; i<Y.M(); i++){
        out(i) = sol(0, i);
      }

      sol.deallocate();
    };  



    //g(x_index) - reconstruction mapping plus tangent plane
    void g(unsigned int index, Vector<TPrecision> &out, Matrix<TPrecision> &J){
      DenseMatrix<TPrecision> sol = LeastSquares(index);

      for(unsigned int i=0; i<Y.M(); i++){
        out(i) = sol(0, i);
      }
      for(unsigned int i=0; i< X.M(); i++){
        for(unsigned int j=0; j< Y.M(); j++){
          J(j, i) = sol(1+i, j);
        }
      }

      sol.deallocate();
    };




    //g(x) - reconstruction mapping
    void g( Vector<TPrecision> &x, Vector<TPrecision> &out){
      DenseMatrix<TPrecision> sol = LeastSquares(x);

      for(unsigned int i=0; i<Y.M(); i++){
        out(i) = sol(0, i);
      }

      sol.deallocate();
    };   




    //g(x) - reconstruction mapping + tangent plance
    void g( Vector<TPrecision> &x, Vector<TPrecision> &out, Matrix<TPrecision> &J){
      DenseMatrix<TPrecision> sol = LeastSquares(x);
      for(unsigned int i=0; i<Y.M(); i++){
        out(i) = sol(0, i);
      }     
      for(unsigned int i=0; i< X.M(); i++){
        for(unsigned int j=0; j< Y.M(); j++){
          J(j, i) = sol(1+i, j);
        }
      }

      sol.deallocate();
    };




    //get original Y's
    DenseMatrix<TPrecision> getY(){
      return Y;
    };

    //get X (parameters for f
    DenseMatrix<TPrecision> getX(){
      return X;
    };








    DenseMatrix<TPrecision> reconstruct(DenseMatrix<TPrecision> &Xpoints){
      DenseMatrix<TPrecision> proj(Y.M(), Xpoints.N());
      reconstruct(Xpoints, proj);     
      return proj;
    };





    void reconstruct(DenseMatrix<TPrecision> &Xpoints, DenseMatrix<TPrecision> &proj){

      DenseVector<TPrecision> tmp(X.M()); 
      DenseVector<TPrecision> yp(Y.M()); 
      for(unsigned int i=0; i < Xpoints.N(); i++){
        Linalg<TPrecision>::ExtractColumn(Xpoints, i, tmp);
        g(tmp, yp);
        Linalg<TPrecision>::SetColumn(proj, i, yp);
      }
      yp.deallocate();
      tmp.deallocate();
    };   


    DenseMatrix<TPrecision> reconstruct(){
      DenseMatrix<TPrecision> proj(Y.M(), Y.N());
      DenseVector<TPrecision> yp(Y.M()); 
      for(unsigned int i=0; i < Y.N(); i++){
        g(i, yp);
        Linalg<TPrecision>::SetColumn(proj, i, yp);
      }
      yp.deallocate();
      return proj;
    };






    TPrecision getSigmaX(){
      return kernelX.getKernelParam();
    };

    GaussianKernel<Precision> &getKernelX(){
      return kernelX;
    };





  private:




    //Initialization
    void init(){

      Yp = Linalg<TPrecision>::Copy(Y);
      if(verbose){
        std::cout << "Initalizing" << std::endl;
      }    

      if(knnSigma > Y.N()){
        knnSigma = Y.N();
      }
      KNNX = DenseMatrix<int>(knnSigma, X.N());
      KNNXD = DenseMatrix<TPrecision>(knnSigma,X.N());
      updateKNNX();

      kernelX = GaussianKernel<TPrecision>( X.M());


      if(verbose) {   
        std::cout << "Initalizing done" << std::endl;
      }

    };     




    //Least squares helper function for computing principal curve g (locally
    //linear regression)
    DenseMatrix<TPrecision> LeastSquares(Vector<TPrecision> &x){
      std::list<TPrecision> dists;
      std::list<unsigned int> index;

      TPrecision tmpCutoff = xCutoff;   
      while(dists.size() < (unsigned int) std::min(20, (int)Y.N()) ){   
        dists.clear();
        index.clear();   
        for(unsigned int i =0; i<Y.N(); i++){
          TPrecision tmp = sl2metric.distance(X, i, x);
          if(tmp < tmpCutoff){
            dists.push_back(tmp);
            index.push_back(i);
          }
        }
        tmpCutoff*=2;
      }




      unsigned int N = dists.size();
      DenseMatrix<TPrecision> A(N, X.M()+1);
      DenseMatrix<TPrecision> b(N, Y.M());


      for(int i = 0; index.size() > 0; i++){
        unsigned int nn = index.front();
        TPrecision w = kernelX.f(dists.front());
        A(i, 0) = w;
        for(unsigned int j=0; j< X.M(); j++){
          A(i, j+1) = (X(j, nn)-x(j)) * w;
        }

        for(unsigned int m = 0; m<Y.M(); m++){
          b(i, m) = Y(m, nn) *w;
        }
        index.pop_front();
        dists.pop_front();
      }

      DenseMatrix<TPrecision> sol = Linalg<TPrecision>::LeastSquares(A, b);

      A.deallocate();
      b.deallocate();
      return sol;

    };


    DenseMatrix<TPrecision> LeastSquares(unsigned int n){
      std::list<TPrecision> dists;
      std::list<unsigned int> index;

      TPrecision tmpCutoff = xCutoff;   
      while(dists.size() < (unsigned int) std::min(20, (int)Y.N()-1) ){   
        dists.clear();
        index.clear();   
        for(unsigned int i =0; i<Y.N(); i++){
          if(i==n) continue;
          TPrecision tmp = sl2metric.distance(X, i, X, n);
          if(tmp < tmpCutoff){
            dists.push_back(tmp);
            index.push_back(i);
          }
        }
        tmpCutoff*=2;
      }


      unsigned int N = dists.size();
      DenseMatrix<TPrecision> A(N, X.M()+1);
      DenseMatrix<TPrecision> b(N, Y.M());

      for(unsigned int i=0; i < N; i++, index.pop_front(), dists.pop_front()){
        unsigned int nn = index.front();
        TPrecision w = kernelX.f(dists.front());
        A(i, 0) = w;
        for(unsigned int j=0; j< X.M(); j++){
          A(i, j+1) = (X(j, nn)-X(j, n)) * w;
        }

        for(unsigned int m = 0; m<Y.M(); m++){
          b(i, m) = Y(m, nn) *w;
        }
      }

      DenseMatrix<TPrecision> sol = Linalg<TPrecision>::LeastSquares(A, b);

      A.deallocate();
      b.deallocate();
      return sol;

    };



    //helper functions
    void updateKNNX(){
      Geometry<TPrecision>::computeKNN(X, KNNX, KNNXD, sl2metric);
    };


    void computeYp(){ 
      DenseVector<TPrecision> tmp(Y.M());
      for(unsigned int i=0; i<Y.N(); i++){
        g(i, tmp);
        Linalg<TPrecision>::SetColumn(Yp, i, tmp);
      }
      tmp.deallocate();
    };







    //numerical gradient computation
 //Gradient descent for all points 
    void gradDescent(unsigned int nIterations, TPrecision scaling, int verbose=1){


      TPrecision objPrev = mse();

      if(verbose > 0){
        std::cout << "Mse start: " << objPrev << std::endl;
      }


      //---Storage for syncronous updates 
      DenseMatrix<TPrecision> sync(X.M(), X.N());

      //---Do nIterations of gradient descent     
      DenseMatrix<TPrecision> Xtmp(X.M(), X.N());
      DenseMatrix<TPrecision> Xswap;

      //gradient direction
      DenseVector<TPrecision> gx(X.M());     
      if(verbose > 0){
        std::cout << "Start Gradient Descent" << std::endl << std::endl;
      }

      for(unsigned int i=0; i<nIterations; i++){
        //compute gradient for each point
        TPrecision maxL = 0;
        for(unsigned int j=0; j < X.N(); j++){
          //compute gradient
          //gradX(j, gx);
          numGradX(j, gx, sX/10.f);

          //store gradient for syncronous updates
          TPrecision l = Linalg<TPrecision>::Length(gx);
          if(maxL < l){
            maxL = l;
          }
          //Linalg<TPrecision>::Scale(gx, 1.f/l, gx);
          for(unsigned int k=0; k<X.M(); k++){
            sync(k, j) = gx(k);
          }
        }



        //sync updates
        TPrecision s;
        if(maxL == 0 )
          s = scaling;
        else{
          s = scaling * sX/maxL;
        }     
        if(verbose > 1){
          std::cout << "scaling: " << s << std::endl;
        }


        //Approximate line search with quadratic fit
        DenseMatrix<TPrecision> A(3, 3);
        DenseMatrix<TPrecision> b(3, 1);
        Linalg<TPrecision>::Zero(A);

        b(0, 0) = mse();
        Linalg<TPrecision>::AddScale(X, -1*s, sync, Xtmp);
        Xswap = X;
        X = Xtmp;
        Xtmp = Xswap;
        update();

        b(1, 0) = mse();
        Linalg<TPrecision>::AddScale(Xswap, -2*s, sync, X);
        update();

        b(2, 0) = mse();

        if(verbose > 1){
          std::cout << "line search: " << std::endl;
          std::cout << b(0, 0) << std::endl;
          std::cout << b(1, 0) << std::endl;
          std::cout << b(2, 0) << std::endl;
        }

        A(0, 2) = 1;
        A(1, 0) = 1*s*s;
        A(1, 1) = -1*s;
        A(1, 2) = 1;
        A(2, 0) = 4*s*s;
        A(2, 1) = -2*s;
        A(2, 2) = 1;

        DenseMatrix<TPrecision> q = Linalg<TPrecision>::Solve(A, b);

        //do step
        if( q(0, 0) > 0){
          TPrecision h = -q(1, 0)/(2*q(0, 0));
          if(h < -2*s){
            h = -2*s;
          }
          else if( h > 1){
            h = 1;
          }
          Linalg<TPrecision>::AddScale(Xtmp, h, sync, X);
        }
        else if( b(0,0) > b(1, 0) ){
          //do nothing step to -10*s
        }
        else{
          //stop gradient descent - no step
          Xswap = Xtmp;
          Xtmp = X;
          X = Xswap;
          //Linalg<TPrecision>::AddScale(Ztmp, -s, sync, Z);
        }

        A.deallocate();
        b.deallocate();
        q.deallocate();


        update();

        TPrecision obj = mse(ortho); 
        if(verbose > 0){
          std::cout << "Iteration: " << i << std::endl;
          std::cout << "MSE: " <<  obj << std::endl;     
          std::cout << "Ortho: " <<  ortho << std::endl << std::endl;
        }   
        if(objPrev <= obj){// || orthoPrev >= ortho){
          break;
        }
        objPrev = obj;      

        }


        //cleanup 
        sync.deallocate();
        gx.deallocate();
        Ztmp.deallocate();

      };

    TPrecision mse(int index){
      TPrecision e = 0;
      DenseVector<TPrecision> gfy(Y.M());
      //leave one out
      for(unsigned int i=1; i < knnX; i++){
        int nn = KNNX(i, index);
        g(nn, gfy);
        e += sl2metric.distance(Y, nn, gfy); 
      }
      gfy.deallocate();
      return e/(knnX-1);
    };


    virtual void numGradX(int r, DenseVector<TPrecision> &gx, TPrecision epsilon){

      TPrecision eg = 0;
      TPrecision e = mse(r);
      for(unsigned int i=0; i<gx.N(); i++){
        X(i, r) += epsilon;
        eg = mse(r);
        gx(i) = ( eg - e ) / epsilon;
        X(i, r) -= epsilon;
      }
    };



    // orthogonal projections within current boundary of x
    void boundary(DenseVector<TPrecision> &zmin, DenseVector<TPrecision> &zmax){
      for(unsigned int i=0; i< zmin.N(); i++){
        zmin(i) = std::numeric_limits<TPrecision>::max();
        zmax(i) = std::numeric_limits<TPrecision>::min();
      }
      for(unsigned int i=0; i< X.N(); i++){
        for(unsigned int k=0; k<zmin.N(); k++){
          if(zmin(k) > X(k, i)){
            zmin(k) = X(k, i);
          }
          if(zmax(k) < X(k, i)){
            zmax(k) = X(k, i);
          }
        }
      } 
    };


    DenseMatrix<TPrecision> project(DenseMatrix<TPrecision> &Ydata){
      computeYp();
      DenseVector<TPrecision> y(Ydata.M());
      DenseVector<TPrecision> yout(Ydata.M());
      DenseVector<TPrecision> x(X.M());     
      DenseVector<TPrecision> zmin(X.M());
      DenseVector<TPrecision> zmax(X.M());
      boundary(zmin, zmax);
      DenseMatrix<TPrecision> result(Y.M(), Ydata.N());
      for(unsigned int i=0; i<Ydata.N(); i++){
        Linalg<TPrecision>::ExtractColumn(Ydata, i, y);
        project(y, x, yout, sX/10.f, zmin, zmax);
        Linalg<TPrecision>::SetColumn(result, i, yout); 
      }
      return result;
    };



    TPrecision project(DenseVector<TPrecision> &y, DenseVector<TPrecision> &xout,
        DenseVector<TPrecision> &yout, TPrecision epsilon, DenseVector<TPrecision>
        &zmin, DenseVector<TPrecision> &zmax){


      DenseVector<int> knn(1); 
      DenseVector<TPrecision> knnDist(1);
      Geometry<TPrecision>::computeKNN(Yp, y, knn, knnDist, sl2metric);

      Linalg<TPrecision>::ExtractColumn(X, knn(0), xout);
      g(xout, yout);
      TPrecision sdist = sl2metric.distance(y, yout);

      DenseVector<TPrecision> gx(xout.N());
      DenseVector<TPrecision> tmp(xout.N());
      for(int i=0; i<1000; i++){
        //compute gradient direction
        for(unsigned int k=0; k< xout.N(); ++k){
          xout(k) += epsilon;
          g(xout, yout);
          TPrecision d = sl2metric.distance(y, yout);
          gx(k) = (d - sdist)/epsilon;
          xout(k) -= epsilon;
        }
        TPrecision l = Linalg<TPrecision>::Length(gx);
        Linalg<TPrecision>::Scale(gx, 0.01*sX/l, gx);
        Linalg<TPrecision>::Subtract(xout, gx, tmp);
        int nb = 0;
        for(unsigned int k=0;k<xout.N(); k++){
          if(tmp(k) < zmin(k)){
            tmp(k) = zmin(k);
            nb++;
          }
          if(tmp(k) > zmax(k)){
            tmp(k) = zmax(k);
            nb++;
          }
        }
        g(tmp, yout);
        TPrecision d = sl2metric.distance(y, yout);

        if(d>=sdist){
          g(xout, yout);
          break;
        }
        sdist = d;
        Linalg<TPrecision>::Copy(tmp, xout); 
        if(nb == 2){
          break;
        }
      }

      gx.deallocate();
      tmp.deallocate();
      knn.deallocate();
      knnDist.deallocate();
      return sdist;
    };









}; 


#endif


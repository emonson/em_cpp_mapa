#ifndef UKR_H
#define UKR_H


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
class UKR{

  protected:    
    DenseMatrix<TPrecision> Y;
    DenseMatrix<TPrecision> Yp;
    DenseMatrix<TPrecision> Z;

    DenseMatrix<int> KNNX;
    DenseMatrix<TPrecision> KNNXD;


    unsigned int knnSigma;
    
    EuclideanMetric<TPrecision> l2metric;
    SquaredEuclideanMetric<TPrecision> sl2metric;

    
    GaussianKernel<TPrecision> kernelX;
  
  
  private:


    TPrecision sX;
    TPrecision xCutoff;

    bool verbose;


  public:
  
   void cleanup(){      
    KNNX.deallocate();
    KNNXD.deallocate();
    Y.deallocate();
    Z.deallocate();

   };

   //Create KernelMap 
   UKR(DenseMatrix<TPrecision> Ydata, DenseMatrix<TPrecision> Zinit, 
       TPrecision alpha, unsigned int nnSigma) :
       Y(Ydata), Z(Zinit), knnSigma(nnSigma){
     
     verbose = true;//false;
     init();
     computeKernelX(alpha);
     xCutoff = 3*kernelX.getKernelParam();
     xCutoff = xCutoff*xCutoff;
   };



   UKR(DenseMatrix<TPrecision> Ydata, DenseMatrix<TPrecision> Zopt, 
       unsigned int nnSigma, double sigmaX): Y(Ydata),
       Z(Zopt), knnSigma(nnSigma){

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
     DenseMatrix<TPrecision> J(Y.M(), Z.M());
     
     //Temp vars.
     DenseVector<TPrecision> gfy(Y.M());
     DenseVector<TPrecision> diff(Y.M());
     DenseVector<TPrecision> pDot(Z.M());

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
     o = o/(Z.M()*Z.N())/ M_PI * 180;
          
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
  

   
     


   void boundary(DenseVector<TPrecision> &zmin, DenseVector<TPrecision> &zmax){
      for(unsigned int i=0; i< zmin.N(); i++){
        zmin(i) = std::numeric_limits<TPrecision>::max();
        zmax(i) = std::numeric_limits<TPrecision>::min();
      }
      for(unsigned int i=0; i< Z.N(); i++){
        for(unsigned int k=0; k<zmin.N(); k++){
          if(zmin(k) > Z(k, i)){
            zmin(k) = Z(k, i);
          }
          if(zmax(k) < Z(k, i)){
            zmax(k) = Z(k, i);
          }
        }
      } 
   };

   void minimize(int nIter = 100){
     DenseVector<TPrecision> y(Y.M());
     DenseVector<TPrecision> yout(Y.M());
     DenseVector<TPrecision> x(Z.M());
     DenseMatrix<TPrecision> Ztmp(Z.M(), Z.N());
     DenseVector<TPrecision> zmin(Z.M());
     DenseVector<TPrecision> zmax(Z.M());
     boundary(zmin, zmax);

     TPrecision msep = std::numeric_limits<TPrecision>::max();
     for(int i=0; i<nIter; i++){
        computeYp();          
        TPrecision mse = 0;
        for(unsigned int i=0; i<Y.N(); i++){
  	      Linalg<TPrecision>::ExtractColumn(Y, i, y);
          mse += project(y, x, yout, sX/10.f, zmin, zmax);
          Linalg<TPrecision>::SetColumn(Ztmp, i, x);
        }
        mse /= Y.N();
        std::cout << "MSE: " << mse << std::endl;
        if(mse > msep){ break; }
        
        Linalg<TPrecision>::Copy(Ztmp, Z);
        msep = mse;

      }

   };




   //Gradient descent for all points 
   void gradDescent(unsigned int nIterations, TPrecision scaling){
     
     TPrecision orthoPrev =0;// = orthogonality();
     TPrecision ortho;
     TPrecision objPrev = mse(orthoPrev);
          
     if(verbose){
       std::cout << "Mse start: " << objPrev << std::endl;
       std::cout << "Ortho start: " << orthoPrev << std::endl;
     }


    
     
     //---Storage for syncronous updates 
     DenseMatrix<TPrecision> sync(Z.M(), Z.N());

     //---Do nIterations of gradient descent     
     
     DenseMatrix<TPrecision> Ztmp(Z.M(), Z.N());
     DenseMatrix<TPrecision> Zswap;

     //gradient direction
     DenseVector<TPrecision> gx(Z.M());     
     if(verbose){
     std::cout << "gradDescent all:" << std::endl;
     }

     for(unsigned int i=0; i<nIterations; i++){
      updateKNNX();
      //compute gradient for each point
      TPrecision maxL = 0;
      for(unsigned int j=0; j < Z.N(); j++){
        //compute gradient
        //gradX(j, gx);
	      gradX(j, gx, sX/10.f);
        
        //store gradient for syncronous updates
        TPrecision l = Linalg<TPrecision>::Length(gx);
	      if(maxL < l){
          maxL = l;
	      }
	      //Linalg<TPrecision>::Scale(gx, 1.f/l, gx);
        for(unsigned int k=0; k<Z.M(); k++){
          sync(k, j) = gx(k);
        }
      }
      std::cout << std::endl;


      //LinalgIO<TPrecision>::writeMatrix("G.data", sync);

      //sync updates
      TPrecision s;
      if(maxL == 0 )
	      s = scaling;
      else{
	      s = scaling * sX/maxL;
      }     
      if(verbose){
        std::cout << "scaling: " << s << std::endl;
      }
      
      
      
      //Approximate line search with quadratic fit
      DenseMatrix<TPrecision> A(3, 3);
      DenseMatrix<TPrecision> b(3, 1);
      Linalg<TPrecision>::Zero(A);

      b(0, 0) = mse();
      std::cout << b(0, 0) << std::endl;
      Linalg<TPrecision>::AddScale(Z, -1*s, sync, Ztmp);
      Zswap = Z;
      Z = Ztmp;
      Ztmp = Zswap;
      
      b(1, 0) = mse();
      std::cout << b(1, 0) << std::endl;
      Linalg<TPrecision>::AddScale(Zswap, -2*s, sync, Z);
      
      b(2, 0) = mse();
      std::cout << b(2, 0) << std::endl;
        
      A(0, 2) = 1;
      A(1, 0) = 1;
      A(1, 1) = -1;
      A(1, 2) = 1;
      A(2, 0) = 4;
      A(2, 1) = -2;
      A(2, 2) = 1;

      DenseMatrix<TPrecision> q = Linalg<TPrecision>::Solve(A, b);

      //do step
      if( q(0, 0) > 0){
        TPrecision h = -q(1, 0)/(2*q(0, 0));
        if(h < -2){
         h = -2;
        }
        else if( h > 1){
          h = 1;
        }
        Linalg<TPrecision>::AddScale(Ztmp, h*s, sync, Z);
      }
      else if( b(0,0) > b(1, 0) ){
        //do nothing step to -10*s
      }
      else{
        //small step
        Linalg<TPrecision>::AddScale(Ztmp, -s, sync, Z);
      }

      A.deallocate();
      b.deallocate();
      q.deallocate();
      

      std::cout << std::endl;
      TPrecision obj = mse(ortho); 
      if(verbose){
      std::cout << "Iteration: " << i << std::endl;
      std::cout << "MSE: " <<  obj << std::endl;     
      std::cout << "Ortho: " <<  ortho << std::endl;  
      }   
      if(objPrev < obj){
        break;
      }
      objPrev = obj;
      orthoPrev = ortho;
     }

     //DEBUG
     //DenseMatrix<TPrecision> Ytest =
     //LinalgIO<TPrecision>::readMatrix("Ytest.data.hdr");
     //DEBUG
     //DenseMatrix<TPrecision> Ygt =
     //LinalgIO<TPrecision>::readMatrix("Ygt.data.hdr");


    // computeYp();
    // DenseVector<TPrecision> y(Ytest.M());
    // DenseVector<TPrecision> yout(Ytest.M());
   //  DenseVector<TPrecision> x(Z.M());     
   //  DenseVector<TPrecision> zmin(Z.M());
    // DenseVector<TPrecision> zmax(Z.M());
    // boundary(zmin, zmax);
   //  TPrecision mset = 0;
   //  for(unsigned int i=0; i<Ytest.N(); i++){
  //	    Linalg<TPrecision>::ExtractColumn(Ytest, i, y);
   //     project(y, x, yout, sX/10.f, zmin, zmax); 
  //      mset += sl2metric.distance(Ygt, i, yout);
  //   }
  //   std::cout << "MSE-Test: " << mset/Ytest.N() << std::endl;


     //cleanup 
     sync.deallocate();
     gx.deallocate();
     Ztmp.deallocate();
   
   };
 
   
   
   DenseMatrix<TPrecision> parametrize(DenseMatrix<TPrecision> &Ydata){
       computeYp();
       DenseVector<TPrecision> y(Ydata.M());
       DenseVector<TPrecision> yout(Ydata.M());
       DenseVector<TPrecision> x(Z.M());     
       DenseVector<TPrecision> zmin(Z.M());
       DenseVector<TPrecision> zmax(Z.M());
       boundary(zmin, zmax);
       DenseMatrix<TPrecision> result(Z.M(), Ydata.N());
       for(unsigned int i=0; i<Ydata.N(); i++){
    	   Linalg<TPrecision>::ExtractColumn(Ydata, i, y);
         project(y, x, yout, sX/10.f, zmin, zmax);
         Linalg<TPrecision>::SetColumn(result, i, x); 
       }
       return result;
   };
    
   DenseMatrix<TPrecision> project(DenseMatrix<TPrecision> &Ydata){
       computeYp();
       DenseVector<TPrecision> y(Ydata.M());
       DenseVector<TPrecision> yout(Ydata.M());
       DenseVector<TPrecision> x(Z.M());     
       DenseVector<TPrecision> zmin(Z.M());
       DenseVector<TPrecision> zmax(Z.M());
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
    
    Linalg<TPrecision>::ExtractColumn(Z, knn(0), xout);
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
     for(unsigned int i=0; i< Z.M(); i++){
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
     for(unsigned int i=0; i< Z.M(); i++){
       for(unsigned int j=0; j< Y.M(); j++){
         J(j, i) = sol(1+i, j);
       }
     }

     sol.deallocate();
   };



   //numerical gradient computation
   void gradX(int r, DenseVector<TPrecision> &gx, TPrecision epsilon){
    TPrecision eg = 0;
    TPrecision e = mse(r);
    DenseVector<TPrecision> fy(Z.M());
    for(unsigned int i=0; i<gx.N(); i++){
      Z(i, r) += epsilon;
      eg = mse(r);
      gx(i) = ( eg - e ) / epsilon;
      Z(i, r) -= epsilon;
    }
    
   };


   //get original Y's
   DenseMatrix<TPrecision> getY(){
     return Y;
   };
   
   //get Z (parameters for f
   DenseMatrix<TPrecision> getZ(){
     return Z;
   };




  
   //Compute the log probability of Yt_i belonging to this manifold, by computing a
   //local variance orthogonal to the manifold. The probability is the a
   //gaussian according to this variance and mean zero off the manifold
   // -Yt  data to test
   // -Ytp projection of Yt onto this manifold
   // -Xt  manifold paprametrization of Yt
   // -p   output of pdf values for each point basde on projection distance
   //      (variance of the manifold at Ytp) 
   // -var variances (off the amnifold) for each point
   // -pk  kernel density based on the coordinate mapping
   void pdf(DenseMatrix<TPrecision> Yt, DenseMatrix<TPrecision> Ytp,
       DenseMatrix<TPrecision> Xt,
       DenseVector<TPrecision> &p, DenseVector<TPrecision> &var,
       DenseVector<TPrecision> &pk, bool useDensity){

     //update fY if necessary

     DenseMatrix<TPrecision> Xpr = parametrize(Y);
     DenseMatrix<TPrecision> Ypr = reconstruct(Xpr); 
     
     TPrecision cod = (TPrecision) (Y.M() - Z.M())/2.0;

     //compute trainig set squared distances
     DenseVector<TPrecision> sdist(Y.N());
     for(unsigned int i=0; i< Y.N(); i++){
       sdist(i) = sl2metric.distance(Y, i, Ypr, i);
     }

     DenseVector<TPrecision> k(Y.N());

     //compute variances and pdf values
     TPrecision c = -cod/2.f * log(2*M_PI);
     for(unsigned int i=0; i < Xt.N(); i++){

       TPrecision sum = 0;
       TPrecision vartmp = 0;
       for(unsigned int j=0; j < Xpr.N(); j++){
         k(j) = kernelX.f(Xt, i, Xpr, j);
         sum += k(j);
         vartmp += sdist(j) * k(j); 
       } 
       var(i) = vartmp / sum;
       
       TPrecision d = sl2metric.distance(Yt, i, Ytp, i);
       p(i) = c - cod/2.f * log(var(i)) - d / ( 2.f * var(i) ) ;
     }
     
     if(useDensity){
       KernelDensity<Precision> kd(Xpr, kernelX);
       for(unsigned int i=0; i<Xt.N(); i++){
         pk(i) = kd.p(Xt, i, true);
       }
     }


     sdist.deallocate(); 
     k.deallocate();
   };





//Mean curvature vector - see "Curvature Computations for n-manifolds in
   //R^{n+m} and solution to an open problem proposed by R. Goldman", Qin Zhang
   //and Guoliang Xu 
   DenseVector<TPrecision> curvature(DenseVector<TPrecision> &x, TPrecision
       &meanC, TPrecision &gaussC, TPrecision &detg, TPrecision &detB,
       TPrecision &frob){
     
     //Block hessian matrix
     DenseMatrix<TPrecision> T[Z.M()];
     for(unsigned int i=0; i<Z.M(); i++){
       T[i] = DenseMatrix<Precision>(Y.M(), Z.M());     
       Linalg<TPrecision>::Zero(T[i]);
     }
        
     //Jacobian of g(x)
     DenseMatrix<TPrecision> J(Y.M(), Z.M());
     Linalg<TPrecision>::Zero(J); 
     
     //Kernel values
     DenseVector<TPrecision> k(Z.N());
     //Kernel gradients
     DenseMatrix<TPrecision> kg(Z.M(), Z.N());
     //Kernel hessians
     DenseMatrix<TPrecision> kh[Z.N()];
     
     //Precomputed sums of kernel value
     TPrecision sumk = 0;
     DenseVector<TPrecision> sumkg(Z.M());
     Linalg<TPrecision>::Zero(sumkg);
     DenseMatrix<TPrecision> sumkh(Z.M(), Z.M());
     Linalg<TPrecision>::Zero(sumkh);

     //Temp vars.
     DenseVector<TPrecision> gtmp(Z.M());
     DenseVector<TPrecision> z(Z.M());


     //Precompte kernel values and first and second order derivatives 
     //(gradient and hessian of kernel function)
     
     for(unsigned int i=0; i < Z.N(); i++){
       Linalg<TPrecision>::ExtractColumn(Z, i, z);

       k(i) = kernelX.gradf(x, z, gtmp);
       kh[i] = kernelX.hessian(x, z);
       sumk += k(i);

        for(unsigned int k=0; k < Z.M(); k++){
          kg(k, i) = gtmp(k);
          sumkg(k) += kg(k, i);
          for( unsigned int l=0; l<Z.M(); l++){
            sumkh(k, l) += kh[i](k, l);
          }
        }
      }

      TPrecision sumk2 = sumk*sumk;
      TPrecision sumk4 = sumk2 * sumk2;

      //Build T - block hessian matrix and J - Jacobian
      for(unsigned int n = 0; n < Z.N(); n++){
        for(unsigned int i=0; i<Z.M(); i++){
          
          //T
          for(unsigned int j = 0; j<Z.M(); j++){
            //First order and second order kernel derivatives dx_i, dx_j
            TPrecision c  = kh[n](i, j) * sumk      +  kg(j, n) * sumkg(i)   ;
                       c -= kg(i, n)    * sumkg(j)  +  k(n)     * sumkh(i, j);
                       c *= sumk2;
                       c -= (kg(j, n) * sumk        -  k(n)     * sumkg(j)) * 2.0 * sumk * sumkg(i);
            //times dependant variable y
            for(unsigned int r=0; r < Y.M() ; r++){
              T[i](r, j) +=  c * Y(r, n);
            }
          }
          
          //J
          TPrecision c  = kg(i, n) * sumk      -  k(n) * sumkg(i)   ;
          for(unsigned int r=0; r< Y.M(); r++){
            J(r, i) += c *Y(r, n);
          }
        }
      }    
   
     Linalg<TPrecision>::Scale(J, 1.0/sumk2, J);

     for(unsigned int i=0; i<Z.M(); i++){
      Linalg<TPrecision>::Scale(T[i], 1.0/sumk4, T[i]);
     }


     //Measure tensor / first fundamental form
     DenseMatrix<TPrecision> g = Linalg<TPrecision>::Multiply(J, J, true, false);
     DenseMatrix<TPrecision> gInv = Linalg<TPrecision>::Inverse(g);
    
     
     //Curvatures
     
     //Normal projection
     DenseMatrix<TPrecision> Q1 = Linalg<TPrecision>::Multiply(gInv, J, false, true);
     DenseMatrix<TPrecision> Q = Linalg<TPrecision>::Multiply(J, Q1);
     Linalg<TPrecision>::Scale(Q, -1.0, Q);

     for(unsigned int i = 0; i < Q.M(); i++){
       Q(i, i) = 1 + Q(i, i);
     }
     
     //Block trace of TgInv
     DenseMatrix<TPrecision> TgInv[Z.M()];
     DenseMatrix<TPrecision> QTgInv[Z.M()];
     for(unsigned int i=0; i < Z.M(); i++){
      TgInv[i] = Linalg<TPrecision>::Multiply(T[i], gInv);
      QTgInv[i] = Linalg<TPrecision>::Multiply(Q, TgInv[i]);
     }
     DenseVector<TPrecision> H1(Y.M());
     Linalg<TPrecision>::Zero(H1);
     frob = 0;
     for(unsigned int i = 0; i < Z.M(); i++){
        for(unsigned int r = 0; r < Y.M(); r++){
           H1(r) += TgInv[i](r, i);
           for(unsigned int q=0; q< Z.M(); q++){
            frob += QTgInv[i](r, q) * QTgInv[i](r, q); 
           }
        }
     }
     frob = sqrt(frob);
     
     DenseVector<TPrecision> H = Linalg<TPrecision>::Multiply(Q, H1);
     Linalg<TPrecision>::Scale(H, 1.0/ Z.M(), H);
     
     //Mean curvature
     meanC = Linalg<TPrecision>::Length(H);
     
     
     //Gauss curvature
     Linalg<TPrecision>::Scale(H, 1.0/meanC, H);
    
     DenseMatrix<TPrecision> B(Z.M(), Z.M());
     for(unsigned int i=0; i<B.M(); i++){
        for(unsigned int j=0; j< B.N(); j++){
          TPrecision dot = 0;
          for(unsigned int r=0; r<Y.M(); r++){
            dot += T[i](r, j) * H(r);
          }
          B(i, j) = dot;
        }
     }
     
     detB = Linalg<TPrecision>::Det(B);
     detg = Linalg<TPrecision>::Det(g);
     
     gaussC =  detB / detg;


     //cleanup
     Q.deallocate();
     Q1.deallocate();
     for(unsigned int i=0; i < Z.M(); i++){
      TgInv[i].deallocate();
      QTgInv[i].deallocate();
      T[i].deallocate();
     }
     g.deallocate();
     gInv.deallocate();
     H1.deallocate();
     B.deallocate();
     J.deallocate();
     gtmp.deallocate();
     k.deallocate();
     kg.deallocate();
     for(unsigned int i=0; i< Y.N(); i++){
        kh[i].deallocate();
     }
     sumkg.deallocate();
     sumkh.deallocate();
  
      
     return H;

   };



  


   DenseMatrix<TPrecision> reconstruct(DenseMatrix<TPrecision> &Xpoints){
     DenseMatrix<TPrecision> proj(Y.M(), Xpoints.N());
     reconstruct(Xpoints, proj);     
     return proj;
   };



  

   void reconstruct(DenseMatrix<TPrecision> &Xpoints, DenseMatrix<TPrecision> &proj){
     
     DenseVector<TPrecision> tmp(Z.M()); 
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


   void init(){

     Yp = Linalg<TPrecision>::Copy(Y);
     if(verbose){
      std::cout << "Initalizing" << std::endl;
     }    
     
     if(knnSigma > Y.N()){
       knnSigma = Y.N();
     }
     KNNX = DenseMatrix<int>(knnSigma, Z.N());
     KNNXD = DenseMatrix<TPrecision>(knnSigma, Z.N());
     updateKNNX();
    
     kernelX = GaussianKernel<TPrecision>( Z.M());
     

     if(verbose){
      std::cout << "Computing knn for Y" << std::endl;
     }


     if(verbose){
      std::cout << "Computing kernel values for Y" << std::endl;
     }

     if(verbose) {   
      std::cout << "Initalizing done" << std::endl;
     }

   };     


  DenseMatrix<TPrecision> LeastSquares(Vector<TPrecision> &x){
    std::list<TPrecision> dists;
    std::list<unsigned int> index;
        
    TPrecision tmpCutoff = xCutoff;   
    while(dists.size() < (unsigned int) std::min(20, (int)Y.N()) ){   
      dists.clear();
      index.clear();   
      for(unsigned int i =0; i<Y.N(); i++){
        TPrecision tmp = sl2metric.distance(Z, i, x);
        if(tmp < tmpCutoff){
          dists.push_back(tmp);
          index.push_back(i);
        }
      }
      tmpCutoff*=2;
    }


   

    unsigned int N = dists.size();
    DenseMatrix<TPrecision> A(N, Z.M()+1);
    DenseMatrix<TPrecision> b(N, Y.M());


    for(int i = 0; index.size() > 0; i++){
       unsigned int nn = index.front();
       TPrecision w = kernelX.f(dists.front());
       A(i, 0) = w;
       for(unsigned int j=0; j< Z.M(); j++){
         A(i, j+1) = (Z(j, nn)-x(j)) * w;
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
        TPrecision tmp = sl2metric.distance(Z, i, Z, n);
        if(tmp < tmpCutoff){
          dists.push_back(tmp);
          index.push_back(i);
        }
      }
      tmpCutoff*=2;
    }


    unsigned int N = dists.size();
    DenseMatrix<TPrecision> A(N, Z.M()+1);
    DenseMatrix<TPrecision> b(N, Y.M());

    for(unsigned int i=0; i < N; i++, index.pop_front(), dists.pop_front()){
       unsigned int nn = index.front();
       TPrecision w = kernelX.f(dists.front());
       A(i, 0) = w;
       for(unsigned int j=0; j< Z.M(); j++){
         A(i, j+1) = (Z(j, nn)-Z(j, n)) * w;
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



  void updateKNNX(){
     Geometry<TPrecision>::computeKNN(Z, KNNX, KNNXD, sl2metric);
  };


  void computeYp(){ 
      DenseVector<TPrecision> tmp(Y.M());
      for(unsigned int i=0; i<Y.N(); i++){
        g(i, tmp);
        Linalg<TPrecision>::SetColumn(Yp, i, tmp);
      }
      tmp.deallocate();
  };



  void computeKernelX(TPrecision alpha){
    TPrecision sigma = 0;
    sX = 0;
    for(unsigned int i=0; i < Z.N(); i++){
      sigma += sqrt( KNNXD(knnSigma-1, i) );
    }
    sigma *= alpha/Z.N();
    sX = sigma/alpha;
    

    if(verbose){
      std::cout << "sigmaX: " << sigma << std::endl;
      std::cout << "scale: " << sX << std::endl;
    }

    kernelX.setKernelParam(sigma);
  };


  
}; 


#endif


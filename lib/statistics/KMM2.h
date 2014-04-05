#ifndef KMM_H
#define KMM_H


#include "Geometry.h"
#include "Matrix.h"
#include "EuclideanMetric.h"
#include "SquaredEuclideanMetric.h"
#include "KernelDensity.h"
#include "Linalg.h"
#include "LinalgIO.h"
#include "GaussianKernel.h"
#include "EpanechnikovKernel.h"
#include "Random.h"
#include "PCA.h"

#include <list>
#include <iterator>
#include <stdlib.h>
#include <limits>
#include <math.h>


template <typename TPrecision>
class KMM{

  protected:    
    DenseMatrix<TPrecision> Y;
    DenseMatrix<TPrecision> Z;
    DenseMatrix<TPrecision> fY;

    unsigned int knnSigma;
    
    EuclideanMetric<TPrecision> l2metric;
    SquaredEuclideanMetric<TPrecision> sl2metric;

    
    DenseMatrix<TPrecision> KY;
    DenseVector<TPrecision> sumKY;
    DenseMatrix<TPrecision> KYN;


    DenseMatrix<int> KNNY;
    DenseMatrix<TPrecision> KNNYD;

   
    
    GaussianKernel<TPrecision> kernelX;
    GaussianKernel<TPrecision> kernelY;
  
  
  private:
    
    DenseVector<bool> lout;

    Random<TPrecision> random;
    TPrecision sX;
    TPrecision xCutoff;

    bool verbose;
  
  
  
  public:
  
   virtual void cleanup(){      
    KNNY.deallocate();
    KNNYD.deallocate();
    KY.deallocate();
    sumKY.deallocate();
    KYN.deallocate();
    Y.deallocate();
    Z.deallocate();
    fY.deallocate();
   };

   //Create KernelMap 
   KMM(DenseMatrix<TPrecision> Ydata, DenseMatrix<TPrecision> Zinit, 
       TPrecision alpha, unsigned int nnSigma) :
       Y(Ydata), Z(Zinit), knnSigma(nnSigma){
     
     verbose = true;//false;
     init();
     computeKernelX(alpha);
     xCutoff = 3*kernelX.getKernelParam();
     xCutoff = xCutoff*xCutoff;     
     update();  
   };



   KMM(DenseMatrix<TPrecision> Ydata, DenseMatrix<TPrecision> Zopt, 
       unsigned int nnSigma, double sigmaX): Y(Ydata),
       Z(Zopt), knnSigma(nnSigma) {

     verbose = true;//false;
     init();
     kernelX.setKernelParam(sigmaX);
     sX = kernelX.getKernelParam();   
     xCutoff = 3*kernelX.getKernelParam();
     xCutoff = xCutoff*xCutoff;
     update();

     if(verbose){
       std::cout << "knnSigma: " << knnSigma<<std::endl;
       std::cout << "sigmaX: " << kernelX.getKernelParam() <<std::endl;
       std::cout << "sigmaY: " << kernelY.getKernelParam() <<std::endl;
       std::cout << "adaptive: " << false <<std::endl;
     }

   }; 





   //evalue objective function, squared error
   TPrecision mse(){
     TPrecision e = 0;
     DenseVector<TPrecision> gfy(Y.M());
     for(unsigned int i=0; i < Y.N(); i++){
       g(i, gfy);
       e += sl2metric.distance(Y, i, gfy);
     }
     gfy.deallocate();
     return e/Y.N();
   }

   //evalue objective function, squared error
   virtual TPrecision mse(TPrecision &o){
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
         };
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


   virtual TPrecision mse(int index){
     TPrecision e = 0;
     
     
     //Temp vars.
     DenseVector<TPrecision> gfy(Y.M());

     int n =0;
     for(unsigned int i=0; i < knnSigma; i++){
       int nn = KNNY(i, index);
       g(index, gfy);
       e += sl2metric.distance(Y, index, gfy); 
     }
          
     gfy.deallocate();
     return e/knnSigma;
   };
  

   
     





   //Gradient descent for all points 
   void gradDescent(unsigned int nIterations, TPrecision scaling){

     TPrecision orthoPrev =0;
     TPrecision ortho;

     TPrecision objPrev = mse(orthoPrev);

     if(verbose){
       std::cout << "Mse start: " << objPrev << std::endl;
       std::cout << "Ortho start: " << orthoPrev << std::endl;
     }



     //DEBUG
     //DenseMatrix<TPrecision> Ytest =
     //LinalgIO<TPrecision>::readMatrix("Ytest.data.hdr");
     //DEBUG
     //DenseMatrix<TPrecision> Ygt =
     //LinalgIO<TPrecision>::readMatrix("Ygt.data.hdr");

     
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
      startIteration(i);
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
      update();

      b(1, 0) = mse();
      std::cout << b(1, 0) << std::endl;
      Linalg<TPrecision>::AddScale(Zswap, -2*s, sync, Z);
      update();

      b(2, 0) = mse();
      std::cout << b(2, 0) << std::endl;
        
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
        Linalg<TPrecision>::AddScale(Ztmp, h, sync, Z);
      }
      else if( b(0,0) > b(1, 0) ){
        //do nothing step to -10*s
      }
      else{
        //stop gradient descent - no step
        Zswap = Ztmp;
        Ztmp = Z;
        Z = Zswap;
        //Linalg<TPrecision>::AddScale(Ztmp, -s, sync, Z);
      }

      A.deallocate();
      b.deallocate();
      q.deallocate();

      endIteration(i);


      update();

      std::cout << std::endl;
      TPrecision obj = mse(ortho); 
      if(verbose){
        std::cout << "Iteration: " << i << std::endl;
        std::cout << "MSE: " <<  obj << std::endl;     
        std::cout << "Ortho: " <<  ortho << std::endl;
      }   
      if(objPrev <= obj){// || orthoPrev >= ortho){
        break;
      }
      objPrev = obj;      


      //DEBUG
      //DenseMatrix<TPrecision> Xt = parametrize(Ytest);
      //DenseMatrix<TPrecision> Ytp = reconstruct(Xt);

     // TPrecision mset = 0;
      //for(unsigned int k=0; k<Ytp.N(); k++){
     //   mset += sl2metric.distance(Ygt, k, Ytp, k);
     // }
     // std::cout << "MSE-Test: " << mset/Ytp.N() << std::endl;
     // Xt.deallocate();
     // Ytp.deallocate();

     }
     
     


     //cleanup 
     sync.deallocate();
     gx.deallocate();
     Ztmp.deallocate();
   
   };
 

 


  //f(x_index) - reconstruction mapping
  void f(unsigned int index, Vector<TPrecision> &out){
    Linalg<TPrecision>::Zero(out);
    TPrecision sumw = 0;
    for(unsigned int i=0; i < Y.N(); i++){
      //if(i==index) continue;
      TPrecision w = KY(i, index);
      Linalg<TPrecision>::AddScale(out, w, Z, i, out);
      sumw += w;
    }     
    Linalg<TPrecision>::Scale(out, 1.f/sumw, out);
  };




   //f(x) - reconstruction mapping
   void f( DenseVector<TPrecision> &y, Vector<TPrecision> &out){
    Linalg<TPrecision>::Zero(out);
    TPrecision sumw = 0;
    for(unsigned int i=0; i < Y.N(); i++){
      TPrecision w = kernelY.f(y, Y, i);
      Linalg<TPrecision>::AddScale(out, w, Z, i, out);
      sumw += w;
    }     
    Linalg<TPrecision>::Scale(out, 1.f/sumw, out);
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
   virtual void gradX(int r, DenseVector<TPrecision> &gx, TPrecision epsilon){


    
    TPrecision eg = 0;
    TPrecision e = mse(r);
    DenseVector<TPrecision> fy(Z.M());
    for(unsigned int i=0; i<gx.N(); i++){
      Z(i, r) += epsilon;
      //update nearest neighbors
      for(unsigned int k=0; k<knnSigma; k++){
        int nn = KNNY(k, r);
      	f(nn, fy);
        Linalg<TPrecision>::SetColumn(fY, nn, fy);
      }
      //f(r, fy);
      ///Linalg<TPrecision>::SetColumn(fY, r, fy);
      eg = mse(r);
      gx(i) = ( eg - e ) / epsilon;
      Z(i, r) -= epsilon;
    }

    //f(r, fy);
    //Linalg<TPrecision>::SetColumn(fY, r, fy);

    //update nearest neighbors
    for(unsigned int k=0; k<knnSigma; k++){
      int nn = KNNY(k, r);
      f(nn, fy);
      Linalg<TPrecision>::SetColumn(fY, nn, fy);
    }

    fy.deallocate();
   };




   //get original Y's
   DenseMatrix<TPrecision> getY(){
     return Y;
   };
   


   //get Z (parameters for f
   DenseMatrix<TPrecision> getZ(){
     return Z;
   };


   //coordinate mapping fo Ypoints
   DenseMatrix<TPrecision> parametrize(DenseMatrix<TPrecision> &Ypoints){

     DenseMatrix<TPrecision> proj(Z.M(), Ypoints.N());
     parametrize(Ypoints, proj);

     return proj;
   };


   //
   void parametrize(DenseMatrix<TPrecision> &Ypoints, DenseMatrix<TPrecision> &proj){

     DenseVector<TPrecision> tmp(Y.M()); 
     DenseVector<TPrecision> xp(Z.M()); 

     for(unsigned int i=0; i < Ypoints.N(); i++){
       Linalg<TPrecision>::ExtractColumn(Ypoints, i, tmp);
       f(tmp, xp);
       Linalg<TPrecision>::SetColumn(proj, i, xp);
     }
     xp.deallocate();
     tmp.deallocate();
   };

  


   DenseMatrix<TPrecision> &parametrize(){
     return fY;
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




   //Compute the log probability of Yt_i belonging to this manifold, by computing a
   //local variance orthogonal to the manifold. The probability is the a
   //gaussian according to this variance and mean zero off the manifold
   //-Yt  data to test
   //-Ytp projection of Yt onto this manifold
   //-Xt manifold paprametrization of Yt
   //-Yp rpojection of the training data set onto this manifold
   //-p - output of pdf values for each point
   //-var - variances for each point
   void pdf(DenseMatrix<TPrecision> Yt, DenseMatrix<TPrecision> Ytp,
       DenseMatrix<TPrecision> Xt,
       DenseVector<TPrecision> &p, DenseVector<TPrecision> &var,
       DenseVector<TPrecision> &pk, bool useDensity){

     //update fY if necessary

     DenseMatrix<TPrecision> Yp = reconstruct(fY);
     
     TPrecision cod = (TPrecision) (Y.M() - fY.M())/2.0;

     //compute trainig set squared distances
     DenseVector<TPrecision> sdist(Y.N());
     for(unsigned int i=0; i< Y.N(); i++){
       sdist(i) = sl2metric.distance(Y, i, Yp, i);
     }

     DenseVector<TPrecision> k(Y.N());

     //compute variances and pdf values
     TPrecision c = -cod * log(2*M_PI);
     DenseVector<TPrecision> xt(Xt.M());

     for(unsigned int i=0; i < Xt.N(); i++){

       TPrecision sum = 0;
       TPrecision vartmp = 0;
       for(unsigned int j=0; j < fY.N(); j++){
         k(j) = kernelX.f(Xt, i, fY, j);
         sum += k(j);
         vartmp += sdist(j) * k(j); 
       } 
       var(i) = vartmp / sum;
       
       TPrecision d = sl2metric.distance(Yt, i, Ytp, i);
       p(i) = c - cod * log(var(i)) - d / ( 2 * var(i) ) ;
     }
     
     if(useDensity){
       TPrecision n = log(fY.N());
       KernelDensity<Precision> kd(fY, kernelX);
       for(unsigned int i=0; i<p.N(); i++){
         pk(i) = log( kd.p(Xt, i, true)) - n;
       }
     }


     xt.deallocate(); 
     sdist.deallocate(); 
     k.deallocate();
     Yp.deallocate();
   };


   TPrecision getSigmaX(){
     return kernelX.getKernelParam();
   };

   GaussianKernel<Precision> getKernelX(){
     return kernelX;
   };



     protected:
   virtual void startIteration(int iter){};
   virtual void endIteration(int iter){};

private:


   void init(){
     if(verbose){
      std::cout << "Initalizing" << std::endl;
     }
    
     kernelY = GaussianKernel<TPrecision>( Z.M());
     kernelX = GaussianKernel<TPrecision>( Z.M());

     if(knnSigma > Y.N()){
       knnSigma = Y.N();
     }

     if(verbose){
      std::cout << "Computing knn for Y" << std::endl;
     }



     int knn = knnSigma;//std::max(std::max(knnSigma, knnGrad), knnGradEval);
     KNNY =  DenseMatrix<int>(knn+1, Y.N());
     KNNYD = DenseMatrix<TPrecision>(knn+1, Y.N());
     Geometry<TPrecision>::computeKNN(Y, KNNY, KNNYD, sl2metric);
     
     


     if(verbose){
      std::cout << "Computing kernel values for Y" << std::endl;
     }

     computeKY();

     fY = Linalg<TPrecision>::Copy(Z);
     if(verbose) {   
      std::cout << "Initalizing done" << std::endl;
     }

   };     


  DenseMatrix<TPrecision> LeastSquares(Vector<TPrecision> &x){
    std::list<TPrecision> dists;
    std::list<unsigned int> index;
     
    TPrecision tmpCutoff = xCutoff;   
    while(dists.size() < 20){
      dists.clear();
      index.clear();   
      for(unsigned int i =0; i<Y.N(); i++){
        TPrecision tmp = sl2metric.distance(fY, i, x);
        if(tmp < tmpCutoff){
          dists.push_back(tmp);
          index.push_back(i);
        }
      }
      tmpCutoff*=2;
    }


    int N = dists.size();
    DenseMatrix<TPrecision> A(N, Z.M()+1);
    DenseMatrix<TPrecision> b(N, Y.M());

    for(unsigned int i=0; i < N; i++, index.pop_front(), dists.pop_front()){
       unsigned int nn = index.front();
       TPrecision w = kernelX.f(dists.front());
       A(i, 0) = w;
       for(unsigned int j=0; j< Z.M(); j++){
         A(i, j+1) = (fY(j, nn)-x(j)) * w;
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






  DenseMatrix<TPrecision> LeastSquares(unsigned int n){
    std::list<TPrecision> dists;
    std::list<unsigned int> index;
   
    TPrecision tmpCutoff = xCutoff;   
    while(dists.size() < 20){   
      dists.clear();
      index.clear();   
      for(unsigned int i =0; i<Y.N(); i++){
        if(i==n) continue;
        TPrecision tmp = sl2metric.distance(fY, i, fY, n);
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
         A(i, j+1) = (fY(j, nn)-fY(j, n)) * w;
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


  //void computeNNX(){
  //  Geometry<TPrecision>::computeKNN(fY, KNNX, KNNXD, sl2metric);
  //};


  void update(){
      computefY();
  };


  void computefY(){
    DenseVector<TPrecision> tmp(Z.M());
    for(unsigned int i=0; i<Y.N(); i++){
      f(i, tmp);
      Linalg<TPrecision>::SetColumn(fY, i, tmp);
    }
    tmp.deallocate();
  };
 





  void computeKernelX(TPrecision alpha){
    TPrecision sigma = 0;
    DenseMatrix<int> knns(knnSigma, fY.N());
    DenseMatrix<TPrecision> knnDists(knnSigma, fY.N());
    Geometry<TPrecision>::computeKNN(fY, knns, knnDists, sl2metric);
    sX = 0;
    for(unsigned int i=0; i < fY.N(); i++){
      sigma += sqrt( knnDists(knnSigma-1, i) );
      sX += sqrt(knnDists(1, i));

    }
    sigma *= alpha/(fY.N());
    
    sX = sigma;
    

    if(verbose){
      std::cout << "sigmaX: " << sigma << std::endl;
      std::cout << "scale: " << sX << std::endl;
    }

    kernelX.setKernelParam(sigma);
    knns.deallocate();
    knnDists.deallocate();
  };


  

  void computeKY(){
    unsigned int N = Y.N();
    KY = DenseMatrix<TPrecision>(N, N);
    sumKY = DenseVector<TPrecision>(N);
    Linalg<TPrecision>::Set(sumKY, 0);
    KYN = DenseMatrix<TPrecision>(N, N);


    Precision sigma = 0;
    for(unsigned int i=0; i<N; i++){
      sigma += sqrt( KNNYD(knnSigma-1, i) ); 
    }
    sigma /= 3*N;
    if(verbose){
      std::cout << "sigmaY: " << sigma << std::endl;
    }
    kernelY = GaussianKernel<TPrecision>(sigma, Z.M());

    
    if(verbose){
      std::cout << "Compute KY" << std::endl;
    }
    for(unsigned int i=0; i < N; i++){
      for(unsigned int j=0; j < N; j++){
        KY(j, i) = kernelY.f(Y, j, Y, i); 
        sumKY(i) += KY(j, i);
      }
    }

    for(unsigned int i=0; i < KY.M(); i++){
      for(unsigned int j=0; j< KY.N(); j++){
        KYN(i, j) = KY(i, j) / sumKY(j); 
      }
    }
  };

}; 


#endif


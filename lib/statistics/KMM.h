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

#include <stdlib.h>
#include <limits>
#include <math.h>


template <typename TPrecision>
class KMM{

  private:
    DenseMatrix<TPrecision> Y;
    DenseMatrix<TPrecision> Z;
    DenseMatrix<TPrecision> fY;

    unsigned int knnSigma;
    unsigned int knnY;


    DenseMatrix<int> KNNY;
    DenseMatrix<TPrecision> KNNYD;

    EuclideanMetric<TPrecision> l2metric;
    SquaredEuclideanMetric<TPrecision> sl2metric;

    GaussianKernel<TPrecision> kernelX;
    GaussianKernel<TPrecision> kernelY;
    
    DenseMatrix<TPrecision> KY;
    DenseVector<TPrecision> sumKY;
    DenseMatrix<TPrecision> KYN;

    bool Zchanged;


    TPrecision sX;
    TPrecision sX3s;     
    
    DenseVector<TPrecision> wTmp;

  public:
  
   void cleanup(){      
    wTmp.deallocate();
    KNNY.deallocate();
    KNNYD.deallocate();
    KY.deallocate();
    sumKY.deallocate();
    KYN.deallocate();
    Y.deallocate();
    Z.deallocate();
    fY.deallocate();
    delete[] kernelY; 
   };

   //Create KernelMap 
   KMM(DenseMatrix<TPrecision> Ydata, DenseMatrix<TPrecision> Zinit, 
       TPrecision alpha, unsigned int nnSigma, unsigned int nnY) :
       Y(Ydata), Z(Zinit), knnSigma(nnSigma), knnY(nnY){
     
     init();
     computeKernelX(alpha);
   };



   KMM(DenseMatrix<TPrecision> Ydata, DenseMatrix<TPrecision> Zopt, 
       unsigned int nnSigma, unsigned int nnY, double sigmaX): Y(Ydata),
       Z(Zopt), knnSigma(nnSigma), knnY(nnY){

     init();
     kernelX.setKernelParam(sigmaX);   
     sX3s = sigmaX*sigmaX*9;
     sX = sigmaX/3;

     std::cout << "knnSigma: " << knnSigma<<std::endl;
     std::cout << "knnX: " << Y.N() <<std::endl;
     std::cout << "knnY: " << nnY <<std::endl;
     std::cout << "sigmaX: " << kernelX.getKernelParam() <<std::endl;
     std::cout << "sigmY: " << kernelY.getKernelParam() <<std::endl;
     std::cout << "adaptive: " << false <<std::endl;
   }; 


   void init(){
     std::cout << "Initalizing" << std::endl;
     
     wTmp = DenseVector<TPrecision>(Z.N());
     kernelY = GaussianKernel<TPrecision>((int) Y.M());
     sX = 1;
     sX3s = std::numeric_limits<TPrecision>::max();
     if(knnY > Y.N()){
      knnY = Y.N();
     }
     if(knnSigma > Y.N()){
       knnSigma = Y.N();
     }

     std::cout << "Computing knn for Y" << std::endl;
     int knn = std::max(knnSigma, knnY);
     KNNY =  DenseMatrix<int>(knn, Y.N());
     KNNYD = DenseMatrix<TPrecision>(knn, Y.N());
     Geometry<TPrecision>::computeKNN(Y, KNNY, KNNYD, l2metric);

     std::cout << "Computing kernel values for Y" << std::endl;
     computeKY();

     fY = DenseMatrix<TPrecision>(Z.M(), Z.N()); 
     Zchanged = true;
     computefY();    

     std::cout << "Initalizing done" << std::endl;
   };     
   

   //evalue objective function, squared error
   TPrecision evaluate(){
     TPrecision e = 0;
     DenseVector<TPrecision> gfy(Y.M());
     for(unsigned int i=0; i < Y.N(); i++){
       g(i, gfy);
       e += sl2metric.distance(Y, i, gfy); 
     }
     gfy.deallocate();
     return e;
   };



   TPrecision orthogonality(){
     TPrecision o = 0;

     computefY();  
     
     //Jacobian of g(x)
     DenseMatrix<TPrecision> J(Y.M(), Z.M());
     
     //Kernel values
     DenseVector<TPrecision> k(Z.N());
     //Kernel gradients
     DenseMatrix<TPrecision> kg(Z.M(), Z.N());
     
     //Precomputed sums of kernel value
     TPrecision sumk = 0;
     DenseVector<TPrecision> sumkg(Z.M());

     //Temp vars.
     DenseVector<TPrecision> gfy(Y.M());
     DenseVector<TPrecision> diff(Y.M());
     DenseVector<TPrecision> gtmp(Z.M());
     DenseVector<TPrecision> pDot(Z.M());
     Linalg<TPrecision>::Zero(pDot);
    
     DenseMatrix<TPrecision> ortho(Z.M(), Z.N());

     for(unsigned int i=0; i < Y.N(); i++){
       g(i, gfy);

       //Precompte kernel values and gradient 
       sumk = 0;     
       Linalg<TPrecision>::Zero(sumkg);
       Linalg<TPrecision>::Zero(J); 

       for(unsigned int j=0; j < fY.N(); j++){

         k(j) = kernelX.gradf(fY, i, fY, j, gtmp);
         sumk += k(j);

         for(unsigned int k=0; k < Z.M(); k++){
           kg(k, j) = gtmp(k);
           sumkg(k) += kg(k, j);
         }
       }

       TPrecision sumk2 = sumk*sumk;

       for(unsigned int n = 0; n < fY.N(); n++){
         for(unsigned int j=0; j<Z.M(); j++){
           TPrecision c  = kg(j, n) * sumk      -  k(n) * sumkg(j)   ;
           for(unsigned int r=0; r< Y.M(); r++){
             J(r, j) += c *Y(r, n);
           }
         }
       }  
       Linalg<TPrecision>::Scale(J, 1.0/sumk2, J); 
       
       Linalg<TPrecision>::Subtract(gfy, Y, i, diff);  
   
       for(unsigned int n=0; n< J.N(); n++){
         TPrecision norm = 0;
         for(unsigned int j=0; j<J.M(); j++){
           norm += J(j, n) * J(j, n);
         }
         norm = sqrt(norm);
         for(unsigned int j=0; j<J.M(); j++){
           J(j, n) /= norm;
         }
       }

       Linalg<TPrecision>::Normalize(diff);
       Linalg<TPrecision>::Multiply(J, diff, pDot, true);
       Linalg<TPrecision>::SetColumn(ortho, i, pDot);

       for(unsigned int n=0; n< pDot.N(); n++){
        o += acos(sqrt(pDot(n)*pDot(n)));
       } 
     }

     LinalgIO<TPrecision>::writeMatrix("ortho.data", ortho);
     ortho.deallocate();

     pDot.deallocate();
     gfy.deallocate();
     gtmp.deallocate();
     k.deallocate();
     kg.deallocate();
     sumkg.deallocate();
     diff.deallocate();
     J.deallocate();

     return o/(Z.M()*Z.N())/ M_PI * 180;
   };
  


   //Gradient descent for all points 
   void gradDescent(unsigned int nIterations, TPrecision scaling){
     
     TPrecision msePrev = evaluate();
     std::cout << "MSE start: " << msePrev/Y.N() << std::endl;

     TPrecision orthoPrev = orthogonality();
     std::cout << "Ortho start: " << orthoPrev << std::endl;
     
     //---Storage for syncronous updates 
     DenseMatrix<TPrecision> sync(Z.M(), Z.N());

     //---Do nIterations of gradient descent     
     
     DenseMatrix<TPrecision> Ztmp(Z.M(), Z.N());
     DenseMatrix<TPrecision> Zswap;

     //gradient direction
     DenseVector<TPrecision> gx(Z.M());
     std::cout << "gradDescent all:" << std::endl;
     for(unsigned int i=0; i<nIterations; i++){

      //compute gradient for each point
      TPrecision maxL = 0;
      for(unsigned int j=0; j < Z.N(); j++){
        //compute gradient
        gradX(j, gx);
        
        //store gradient for syncronous updates
        TPrecision l = Linalg<TPrecision>::Length(gx);
        if(l > maxL){
          maxL = l;
        }
        for(unsigned int k=0; k<Z.M(); k++){
          sync(k, j) = gx(k);
        }
      }
      std::cout << std::endl;

      LinalgIO<TPrecision>::writeMatrix("sync.data", sync);
      //sync updates
      TPrecision s = scaling * sX / maxL;
      if(s == std::numeric_limits<TPrecision>::infinity()){
        s = scaling;
      }
      std::cout << "scaling: " << s << std::endl;
      
      //evaluate();
      
      //Linalg<TPrecision>::AddScale(Z, -s, sync, Z);

      
      //Approximate line search with quadratic fit
      DenseMatrix<TPrecision> A(3, 3);
      DenseMatrix<TPrecision> b(3, 1);
      Linalg<TPrecision>::Zero(A);

      b(0, 0) = evaluate();
      Linalg<TPrecision>::AddScale(Z, -5*s, sync, Ztmp);
      Zswap = Z;
      Z = Ztmp;
      Ztmp = Zswap;
      Zchanged = true;
      b(1, 0) = evaluate();
      Linalg<TPrecision>::AddScale(Zswap, -10*s, sync, Z);
      Zchanged = true;
      b(2, 0) = evaluate();
        
      A(0, 2) = 1;
      A(1, 0) = 25;
      A(1, 1) = -5;
      A(1, 2) = 1;
      A(2, 0) = 100;
      A(2, 1) = -10;
      A(2, 2) = 1;

      DenseMatrix<TPrecision> q = Linalg<TPrecision>::Solve(A, b);

      //do step
      if( q(0, 0) > 0){
        //move Z to minimal point
        TPrecision h = -q(1, 0)/(2*q(0, 0));
        if(h<0){
          std::cout << "a1 Step: " << h*s << std::endl;
          Linalg<TPrecision>::AddScale(Ztmp, h*s, sync, Z);
          if(abs(h) < scaling/10){
            break;
          }
        }
        else if(b(0,0) < b(1, 0)){
         //small step
         std::cout << "a2 Step: " << -s << std::endl;
         Linalg<TPrecision>::AddScale(Ztmp, -s, sync, Z);
        }
      }
      else if( b(0,0) > b(1, 0) ){
        //do nothing step to -10*s
        std::cout << "b Step: " << -10*s << std::endl;
      }
      else{
        //small step
        std::cout << "c Step: " << -s << std::endl;
        Linalg<TPrecision>::AddScale(Ztmp, -s, sync, Z);
      }

      A.deallocate();
      b.deallocate();
      q.deallocate();
      

      Zchanged = true;



      std::cout << std::endl;
      std::cout << "Iteration: " << i << std::endl;
      TPrecision mse = evaluate()/Y.N(); 
      std::cout << "MSE: " <<  mse << std::endl;     
      TPrecision ortho = orthogonality();
      std::cout << "Ortho: " << ortho << std::endl;
      if(ortho < orthoPrev || msePrev < mse){
        Zswap = Z;
        Z = Ztmp;
        Ztmp = Zswap;
        break;
      }
      orthoPrev = ortho;
      msePrev = mse;

      std::cout << std::endl << std::endl;
     }


     //cleanup
     sync.deallocate();
     gx.deallocate();
     Ztmp.deallocate();
   
   };
  




   //f(y) - coordinate mapping
   void f( DenseVector<TPrecision> &y, DenseVector<TPrecision> &out ){
     Linalg<TPrecision>::Set(out, 0);

     
     //do kernel regression 
     TPrecision sum = 0;
     for(unsigned int i=0; i<Y.N(); i++){
       wTmp(i) = kernelY.f(y, Y, i);
       sum += wTmp(i);
     }

     for(unsigned int i=0; i < Y.N(); i++){
       Linalg<TPrecision>::AddScale(out, wTmp(i), Z, i, out); 
     }
     Linalg<TPrecision>::Scale(out, 1.f/sum, out);

   };


   //f(y_i) - coordinate mapping
   void f(unsigned int yi, Vector<TPrecision> &out){
     Linalg<TPrecision>::Set(out, 0);
     TPrecision sum = 0;
     for(unsigned int i=0; i < fY.N(); i++){
        /*if(yi ==i){
         wTmp(i) = 0;
        }
        else{*/ 
          wTmp(i) = KY(i, yi);
          sum += wTmp(i);
        //}
     }
     for(unsigned int i=0; i < fY.N(); i++){
        Linalg<TPrecision>::AddScale(out, wTmp(i), Z, i, out);
     }
     Linalg<TPrecision>::Scale(out, 1.f/sum, out);
   };
  

  //g(y_index) - reconstruction mapping
  void g(unsigned int index, Vector<TPrecision> &out){
     Linalg<TPrecision>::Set(out, 0);

     TPrecision sum = 0;

     computefY();

     
     for(unsigned int i=0; i < fY.N(); i++){
        /*if(index ==i){
         wTmp(i) = 0;
        }
        else{*/ 
          wTmp(i) = kernelX.f(fY, i, fY, index);
          sum += wTmp(i);
        //}
     }

     for(unsigned int i=0; i < fY.N(); i++){
       Linalg<TPrecision>::AddScale(out, wTmp(i), Y, i, out); 
     } 
     Linalg<TPrecision>::Scale(out, 1.f/sum, out);
   };


   //g(x) - reconstruction mapping
   void g( Vector<TPrecision> &x, Vector<TPrecision> &out){
     Linalg<TPrecision>::Set(out, 0);

     computefY();

     TPrecision sum = 0;
     for(unsigned int i=0; i < fY.N(); i++){ 
        wTmp(i) = kernelX.f(x, fY, i);
        sum += wTmp(i);
     }

     for(unsigned int i=0; i < fY.N(); i++){
        Linalg<TPrecision>::AddScale(out, wTmp(i), Y, i, out); 
     } 
     Linalg<TPrecision>::Scale(out, 1.f/sum, out);
   };

   

   //Compute gradient of f, i.e. z_r 
   void gradX( int r, DenseVector<TPrecision> &gx){
     
     //update fY if necessary
     computefY();

     //initalize output to zero
     Linalg<TPrecision>::Zero(gx);
     
     unsigned int YM = Y.M();
     unsigned int ZM = Z.M();
     unsigned int N = Y.N();



     //Kernel values
     DenseVector<TPrecision> k(knnY);
     TPrecision sumk = 0;
     DenseVector<TPrecision> sumkg(ZM);
     DenseVector<TPrecision> dfkg[knnY];
     for(unsigned int i=0; i<N; i++){
       dfkg[i] = DenseVector<TPrecision>(ZM);
     }
     DenseVector<TPrecision> sumdfkg(ZM);
     
     //g(f(y_r))
     DenseVector<TPrecision> gfy(YM);
     DenseMatrix<TPrecision> dgfdz(YM, ZM);
     
     //Temp vars.     
     DenseVector<TPrecision> gtmp(ZM);
     DenseVector<TPrecision> diff(YM);

     //Compute gradient
     for(unsigned int knn=0; knn < knnY; knn++){

       int yi = KNNY(knn, r);

       //Precompute kernel values and first and second order derivatives 
       sumk=0;     
       Linalg<TPrecision>::Zero(sumdfkg);

       for(unsigned int i=0; i < knnY; i++){
	 int yj = KNNY(i, yi);
	 //df
	 TPrecision df = KYN( yi, r ) - KYN( yj, r );
         k(i) = kernelX.gradf(fY, yi, fY, yj, dfkg[i]);
       

         sumk += k(i);

         Linalg<TPrecision>::Scale(dfkg[i], df, dfkg[i]);
         Linalg<TPrecision>::Add(sumdfkg, dfkg[i], sumdfkg);
       }
       TPrecision sumk2 = sumk*sumk;
       if(sumk2 == 0){
         continue;
       }


       //g(f(y_i)) 
       Linalg<TPrecision>::Zero(gfy);
       for(unsigned int j=0; j < knnY; j++){
         Linalg<TPrecision>::AddScale(gfy, k(j), Y, j, gfy); 
       }
       Linalg<TPrecision>::Scale(gfy, 1.f/sumk, gfy);

 
       //d g(f(yi)) - yi / d z_r
       Linalg<TPrecision>::Zero(dgfdz);
       for(unsigned int j = 0; j < knnY; j++){
         //if(j == r) continue;
         for(unsigned int n=0; n<ZM; n++){
           TPrecision tmp =  ( dfkg[j](n) * sumk - k(j) * sumdfkg(n) ) / sumk2;
           for(unsigned int m=0; m < YM ; m++){
             dgfdz(m, n) +=  tmp * Y(m, j);
           }
         }
       }
       
       Linalg<TPrecision>::Subtract(gfy, Y, yi, diff);
       Linalg<TPrecision>::Scale(diff, 2, diff);
       Linalg<TPrecision>::Multiply(dgfdz, diff, gtmp, true);
       Linalg<TPrecision>::Add(gx, gtmp, gx);
     }


     	
     //cleanup
     dgfdz.deallocate();
     for(unsigned int i=0; i<knnY; i++){
       dfkg[i].deallocate();
     }
     gtmp.deallocate();
     diff.deallocate();
     k.deallocate();
     sumdfkg.deallocate();
     gfy.deallocate();
   };



   //projection distance gradient
   void gradP(DenseVector<TPrecision> &y, DenseVector<TPrecision> &gp){

     computefY();  
     
     //Jacobian of g(x)
     DenseMatrix<TPrecision> Jg(Y.M(), Z.M());
     //Jacobian of f(y)
     DenseMatrix<TPrecision> Jf(Z.M(), Y.M());
     
     //Kernel values
     DenseVector<TPrecision> kx(Z.N());
     DenseVector<TPrecision> ky(Y.N());
     //Kernel gradients
     DenseMatrix<TPrecision> kxg(Z.M(), Z.N());
     DenseMatrix<TPrecision> kyg(Y.M(), Y.N());
     
     //Precomputed sums of kernel value
     TPrecision sumkx = 0;
     TPrecision sumky = 0;
     DenseVector<TPrecision> sumkxg(Z.M());
     DenseVector<TPrecision> sumkyg(Y.M());

     //Temp vars.
     DenseVector<TPrecision> fy(Z.M());
     DenseVector<TPrecision> gfy(Y.M());
     DenseVector<TPrecision> diff(Y.M());
     DenseVector<TPrecision> xtmp(Z.M());
     DenseVector<TPrecision> ytmp(Y.M());
     
     f(y, fy);
     g(fy, gfy);

     //Jacobian g     
     Linalg<TPrecision>::Zero(sumkxg);
     Linalg<TPrecision>::Zero(Jg); 

     for(unsigned int j=0; j < fY.N(); j++){
       Linalg<TPrecision>::ExtractColumn(fY, j, xtmp);
       kx(j) = kernelX.gradf(xtmp, fy, xtmp);
       sumkx += kx(j);

       for(unsigned int k=0; k < Z.M(); k++){
         kxg(k, j) = xtmp(k);
         sumkxg(k) += kxg(k, j);
       }
     }

     TPrecision sumkx2 = sumkx*sumkx;

     for(unsigned int n = 0; n < fY.N(); n++){
       for(unsigned int j=0; j<Z.M(); j++){
         TPrecision c  = kxg(j, n) * sumkx      -  kx(n) * sumkxg(j)   ;
         for(unsigned int r=0; r< Y.M(); r++){
           Jg(r, j) += c *Y(r, n);
         }
       }
     }  
     Linalg<TPrecision>::Scale(Jg, 1.0/sumkx2, Jg); 
       
     //Jacobian f     
     Linalg<TPrecision>::Zero(sumkyg);
     Linalg<TPrecision>::Zero(Jf); 

     for(unsigned int j=0; j < fY.N(); j++){
       Linalg<TPrecision>::ExtractColumn(Y, j, ytmp);
       ky(j) = kernelY.gradf(ytmp, y, ytmp);
       sumky += ky(j);

       for(unsigned int k=0; k < Z.M(); k++){
         kyg(k, j) = ytmp(k);
         sumkyg(k) += kyg(k, j);
       }
     }

     TPrecision sumky2 = sumky*sumky;

     for(unsigned int n = 0; n < fY.N(); n++){
       for(unsigned int j=0; j<Y.M(); j++){
         TPrecision c  = kyg(j, n) * sumky      -  ky(n) * sumkyg(j)   ;
         for(unsigned int r=0; r< Z.M(); r++){
           Jf(r, j) += c * Z(r, n);
         }
       }
     }  
     Linalg<TPrecision>::Scale(Jf, 1.0/sumky2, Jf);


     //
     for(unsigned int i=0; i<Y.M(); i++){
       ytmp(i) = 0;
       for(unsigned int j=0; j<Z.M(); j++){
       	 ytmp(i) += Jg(i, j) * Jf(j, i);
       }
       ytmp(i) -= 1.f;
     }  
     
     
     for(int i=0; i<gfy.N(); i++){
       gp(i) = (gfy(i) - y(i) ) * ytmp(i);
     } 


     gfy.deallocate();
     kx.deallocate();
     kxg.deallocate();
     ky.deallocate();
     kyg.deallocate();
     sumkxg.deallocate();
     sumkyg.deallocate();
     xtmp.deallocate();
     ytmp.deallocate();
     Jf.deallocate();
     Jg.deallocate();
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
     DenseVector<TPrecision> fy(Z.M());

     //update fY if necessary
     computefY();

     //Precompte kernel values and first and second order derivatives 
     //(gradient and hessian of kernel function)
     
     for(unsigned int i=0; i < fY.N(); i++){
       Linalg<TPrecision>::ExtractColumn(fY, i, fy);

       k(i) = kernelX.gradf(x, fy, gtmp);
       kh[i] = kernelX.hessian(x, fy);
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
      for(unsigned int n = 0; n < fY.N(); n++){
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

   

   //get original Y's
   DenseMatrix<TPrecision> getY(){
     return Y;
   };
   


   //get X (parameters for f
   DenseMatrix<TPrecision> getZ(){
    return Z;
   };


   void changeZ(){
     Zchanged = true;
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
     computefY();
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
       DenseVector<TPrecision> &pk, bool
       useDensity){

     //update fY if necessary
     computefY();

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




private:




  void computefY(){
    if(Zchanged){
      std::cout << "updating f(Y)" << std::endl;
      Zchanged = false;
      DenseVector<TPrecision> tmp(Z.M());
      for(unsigned int i=0; i<Y.N(); i++){
        f(i, tmp);
        Linalg<TPrecision>::SetColumn(fY, i, tmp);
      }
      tmp.deallocate();
    }
  };
 


  void computeKernelX(TPrecision alpha){
    TPrecision sigma = 0;
    DenseMatrix<int> knns(knnSigma, fY.N());
    DenseMatrix<TPrecision> knnDists(knnSigma, fY.N());
    Geometry<TPrecision>::computeKNN(fY, knns, knnDists, l2metric);
    sX = 0;
    for(unsigned int i=0; i < fY.N(); i++){
      sigma += knnDists(knnSigma-1, i);
      sX += knnDists(1, i);

    }
    sigma *= alpha/fY.N();
    
    sX /= fY.N();
    sX3s = sigma*sigma*9;

    std::cout << "sigmaX: " << sigma << std::endl;
    std::cout << "scale: " << sX << std::endl;

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
      sigma += KNNYD(knnSigma-1, i); 
    }
    sigma /= N;
    std::cout << "sigmaY: " << sigma << std::endl;
    kernelY = GaussianKernel<TPrecision>(sigma, fY.M());
    //kernelY = EpanechnikovKernel<TPrecision>(sigma, fY.M());
   


    std::cout << "Compute KY" << std::endl;
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

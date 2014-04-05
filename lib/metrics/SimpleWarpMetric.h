#ifndef SIMPLEWARPMETRIC_H
#define SIMPLEWARPMETRIC_H

#include "ImageMetric.h"
#include "ElasticWarp.h"

#include "itkImageRegionConstIterator.h"

#include <math.h>

#include "ImageIO.h"


//Metric that measures deformation. Deformation computed by the
//SimpleWarpRegistration (dense vectorfield transformation). Measures the
//gradient of the vectorfield.
template<typename TPrecision, typename TImage>
class SimpleWarpMetric : public ImageMetric<TPrecision, TImage>{
  
  
  public:
    typedef TImage Image;
    typedef typename Image::Pointer ImagePointer;

    SimpleWarpMetric(ImagePointer mask, TPrecision step = 0.8, int
        iter = 200, TPrecision alpha = 0.5, TPrecision eps =0.025,
        TPrecision lambda = 10,  TPrecision lambdaInc = 100, TPrecision
        lambdaIncT = 0.0001,  TPrecision sSigma = 1, int nS = 3) 
      : scaleSigma(sSigma), nScales(nS)
    
    {
      
        maskImage = mask; 
        
        warp.setAlpha(alpha);
        warp.setMaximumIterations(iter);
        warp.setMaximumMotion(step);
        warp.setEpsilon(eps);
        warp.setLambda(lambda);
        warp.setLambdaIncrease(lambdaInc);
        warp.setLambdaIncreaseThreshold(lambdaIncT);
        
        symmetric = true;

    };


    TPrecision distance(ImagePointer x1, ImagePointer x2){
     
      VImage *vfield = warp.warpMultiresolution(x1, x2, maskImage, nScales,
          scaleSigma); 

      TPrecision alpha = warp.getAlpha();
      TPrecision tmp = alpha * vfield->sumJacobianFrobeniusSquared() + 
                   (1-alpha) * vfield->sumMagnitudeSquared();
      delete vfield;

      tmp = sqrt(tmp);

      if(symmetric){
        vfield = warp.warpMultiresolution(x2, x1, maskImage, nScales,
          scaleSigma); 

        TPrecision tmp2 = alpha * vfield->sumJacobianFrobeniusSquared() + 
                   (1-alpha) * vfield->sumMagnitudeSquared();

        tmp2 = sqrt(tmp2);

        tmp = (tmp + tmp2) / 2;
        
        delete vfield;
      }

      return tmp;


     };



  private:
    typedef SimpleWarp<TImage> Warp; 
    typedef typename Warp::VImage VImage;
    typedef typename VImage::ITKVectorImage ITKVectorImage;
    typedef typename VImage::VectorType VectorType;
 
    typedef typename ITKVectorImage::Pointer ITKVectorImagePointer;

    typedef typename Warp::ImageTransform ImageTransform; 

    typedef typename itk::ImageRegionConstIterator<Image> ImageIterator;
    typedef typename itk::ImageRegionConstIterator<ITKVectorImage> VectorImageIterator;
    
    ImagePointer maskImage; 
    Warp warp;

    TPrecision scaleSigma;
    int nScales;
    bool symmetric;
   
};


#endif

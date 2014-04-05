#ifndef SIMPLEWARPMETRIC2_H
#define SIMPLEWARPMETRIC2_H

#include "ImageMetric.h"
#include "SimpleWarp.h"

#include "itkImageRegionConstIterator.h"
#include "itkJacobianFrobeniusNormImageFilter.h"

#include <math.h>

#include "ImageIO.h"


//Metric that measures deformation. Deformation computed by the
//SimpleWarpRegistration (dense vectorfield transformation). Measures the
//gradient of the vectorfield.
template<typename TPrecision, typename TImage>
class SimpleWarpMetric2 : public ImageMetric<TPrecision, TImage>{
  
  
  public:
    typedef TImage Image;
    typedef typename Image::Pointer ImagePointer;

    SimpleWarpMetric2(ImagePointer mask, TPrecision maxMotion = 0.8, int
        maxIteration = 200, TPrecision diffusionWeight = 0.05,
        TPrecision averageIntensityTol = 0.05, TPrecision sScale = 8, 
        int nS = 5){
        
        sigmaScale = sScale;
        nScales = nS;
      
        maskImage = mask;
        warp.setDiffusionWeight(diffusionWeight);
        warp.setMaximumIterations(maxIteration);
        warp.setMaximumMotion(maxMotion);
        warp.setDifferenceTolerance(averageIntensityTol);

    };


    TPrecision distance(ImagePointer x1, ImagePointer x2){
     
      VImage *vfield = warp.warpMultiscale(x1, x2, maskImage, sigmaScale, nScales); 

      JacobianNormFilterPointer jacobian = JacobianNormFilter::New();
      jacobian->SetInput(vfield->toITK());
      jacobian->Update();
      ImagePointer jnorm = jacobian->GetOutput();

      ImageIterator jIt( jnorm, jnorm->GetLargestPossibleRegion() );
      ImageIterator mIt( maskImage, maskImage->GetLargestPossibleRegion() );

      TPrecision distance = 0;
      TPrecision tmp = 0;
      for(;!mIt.IsAtEnd(); ++jIt, ++mIt){
        if(mIt.Get()!=0){
          tmp = jIt.Get();
          distance += tmp * tmp;
        }
      } 

      delete vfield;
      return distance;
    };



  private:
    typedef SimpleWarp<Image> Warp; 
    typedef typename Warp::VImage VImage;
    typedef typename VImage::ITKVectorImage ITKVectorImage; 
    typedef typename Warp::ImageTransform ImageTransform; 

    typedef typename itk::ImageRegionConstIterator<Image> ImageIterator;

    typedef typename itk::JacobianFrobeniusNormImageFilter<ITKVectorImage,
            Precision> JacobianNormFilter;
    typedef typename JacobianNormFilter::Pointer JacobianNormFilterPointer;


    ImagePointer maskImage; 
    Warp warp;
    int nScales;
    TPrecision sigmaScale;

   
};


#endif

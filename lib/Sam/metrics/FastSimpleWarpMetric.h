#ifndef FASTSIMPLEWARPMETRIC_H
#define FASTSIMPLEWARPMETRIC_H

#include "ImageMetric.h"
#include "FastSimpleWarp.h"

#include "itkImageRegionConstIterator.h"
#include "itkJacobianFrobeniusNormImageFilter.h"

#include <math.h>

#include "ImageIO.h"

template<typename TPrecision, typename TImage>
class FastSimpleWarpMetric : public ImageMetric<TPrecision, TImage>{
  
  
  public:
    typedef TImage Image;
    typedef typename Image::Pointer ImagePointer;


    FastSimpleWarpMetric(ImagePointer mask, TPrecision alpha = 1, TPrecision tol =
        0.001, TPrecision maxMotion = 0.8, int maxIteration = 100){
      
      maskImage = mask;

      warp.setDiffusionWeight(alpha);
      warp.setMaximumIterations(maxIteration);
      warp.setMaximumMotion(maxMotion);
      warp.setAverageDifferenceTolerance(tol);

    };


    TPrecision distance(ImagePointer x1, ImagePointer x2){
     
      VectorImagePointer vfield = warp.warp(x1, x2, maskImage); 

      JacobianNormFilterPointer jacobian = JacobianNormFilter::New();
      jacobian->SetInput(vfield);
      jacobian->Update();
      ImagePointer jnorm = jacobian->GetOutput();

      ImageIterator jIt( jnorm, jnorm->GetLargestPossibleRegion() );
      ImageIterator mIt( maskImage, maskImage->GetLargestPossibleRegion() );

      TPrecision distance = 0;
      TPrecision tmp = 0;
      TPrecision alpha = warp.getDiffusionWeight();
      for(;!mIt.IsAtEnd(); ++jIt, ++mIt){
        if(mIt.Get()!=0){
          tmp = jIt.Get();
          distance += tmp * tmp;
        }

      } 

      return distance;
    };



  private:
    typedef FastSimpleWarp<Image> Warp; 
    typedef typename Warp::VectorImage VectorImage;
    typedef typename Warp::VectorImagePointer VectorImagePointer; 
    typedef typename Warp::ImageTransform ImageTransform; 

    typedef typename itk::ImageRegionConstIterator<Image> ImageIterator;
    typedef typename itk::ImageRegionConstIterator<VectorImage> VectorImageIterator;

    typedef typename itk::JacobianFrobeniusNormImageFilter<VectorImage,
            Precision>
      JacobianNormFilter;
    typedef typename JacobianNormFilter::Pointer JacobianNormFilterPointer;


    ImagePointer maskImage; 
    Warp warp;

   
};


#endif

#ifndef SOBOLEVMETRIC_H
#define SOBOLEVMETRIC_H

#include "ImageMetric.h"

#include "itkImageRegionConstIterator.h"
#include <itkGradientMagnitudeImageFilter.h>
#include <itkSubtractImageFilter.h>

#include <math.h>

template<typename TPrecision, typename TImage>
class SobolevMetric : public ImageMetric<TPrecision, TImage>{
  
  
  public:
    typedef TImage Image;
    typedef typename Image::Pointer ImagePointer;


    SobolevMetric(ImagePointer mask){
      maskImage = mask;
    };


    TPrecision distance(ImagePointer x1, ImagePointer x2){
     
      SubtractFilterPointer sub = SubtractFilter::New();
      sub->SetInput1(x1);
      sub->SetInput2(x2);
      sub->Update();
      ImagePointer diff = sub->GetOutput();
        

      GradientMagnitudeFilterPointer gm = GradientMagnitudeFilter::New();
      gm->SetInput(diff);
      gm->Update();
      ImagePointer g = gm->GetOutput();

      ImageIterator gIt( g, g->GetLargestPossibleRegion() );
      ImageIterator dIt( diff, diff->GetLargestPossibleRegion() );

      ImageIterator mIt( maskImage, maskImage->GetLargestPossibleRegion() );

      TPrecision distance = 0;
      TPrecision tmp1 = 0;
      TPrecision tmp2;
      for(;!mIt.IsAtEnd(); ++gIt, ++mIt, ++dIt){
        if(mIt.Get()!=0){
          tmp1 = gIt.Get();
          tmp2 = dIt.Get();
          distance += tmp1 * tmp1 + tmp2*tmp2;
        }

      } 

      return sqrt(distance);
    };



  private:

    typedef typename itk::ImageRegionConstIterator<Image> ImageIterator;

    typedef typename itk::GradientMagnitudeImageFilter<Image, Image> GradientMagnitudeFilter;
    typedef typename GradientMagnitudeFilter::Pointer GradientMagnitudeFilterPointer;

    typedef typename itk::SubtractImageFilter<Image, Image> SubtractFilter;
    typedef typename SubtractFilter::Pointer SubtractFilterPointer;

    ImagePointer maskImage; 
   
};


#endif

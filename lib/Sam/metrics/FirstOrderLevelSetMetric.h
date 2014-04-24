#ifndef FIRSTORDERLEVELSETMETRIC_H
#define FIRSTORDERLEVELSETMETRIC_H

#include "ImageMetric.h"
#include "ImageVectorConverter.h"
#include "itkGradientMagnitudeImageFilter.h"
#include "itkImageRegionConstIterator.h"
#include <math.h>





template<typename TPrecision, typename TImage>
class FirstOrderLevelSetMetric : public ImageMetric<TPrecision, TImage>{
  
  
  public:
    typedef TImage Image;
    typedef typename Image::Pointer ImagePointer;


    FirstOrderLevelSetMetric(){
      gm1Filter = GradientMagnitudeFilter::New();
      gm2Filter = GradientMagnitudeFilter::New();
      eps = 200;
    };



    TPrecision distance(ImagePointer x1, ImagePointer x2){
      gm1Filter->SetInput(x1);
      gm1Filter->Update();
      ImagePointer g1 = gm1Filter->GetOutput();
      
      gm2Filter->SetInput(x2);
      gm2Filter->Update();
      ImagePointer g2 = gm2Filter->GetOutput();

      return distance(x1, g1, x2, g2);
    };


    
    TPrecision distance(ImagePointer x1, ImagePointer g1, ImagePointer x2, ImagePointer g2){

      TPrecision dist = 0;

      ImageRegionConstIterator x1It(x1, x1->GetLargestPossibleRegion());
      ImageRegionConstIterator g1It(g1, g1->GetLargestPossibleRegion());
      ImageRegionConstIterator x2It(x2, x2->GetLargestPossibleRegion());
      ImageRegionConstIterator g2It(g2, g2->GetLargestPossibleRegion());


      for(; !x1It.IsAtEnd(); ++x1It, ++g1It, ++x2It, ++g2It){

        TPrecision tmp = ( x1It.Get() - x2It.Get() ) / 
                         ( eps + (g1It.Get() + g2It.Get()) / 2.f );
        dist += tmp*tmp;
      }
      
      return sqrt(dist);
    };


    virtual ImagePointer computeDistanceImage(ImagePointer x1, ImagePointer x2){
      return NULL;
    };

    void derivativeX2(ImagePointer x1, ImagePointer g1, ImagePointer x2, ImagePointer g2, ImagePointer dx2){
      ImageRegionConstIterator x1It(x1, x1->GetLargestPossibleRegion());
      ImageRegionConstIterator g1It(g1, g1->GetLargestPossibleRegion());
      ImageRegionConstIterator x2It(x2, x2->GetLargestPossibleRegion());
      ImageRegionConstIterator g2It(g2, g2->GetLargestPossibleRegion());
      ImageRegionIterator dx2It(dx2, dx2->GetLargestPossibleRegion());


      for(; !x1It.IsAtEnd(); ++x1It, ++g1It, ++x2It, ++g2It, ++dx2It){
 
        TPrecision denom = ( eps + (g1It.Get() + g2It.Get()) / 2.f );
        TPrecision tmp = ( x2It.Get() - x1It.Get() ) / (denom*denom);
	      dx2It.Set(tmp);
      }

    };




    void setEpsilon(TPrecision e){
      eps =e;
    };

  private:
    TPrecision eps;

    typedef typename itk::GradientMagnitudeImageFilter<Image, Image>
      GradientMagnitudeFilter;
    typedef typename GradientMagnitudeFilter::Pointer
      GradientMagnitudeFilterPointer;

    typedef typename itk::ImageRegionConstIterator<Image> ImageRegionConstIterator;
    typedef typename itk::ImageRegionIterator<Image> ImageRegionIterator;

    GradientMagnitudeFilterPointer gm1Filter;
    GradientMagnitudeFilterPointer gm2Filter;
};


#endif

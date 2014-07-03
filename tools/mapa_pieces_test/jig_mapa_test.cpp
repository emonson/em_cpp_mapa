#include <iostream>
#include <string>
#include <algorithm>
#include <time.h>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/SparseCore>
#include <igl/sum.h>

#include "CmdLine.h"

#include "mapa_config.h"
#include "UtilityCalcs.h"
#include "TDMgenerator.h"
#include "JIGtokenizer.h"
#include "SvdlibcSVD.h"
#include "Options.h"
#include "Mapa.h"
#include "XMLclusterdoc.h"

int main( int argc, char** argv )
{

    // ---------------------------------------------
    //Command line parsing
    TCLAP::CmdLine cmd("jig_mapa_test", ' ', "0.1");

    TCLAP::UnlabeledValueArg<std::string> filenameArg("jigfile", "Path to Jigsaw .jig data file", true, "", ".jig data file");
    cmd.add(filenameArg);

    TCLAP::ValueArg<int> lengthArg("l","min_term_length", "Minimum number of characters for a term", false, 3, "integer > 1");
    cmd.add(lengthArg);

    TCLAP::ValueArg<int> countArg("c","min_term_count", "Minimum term count per document", false, 2, "integer > 0");
    cmd.add(countArg);

    try
    {
        cmd.parse( argc, argv );
    }
    catch (TCLAP::ArgException &e)
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
        return -1;
    }

	// std::string data_dir = MAPA::UtilityCalcs::PathAppend(MAPA_SOURCE_DIR, "data");
    // std::string filename = MAPA::UtilityCalcs::PathAppend(data_dir, "InfovisVAST-papers.jig");
    std::string filename = filenameArg.getValue();
    
    // NOTE: After checking for unreasonable values, the adjusted values are set to non-defaults...
    int min_term_length = lengthArg.getValue();
    min_term_length = min_term_length > 1 ? min_term_length : 2;
    
    int min_term_count = countArg.getValue();
    min_term_count = min_term_count > 0 ? min_term_count : 1;

    // ---------------------------------------------
    // Load, tokenize, and generate TDM for document data

    MAPA::TDMgenerator tdm_gen(min_term_length, min_term_count);
	MAPA::JIGtokenizer jig_tok(filename, &tdm_gen);
    
    Eigen::SparseMatrix<double,0,long> tdm = tdm_gen.getTDM();
    Eigen::VectorXd tdm_mean = (MatrixXd(tdm)).rowwise().mean();
    
    std::cout << "TDM: " << tdm.rows() << " x " << tdm.cols() << ", " << tdm.nonZeros() << std::endl << std::endl;
    
    // ---------------------------------------------
    // Reduce dimensionality with SVD

    int rank = 50;
    clock_t t = clock();
    
    MAPA::SvdlibcSVD svds(tdm, rank);
    
    t = clock() - t;
    printf("SVD Elapsed time: %.10f sec.\n", (double)t/CLOCKS_PER_SEC );
    
    std::cout << "U: " << svds.matrixU().rows() << " x " << svds.matrixU().cols() << std::endl;
    // std::cout << svds.matrixU() << std::endl << std::endl;
    std::cout << "V: " << svds.matrixV().rows() << " x " << svds.matrixV().cols() << std::endl;
    // std::cout << svds.matrixV() << std::endl << std::endl;
    std::cout << "S" << std::endl;
    std::cout << svds.singularValues().transpose() << std::endl << std::endl;

    Eigen::ArrayXXd Xred = svds.matrixV() * svds.singularValues().asDiagonal();
    
    std::cout << "Xred: " << Xred.rows() << " x " << Xred.cols() << std::endl;

    // ---------------------------------------------
    // Run MAPA on reduced dimensionality data
    
    // Reseed random number generator since Eigen Random.h doesn't do this itself
    srand( (unsigned int)time(NULL) );

    // opts = struct('dmax',3, 'Kmax',15, 'n0',640, 'plotFigs',true);
    MAPA::Opts opts;
    opts.dmax = 6;
    opts.d_hardlimit = 10;
    opts.Kmax = 12;
    // opts.n0 = Xred.rows();
    opts.n0 = 160;
    
    opts.SetDefaults(Xred);
    // std::cout << "options" << std::endl;
    // std::cout << opts << std::endl;
        
    t = clock();
    
    MAPA::Mapa mapa(Xred, opts);
    
    t = clock() - t;
    
    std::cout << std::endl << "Mapa labels:" << std::endl;
    std::cout << mapa.GetLabels().transpose() << std::endl;
    std::cout << std::endl << "Mapa plane dims:" << std::endl;
    std::cout << mapa.GetPlaneDims().transpose() << std::endl;
    std::cout << std::endl << "Mapa disance-based error:" << std::endl;
    std::cout << mapa.GetDistanceError() << std::endl << std::endl;

    printf("Elapsed time: %.10f sec.\n", (double)t/CLOCKS_PER_SEC );

//     double MisclassificationRate = MAPA::UtilityCalcs::ClusteringError(mapa.GetLabels(), true_labels);
//     
//     t = clock() - t;
//     printf("Elapsed time: %.10f sec.for %ld d result\n", (double)t/CLOCKS_PER_SEC, mapa.GetPlaneDims().size() );
//     printf("Misclassification Rate: %.10f\n", MisclassificationRate );

    // Generate output
    MAPA::XMLclusterdoc(&tdm_gen, &mapa, &svds, "jig mapa test");
    
    return EXIT_SUCCESS;
}

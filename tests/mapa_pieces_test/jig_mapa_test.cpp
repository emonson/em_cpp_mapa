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
    
    // required
    TCLAP::UnlabeledValueArg<std::string> filenameArg("jigfile", "Path to Jigsaw .jig data file", false, "", ".jig data file");
    
    // one or the other required (xor below)
    TCLAP::ValueArg<int> kArg("K","K", "Number of clusters", true, 2, "integer > 0");
    TCLAP::ValueArg<int> kmaxArg("","kmax", "Maximum number of clusters", true, 2, "integer >= 0");

    // optional
    TCLAP::ValueArg<std::string> outfileArg("o","outfile", "Path (name) of output .xml file", false, "", "integer > 0");

    TCLAP::ValueArg<int> lengthArg("l","min_term_length", "Minimum number of characters for a term", false, 3, "integer > 1");

    TCLAP::ValueArg<int> countArg("c","min_term_count", "Minimum term count per document", false, 2, "integer > 0");

    TCLAP::ValueArg<int> nlabelsArg("n","n_cluster_labels", "Number of cluster keyword labels in XML", false, 3, "integer > 0");
    TCLAP::ValueArg<int> nstoplabelsArg("s","n_stop_labels", "Number of clusters center labels to ignore when generating cluster labels", false, 10, "integer >= 0");

    TCLAP::ValueArg<int> n0Arg("","n0", "Number of seed points", false, 0, "integer > 0");

    TCLAP::ValueArg<int> rankArg("D","dim_reduced", "Dimensionality to reduce original data to with SVD", false, 50, "integer > 0");

    TCLAP::ValueArg<int> dmaxArg("","dmax", "Suggested limit on dimensionality of each cluster", false, 6, "integer >= 0");
    TCLAP::ValueArg<int> dhardlimitArg("","dhard", "Forced hard limit on max dimensionality of each cluster", false, 10, "integer >= 0");

    TCLAP::SwitchArg verboseArg("V", "verbose", "Verbose output", false);

    try
    {
        cmd.add(filenameArg);
        cmd.add(outfileArg);
        cmd.add(lengthArg);
        cmd.add(countArg);
        cmd.add(nlabelsArg);
        cmd.add(nstoplabelsArg);
        // one or the other must be set
        cmd.xorAdd(kArg, kmaxArg);
        cmd.add(n0Arg);
        cmd.add(dmaxArg);
        cmd.add(dhardlimitArg);
        cmd.add(verboseArg);
        cmd.add(rankArg);
        
        cmd.parse( argc, argv );
    }
    catch (TCLAP::ArgException &e)
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
        return -1;
    }

    std::string filename = filenameArg.getValue();
    if (filename.size() == 0)
    {
	    std::string data_dir = MAPA::UtilityCalcs::PathAppend(MAPA_SOURCE_DIR, "data");
        filename = MAPA::UtilityCalcs::PathAppend(data_dir, "InfovisVAST-papers.jig");
    }
    
    std::string outfile = outfileArg.getValue();
    if (!outfileArg.isSet())
    {
        unsigned pos = filename.find_last_of('.');
        outfile = filename.substr(0,pos);
    }
    
    // NOTE: After checking for unreasonable values, the adjusted values are set to non-defaults...
    int min_term_length = lengthArg.getValue();
    min_term_length = min_term_length > 1 ? min_term_length : 2;
    
    int min_term_count = countArg.getValue();
    min_term_count = min_term_count > 0 ? min_term_count : 1;

    int n_top_terms = nlabelsArg.getValue();
    n_top_terms = n_top_terms > 0 ? n_top_terms : 1;
    int n_stop_terms = nstoplabelsArg.getValue();
    n_stop_terms = n_stop_terms >= 0 ? n_stop_terms : 0;
    

    // Reseed random number generator since Eigen Random.h doesn't do this itself
    srand( (unsigned int)time(NULL) );

    // MAPA options
    // opts = struct('dmax',3, 'Kmax',15, 'n0',640, 'plotFigs',true);
    MAPA::Opts opts;
    
    opts.dmax = dmaxArg.getValue();
    opts.dmax = opts.dmax > 0 ? opts.dmax : 1;
    
    // NOTE: defaulting to hard limit set
    opts.d_hardlimit = dhardlimitArg.getValue();
    // TODO: should probably check against dmax and make way to not set...
    
    int Kref;
    if (kArg.isSet())
    {
        opts.K = kArg.getValue();
        Kref = opts.K;
    }
    else if (kmaxArg.isSet())
    {
        opts.Kmax = kmaxArg.getValue();
        Kref = opts.Kmax;
    }
    else
    {
        // Shouldn't ever reach here!
        throw("Very bad things...");
    }
    
    bool verbose = verboseArg.getValue();
    
    // ---------------------------------------------
    // Load, tokenize, and generate TDM for document data

    MAPA::TDMgenerator tdm_gen(min_term_length, min_term_count);
	MAPA::JIGtokenizer jig_tok(filename, &tdm_gen);
    
    Eigen::SparseMatrix<double,0,long> tdm = tdm_gen.getTDM();
    Eigen::VectorXd tdm_mean = (MatrixXd(tdm)).rowwise().mean();
    
    if (verbose) {
        std::cout << "TDM: " << tdm.rows() << " x " << tdm.cols() << ", " << tdm.nonZeros() << std::endl << std::endl;
    }
    
    // ---------------------------------------------
    // Reduce dimensionality with SVD
    
    // opts.n0 = Xreduced.rows();
    if (n0Arg.isSet())
    {
        opts.n0 = n0Arg.getValue();
        if (opts.n0 < 5.0 * Kref)
        {
            std::cout << "WARNING: Should probably set n0 at least 10 * number of clusters" << std::endl;
            std::cout << "Currently set at n0 = " << opts.n0 << std::endl;
        }
    }
    else
    {
        // TODO: Maybe should have an easy way to set to use all points as seeds...?
        opts.n0 = 20*Kref < tdm.cols() ? 20*Kref : tdm.cols();
    }
    
    // Set reduced dimensionality late since a reasonable number is based on dmax * K
    int rank = rankArg.getValue();
    rank = rank > 0 ? rank : 1;
    if (!rankArg.isSet())
    {
        rank = Kref * opts.dmax;
        std::cout << "Note: data dimensionality is by default being reduced to " << rank << " == K * dmax" << std::endl;
    }
    if (rankArg.isSet() && (rank < Kref * opts.dmax))
    {
        std::cout << "WARNING: you should probably set your reduced dimensionality above K * dmax" << std::endl;
        std::cout << "Currently set at rank = " << rank << std::endl;
    }
    rank = rank <= tdm.rows() ? rank : tdm.rows();

    MAPA::SvdlibcSVD svds(tdm, rank);
    
    clock_t t = clock();

    t = clock() - t;
    if (verbose)
    {
        printf("SVD Elapsed time: %.10f sec.\n", (double)t/CLOCKS_PER_SEC );
    
        std::cout << "U: " << svds.matrixU().rows() << " x " << svds.matrixU().cols() << std::endl;
        // std::cout << svds.matrixU() << std::endl << std::endl;
        std::cout << "V: " << svds.matrixV().rows() << " x " << svds.matrixV().cols() << std::endl;
        // std::cout << svds.matrixV() << std::endl << std::endl;
        std::cout << "S" << std::endl;
        std::cout << svds.singularValues().transpose() << std::endl << std::endl;
    }

    Eigen::ArrayXXd Xreduced = svds.matrixV() * svds.singularValues().asDiagonal();
    
    if (verbose) {
        std::cout << "Xreduced: " << Xreduced.rows() << " x " << Xreduced.cols() << std::endl;
    }

    // ---------------------------------------------
    // Run MAPA on reduced dimensionality data
    
    
    opts.SetDefaults(Xreduced);
    if (verbose)
    {
        std::cout << "options" << std::endl;
        std::cout << opts << std::endl << std::endl;
    }
        
    t = clock();
    
    // RUN MAPA
    MAPA::Mapa mapa(Xreduced, opts);
    
    t = clock() - t;
    
    if (verbose) {
        std::cout << std::endl << "Mapa labels:" << std::endl;
        std::cout << mapa.GetLabels().transpose() << std::endl;
        std::cout << std::endl << "Mapa plane dims:" << std::endl;
        std::cout << mapa.GetPlaneDims().transpose() << std::endl;
        std::cout << std::endl << "Mapa disance-based error:" << std::endl;
        std::cout << mapa.GetDistanceError() << std::endl << std::endl;

        printf("Elapsed time: %.10f sec.\n", (double)t/CLOCKS_PER_SEC );
    }

//     double MisclassificationRate = MAPA::UtilityCalcs::ClusteringError(mapa.GetLabels(), true_labels);
//     
//     t = clock() - t;
//     printf("Elapsed time: %.10f sec.for %ld d result\n", (double)t/CLOCKS_PER_SEC, mapa.GetPlaneDims().size() );
//     printf("Misclassification Rate: %.10f\n", MisclassificationRate );

    // Generate output
    MAPA::XMLclusterdoc(&tdm_gen, &mapa, &svds, outfile, n_top_terms, n_stop_terms);
    
    return EXIT_SUCCESS;
}

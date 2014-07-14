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
#include "DIRtokenizer.h"
#include "JIGtokenizer.h"
#include "SvdlibcSVD.h"
#include "Options.h"
#include "Mapa.h"
#include "XMLclusterdoc.h"

#include "dirent.h"     /* directory navigation */

int main( int argc, char** argv )
{

    // ---------------------------------------------
    // Command line parsing
    
    TCLAP::CmdLine cmd("mapa_clustering", ' ', "0.9");
    
    // required
    TCLAP::UnlabeledValueArg<std::string> pathnameArg("path", "Path to .jig file or directory of text files", true, "", "path string");
    
    // one or the other required (xor below)
    TCLAP::ValueArg<int> kArg("K","K", "Number of clusters", true, 2, "integer");

    TCLAP::ValueArg<int> kmaxArg("","kmax", "Maximum number of clusters", true, 2, "integer");

    // optional
    TCLAP::ValueArg<std::string> outfileArg("o","outfile", "Path (name) of output .xml file (default <data_file_name>.xml)", false, "", "path string");

    TCLAP::ValueArg<std::string> addstopwordsArg("","add_stopwords", "Space-delimited string in quotes of extra stopwords to add", false, "", "string");

    TCLAP::ValueArg<int> lengthArg("","min_term_length", "Minimum number of characters for a term (default = 3)", false, 3, "integer");

    TCLAP::ValueArg<int> countArg("","min_term_count", "Minimum term count per document (default = 2)", false, 2, "integer");

    TCLAP::ValueArg<int> nlabelsArg("","n_cluster_labels", "Number of cluster keyword labels in XML (default = 3)", false, 3, "integer");
    
    TCLAP::ValueArg<int> nstoplabelsArg("","n_stop_labels", "Number of clusters center labels to ignore when generating cluster labels (default = 10)", false, 10, "integer");

    TCLAP::ValueArg<int> n0Arg("","n0", "Number of seed points (default = 20 * K)", false, 0, "integer");

    TCLAP::ValueArg<int> rankArg("D","dim_reduced", "Dimensionality to reduce original data to with SVD (default = K * dmax)", false, 50, "integer");

    TCLAP::ValueArg<int> dmaxArg("","dmax", "Suggested limit on dimensionality of each cluster (default = 6)", false, 6, "integer");
    
    TCLAP::ValueArg<int> dhardlimitArg("","dhard", "Forced hard limit on max dimensionality of each cluster (default = 10)", false, 10, "integer");

    TCLAP::SwitchArg verboseArg("V", "verbose", "Verbose output", false);

    try
    {
        cmd.add(verboseArg);
        cmd.add(nstoplabelsArg);
        cmd.add(nlabelsArg);
        cmd.add(addstopwordsArg);
        cmd.add(countArg);
        cmd.add(lengthArg);
        cmd.add(dhardlimitArg);
        cmd.add(dmaxArg);        
        cmd.add(rankArg);
        cmd.add(n0Arg);
        cmd.xorAdd(kArg, kmaxArg);
        cmd.add(outfileArg);
        cmd.add(pathnameArg);

        cmd.parse( argc, argv );
    }
    catch (TCLAP::ArgException &e)
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
        return -1;
    }

    // ---------------------------------------------
    // Grab command line values
    
    std::string pathname = pathnameArg.getValue();
    
    std::string outfile = outfileArg.getValue();
    if (!outfileArg.isSet())
    {
        unsigned pos = pathname.find_last_of('.');
        outfile = pathname.substr(0,pos);
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
    if (addstopwordsArg.isSet())
    {
        tdm_gen.addStopwords(addstopwordsArg.getValue());
    }
    
    // Check whether directory or file
    DIR *dir;
    struct dirent *ent;
        
    /* Try to open directory stream */
    dir = opendir(pathname.c_str());
    if (dir == NULL)
    {
        MAPA::JIGtokenizer jig_tok(pathname, &tdm_gen);
    }
    else
    {
        MAPA::DIRtokenizer dir_tok(pathname, &tdm_gen);
	}
    
    Eigen::SparseMatrix<double,0,long> tdm = tdm_gen.getTDM();
    Eigen::VectorXd tdm_mean = (MatrixXd(tdm)).rowwise().mean();
    
    if (verbose) {
        std::cout << "TDM: " << tdm.rows() << " x " << tdm.cols() << ", " << tdm.nonZeros() << std::endl << std::endl;
    }
    
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

    // ---------------------------------------------
    // Reduce dimensionality with SVD
    
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
    
    if (verbose) 
    {
        std::cout << "Xreduced: " << Xreduced.rows() << " x " << Xreduced.cols() << std::endl;
    }

    // ---------------------------------------------
    // Finalize options
    
    opts.SetDefaults(Xreduced);
    if (verbose)
    {
        std::cout << "options" << std::endl;
        std::cout << opts << std::endl << std::endl;
    }
        
    t = clock();
    
    // ---------------------------------------------
    // RUN MAPA on reduced dimensionality data
    
    MAPA::Mapa mapa(Xreduced, opts);
    
    t = clock() - t;
    
    if (verbose) 
    {
        std::cout << std::endl << "Mapa labels:" << std::endl;
        std::cout << mapa.GetLabels().transpose() << std::endl;
        std::cout << std::endl << "Mapa plane dims:" << std::endl;
        std::cout << mapa.GetPlaneDims().transpose() << std::endl;
        std::cout << std::endl << "Mapa disance-based error:" << std::endl;
        std::cout << mapa.GetDistanceError() << std::endl << std::endl;

        printf("Elapsed time: %.10f sec.\n", (double)t/CLOCKS_PER_SEC );
    }

    // ---------------------------------------------
    // Generate output
    
    MAPA::XMLclusterdoc(&tdm_gen, &mapa, &svds, outfile, n_top_terms, n_stop_terms);
    
    return EXIT_SUCCESS;
}

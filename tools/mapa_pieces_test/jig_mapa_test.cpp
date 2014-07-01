#include <iostream>
#include <string>
#include <time.h>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/SparseCore>
#include <igl/sum.h>

#include "mapa_config.h"
#include "UtilityCalcs.h"
#include "TDMgenerator.h"
#include "JIGtokenizer.h"
#include "SvdlibcSVD.h"
#include "Options.h"
#include "Mapa.h"

int main( int argc, const char** argv )
{
    // ---------------------------------------------
    // Load, tokenize, and generate TDM for document data

	std::string data_dir = MAPA::UtilityCalcs::PathAppend(MAPA_SOURCE_DIR, "data");
    std::string filename = MAPA::UtilityCalcs::PathAppend(data_dir, "InfovisVAST-papers.jig");
        
    int min_term_length = 2;
    int min_term_count = 3;
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

    // ---------------------------------------------
    // Convert MAPA output to document labels and terms
    
    ArrayXi labels = mapa.GetLabels();
    std::vector<ArrayXd> centers = mapa.GetPlaneCenters();
    std::vector<ArrayXXd> bases = mapa.GetPlaneBases();

    std::vector<std::string> docIDs = tdm_gen.getDocIDs();
    std::vector<std::string> terms = tdm_gen.getTerms();
    
    // TODO: may need to figure out doc closest to center since center not a doc...
    
    int n_top_terms = 10;
    
    // ---------------------------------------------
    // TDM mean vector terms
    
    ArrayXd cent = tdm_mean.array().abs();
    
    // sort columns independently (only one here)
    int dim = 1;
    // sort descending order
    int ascending = false;
    // Sorted output matrix
    ArrayXd Y;
    // sorted indices for sort dimension
    ArrayXi IX;
    
    igl::sort(cent,1,ascending,Y,IX);
    
    std::cout << "tdm mean" << std::endl;
    for (int ii = 0; ii < n_top_terms; ii++)
    {
        std::cout << terms.at(IX(ii)) << " ";
    }
    std::cout << std::endl << std::endl;

    // ---------------------------------------------
    // Centers terms
    std::vector< std::vector<std::string> > centers_top_terms;
    int center_count = 0;
    for(std::vector<ArrayXd>::iterator it = centers.begin(); it != centers.end(); ++it) {
        if ((*it).size() > 0)
        {
            std::vector<std::string> top_terms;
            // Reproject center back into term space
            // Centers come out as column vectors, so U [D x N] * center [N x 1] = cent [D x 1]
            ArrayXd cent = (svds.matrixU() * (*it).matrix()).array().abs();

			// sort columns independently (only one here)
			int dim = 1;
			// sort descending order
			int ascending = false;
			// Sorted output matrix
			ArrayXd Y;
			// sorted indices for sort dimension
			ArrayXi IX;
			
			igl::sort(cent,1,ascending,Y,IX);
			
			for (int ii = 0; ii < n_top_terms; ii++)
			{
			    top_terms.push_back(terms.at(IX(ii)));
			}
            centers_top_terms.push_back(top_terms);
        }
        center_count++;
    }
    
    // ---------------------------------------------
    // Centers Diff terms
    std::vector< std::vector<std::string> > centers_offset_top_terms;
    center_count = 0;
    for(std::vector<ArrayXd>::iterator it = centers.begin(); it != centers.end(); ++it) {
        if ((*it).size() > 0)
        {
            std::vector<std::string> top_terms;
            // Reproject center back into term space
            // Centers come out as column vectors, so U [D x N] * center [N x 1] = cent [D x 1]
            ArrayXd cent = ((svds.matrixU() * (*it).matrix()) - tdm_mean).array().abs();
			
			// sort columns independently (only one here)
			int dim = 1;
			// sort descending order
			int ascending = false;
			// Sorted output matrix
			ArrayXd Y;
			// sorted indices for sort dimension
			ArrayXi IX;
			
			igl::sort(cent,1,ascending,Y,IX);
			
			for (int ii = 0; ii < n_top_terms; ii++)
			{
			    top_terms.push_back(terms.at(IX(ii)));
			}
            centers_offset_top_terms.push_back(top_terms);
        }
        center_count++;
    }
    
    // ---------------------------------------------
    // Gathering up doc IDs for clusters
    // NOTE: Not relying on labels being sequential, even though they should be...
    std::vector< std::vector<std::string> > cluster_docIDs;
    ArrayXi unique_labels = MAPA::UtilityCalcs::UniqueMembers(labels);
    std::map<int, int> label_clusterIdx_map;
    for (int ll = 0; ll < unique_labels.size(); ll++)
    {
        std::vector<std::string> doc_vec;
        cluster_docIDs.push_back(doc_vec);
        label_clusterIdx_map[unique_labels(ll)] = ll;
        std::cout << "(" << unique_labels(ll) << "," << ll << ") ";
    }
    std::cout << std::endl;
    
    // NOTE: labels and docIDs had better be the same length...
    for (int ii = 0; ii < labels.size(); ii++)
    {
        cluster_docIDs.at( label_clusterIdx_map[labels(ii)] ).push_back(docIDs.at(ii));
    }
     
    // ---------------------------------------------
    // Basis vectors terms
    std::cout << "Centers & Basis Vectors" << std::endl;
    center_count = 0;
    for(std::vector<ArrayXXd>::iterator it = bases.begin(); it != bases.end(); ++it) {
        
        // docIDs
        for (int ii = 0; ii < cluster_docIDs.at(center_count).size(); ii++)
        {
            std::cout << cluster_docIDs.at(center_count).at(ii) << " ";
        }
        std::cout << std::endl;
        
        // Centers
        std::cout << center_count << " ";
        for (int ii = 0; ii < n_top_terms; ii++)
        {
            std::cout << centers_top_terms.at(center_count).at(ii) << " ";
        }
        std::cout << std::endl;
        
        // Centers with mean subtracted
        std::cout << center_count << " ";
        for (int ii = 0; ii < n_top_terms; ii++)
        {
            std::cout << centers_offset_top_terms.at(center_count).at(ii) << " ";
        }
        std::cout << std::endl;
        
        if ((*it).size() > 0)
        {
            int n_basis_vecs = (*it).rows();
            for (int bb = 0; bb < n_basis_vecs; bb++)
            {
                // Reproject center back into term space
                // Bases come out as row vectors, so U [D x N] * center [N x 1] = cent [D x 1]
                ArrayXd projected = ((*it).row(bb).matrix() * svds.matrixU().transpose()).array().abs().transpose();
            
                // sort columns independently (only one here)
                int dim = 1;
                // sort descending order
                int ascending = false;
                // Sorted output matrix
                ArrayXd Y;
                // sorted indices for sort dimension
                ArrayXi IX;
            
                igl::sort(projected,1,ascending,Y,IX);
            
                std::cout << "  " << bb << " : ";
                for (int ii = 0; ii < n_top_terms; ii++)
                {
                    std::cout << terms.at(IX(ii)) << " ";
                }
                std::cout << std::endl;
			}
        }
        center_count++;
        std::cout << std::endl;
    }
    
    return EXIT_SUCCESS;
}

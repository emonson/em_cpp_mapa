#ifndef XMLCLUSTERDOC_H
#define XMLCLUSTERDOC_H

/* XMLclusterdoc.h

Generate Jigsaw cluster description XML output document

Eric E Monson – 2014
Duke University

*/

#include <iostream>
#include <string>
#include <map>
#include <vector>

#include <Eigen/Core>

#include "tinyxml2.h"
#include "TDMgenerator.h"
#include "Mapa.h"

using namespace tinyxml2;
using namespace Eigen;

typedef std::vector< std::vector<std::string> > VEC_OF_STR_VECS;

namespace MAPA {

class XMLclusterdoc {

public:

    XMLclusterdoc(MAPA::TDMgenerator *tdm_gen, MAPA::Mapa *mapa, MAPA::SvdlibcSVD *svds, std::string name, int n_top_terms = 3, int n_stop_terms = 10)
    {
        // -------------------------------------
        // Grab MAPA output
        labels = mapa->GetLabels();
        centers = mapa->GetPlaneCenters();
        bases = mapa->GetPlaneBases();

        docIDs = tdm_gen->getDocIDs();
        terms = tdm_gen->getTerms();
        
        U = svds->matrixU();
    
        // TODO: may need to figure out doc closest to center since center not a doc...
    
        // Calculate the mean of the cluster centers
        generate_tmd_mean();
        
        // Gathering up doc IDs for clusters
        generate_cluster_docIDs();
        
        // -------------------------------------
        // DECIDE here what type of term generation to use...
        
        std::vector<std::string> tdm_mean_terms = generate_tdm_mean_terms();

        // Centers terms
        // VEC_OF_STR_VECS centers_top_terms = generate_center_terms();
    
        // Centers Diff terms
        // VEC_OF_STR_VECS centers_offset_top_terms = generate_center_offset_terms();

        // Basis vectors terms
        // VEC_OF_STR_VECS bases_top_terms = generate_bases_terms(n_top_terms);
        
        // Bases terms that don't repeat tdm_mean top 10 terms
        std::vector<std::string> stop_terms;
        n_stop_terms = n_stop_terms < tdm_mean_terms.size() ? n_stop_terms : tdm_mean_terms.size();
        for (int ii = 0; ii < n_stop_terms; ii++)
        {
            stop_terms.push_back( tdm_mean_terms.at(ii) );
        }
        // VEC_OF_STR_VECS label_terms = generate_bases_nomean_terms(stop_terms, n_top_terms);
        VEC_OF_STR_VECS label_terms = generate_centers_nomean_terms(stop_terms, n_top_terms);
        
        // -------------------------------------
        // Generate XML output
        generate_XML_output(label_terms, name, &stop_terms);

    };
    
    void generate_XML_output(VEC_OF_STR_VECS clusters_terms, std::string clusters_name, std::vector<std::string> *common_terms = 0)
    {
        if (clusters_terms.size() != cluster_docIDs.size())
        {
            std::cerr << "ERROR: MAPA::XMLclusterdoc -- number of clusters doesn't match between terms and docIDs!" << std::endl;
            return;
        }
        
        // Make file name out of clusters_name by replacing spaces with underscores
        std::string out_file = clusters_name;
        std::replace( out_file.begin(), out_file.end(), ' ', '_');
        out_file += ".xml";
        
        // If this is a whole path, try to grab just the end file name for the clusters name
        std::string out_name = clusters_name;
        unsigned found = out_name.find_last_of("/\\");
        out_name = out_name.substr(found+1);

        
        // XML building test
        // TODO: change and make more flexible/modular the term geration based on centers & bases!!

        // <?xml version="1.0" encoding="UTF-8"?>
        // <jigsawcluster>
        //   <name>Papers Text Cluster</name>
        //   <type>text-based</type>
        //   <seeddocuments>infovis95--528697,infovis04--1382890,infovis04--1382910,infovis05--1532149,infovis05--1532142,vast09--5333895,infovis09--5290703,infovis04--1382884,infovis03--1249027,vast13--146,vast07--4389009,infovis02--1173149,infovis09--5290708,vast06--4035743,infovis10--216,infovis04--1382899,vast07--4388994,infovis12--250,infovis11--196,infovis12--288</seeddocuments>
        //   <cluster>
        //     <label>network,usefulness,structure</label>
        //     <representativedocument>vast06--4035756</representativedocument>
        //     <documents>infovis95--528697,infovis95--528692,infovis95--528690,infovis95--528685,infovis96--559226,infovis96--559220,infovis96--559215,infovis97--636784,infovis97--636778,infovis99--801861,infovis00--885101,infovis00--885104,infovis00--885105,infovis01--963273,infovis01--963291,infovis02--1173155,infovis02--1173160,infovis02--1173163,infovis03--1284026,infovis03--1249028,infovis03--1249010,infovis03--1249030,infovis03--1249011,infovis04--1382901,infovis04--1382889,infovis05--1532125,infovis05--1532126,infovis05--1532137,infovis06--4015419,infovis06--4015423,infovis06--4015424,infovis07--4376129,infovis07--4376149,infovis07--4376154,infovis08--4658123,infovis08--4658138,infovis08--4658145,infovis09--5290724,infovis09--5290709,infovis09--5290697,infovis10--205,infovis11--213,infovis11--183,infovis11--174,infovis12--208,infovis12--255,infovis12--265,infovis12--286,infovis13--153,infovis13--179,infovis13--227,infovis13--230,infovis13--232,vast06--4035746,vast06--4035752,vast06--4035753,vast06--4035754,vast06--4035755,vast06--4035756,vast07--4389007,vast07--4389012,vast08--4677355,vast08--4677357,vast09--5332610,vast10--5652460,vast10--5652910,vast11--6102441,vast11--6102440,vast13--198,vast13--228</documents>
        //   </cluster>
        //   <cluster>
        //     <label>mining,history,framework</label>
        //     <representativedocument>infovis04--1382890</representativedocument>
        //     <documents>infovis95--528681,infovis96--559213,infovis98--729562,infovis98--729560,infovis98--729553,infovis99--801864,infovis99--801866,infovis99--801867,infovis00--885097,infovis01--963293,infovis04--1382890,infovis04--1382892,infovis04--1382897,infovis05--1532123,infovis07--4376131,infovis08--4658127,infovis08--4658129,infovis09--5290718,infovis09--5290698,infovis10--206,infovis10--222,infovis10--126,infovis11--239,infovis11--163,vast06--4035762,vast07--4389000,vast07--4389005,vast07--4389013,vast08--4677369,vast09--5333020</documents>
        //   </cluster>
        // </jigsawcluster>

        XMLDocument* doc = new XMLDocument();
        XMLNode* xmldecl = doc->InsertFirstChild( doc->NewDeclaration() );
        XMLNode* jigsawcluster = doc->InsertAfterChild( xmldecl, doc->NewElement( "jigsawcluster" ) );
        XMLNode* name = jigsawcluster->InsertFirstChild( doc->NewElement( "name" ) );
        name->InsertFirstChild( doc->NewText( out_name.c_str() ) );
        XMLNode* type = jigsawcluster->InsertAfterChild( name, doc->NewElement( "type" ) );
        type->InsertFirstChild( doc->NewText( "text-based" ) );
        if (common_terms)
        {
            // Labels (terms)
            std::stringstream terms_ss;
            int n_terms = common_terms->size();
            for (int ii = 0; ii < n_terms; ii++)
            {
                terms_ss << common_terms->at(ii);
                if (ii < n_terms-1)
                {
                    terms_ss << ",";
                }
            }
            XMLNode* commonlabels = jigsawcluster->InsertAfterChild( type, doc->NewElement( "commonlabels" ) );
            commonlabels->InsertFirstChild( doc->NewText( terms_ss.str().c_str() ) );
        }        
        
        // Loop over clusters
        for(std::vector<int>::size_type cc = 0; cc != clusters_terms.size(); cc++) {
        
            // XML
            XMLNode* cluster = jigsawcluster->InsertEndChild( doc->NewElement( "cluster" ) );
            
            // Labels (terms)
            std::stringstream terms_ss;
            int n_terms = clusters_terms.at(cc).size();
            for (int ii = 0; ii < n_terms; ii++)
            {
                terms_ss << clusters_terms.at(cc).at(ii);
                if (ii < n_terms-1)
                {
                    terms_ss << ",";
                }
            }
            XMLNode* label = cluster->InsertFirstChild( doc->NewElement( "label" ) );
            label->InsertFirstChild( doc->NewText( terms_ss.str().c_str() ) );
        
            // docIDs
            std::stringstream docIDs_ss;
            int n_ids = cluster_docIDs.at(cc).size();
            for (int ii = 0; ii < n_ids; ii++)
            {
                docIDs_ss << cluster_docIDs.at(cc).at(ii);
                if (ii < n_ids-1)
                {
                    docIDs_ss << ",";
                }
            }
            XMLNode* documents = cluster->InsertAfterChild( label, doc->NewElement( "documents" ) );
            documents->InsertFirstChild( doc->NewText( docIDs_ss.str().c_str() ) );
        }
        
        // doc->Print();
        doc->SaveFile( out_file.c_str() );
        delete doc;
    };
    
private:

    std::string clusters_name;

    ArrayXi labels;
    MatrixXd U;
    MatrixXd tdm_mean;
    std::vector<ArrayXd> centers;
    std::vector<ArrayXXd> bases;
    std::vector<std::string> docIDs;
    std::vector<std::string> terms;
    VEC_OF_STR_VECS cluster_docIDs;

    void generate_cluster_docIDs()
    {
        // NOTE: Not relying on labels being sequential, even though they should be...
        cluster_docIDs.clear();
        ArrayXi unique_labels = MAPA::UtilityCalcs::UniqueMembers(labels);
        std::map<int, int> label_clusterIdx_map;
        for (int ll = 0; ll < unique_labels.size(); ll++)
        {
            std::vector<std::string> doc_vec;
            cluster_docIDs.push_back(doc_vec);
            label_clusterIdx_map[unique_labels(ll)] = ll;
        }
    
        // NOTE: labels and docIDs had better be the same length...
        for (int ii = 0; ii < labels.size(); ii++)
        {
            cluster_docIDs.at( label_clusterIdx_map[labels(ii)] ).push_back(docIDs.at(ii));
        }
    };
    
    void generate_tmd_mean()
    {
        tdm_mean = MatrixXd::Zero(U.rows(),1);
        for(std::vector<ArrayXd>::iterator it = centers.begin(); it != centers.end(); ++it) 
        {
            tdm_mean += U * (*it).matrix();
        }
        tdm_mean /= centers.size();
    };
    
    std::vector<std::string> generate_tdm_mean_terms(int n_terms = -1)
    {
        // Default to all terms
        int n_top_terms = n_terms;
        if (n_terms < 0)
        {
            n_top_terms = U.rows();
        }
        
        std::vector<std::string> tdm_mean_top_terms;
        
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
    
        for (int ii = 0; ii < n_top_terms; ii++)
        {
            tdm_mean_top_terms.push_back(terms.at(IX(ii)));
        }

        return tdm_mean_top_terms;
    };
    
    VEC_OF_STR_VECS generate_center_terms(int n_terms = -1)
    {
        // Default to all terms
        int n_top_terms = n_terms;
        if (n_terms < 0)
        {
            n_top_terms = U.rows();
        }
        
        VEC_OF_STR_VECS centers_top_terms;
        int center_count = 0;
        for(std::vector<ArrayXd>::iterator it = centers.begin(); it != centers.end(); ++it) {
            if ((*it).size() > 0)
            {
                std::vector<std::string> top_terms;
                // Reproject center back into term space
                // Centers come out as column vectors, so U [D x d] * center [d x 1] = cent [D x 1]
                ArrayXd cent = (U * (*it).matrix()).array().abs();

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
        
        return centers_top_terms;
    };
    
    VEC_OF_STR_VECS generate_center_offset_terms(int n_terms = -1)
    {
        // Default to all terms
        int n_top_terms = n_terms;
        if (n_terms < 0)
        {
            n_top_terms = U.rows();
        }
        
        VEC_OF_STR_VECS centers_offset_top_terms;
        int center_count = 0;
        for(std::vector<ArrayXd>::iterator it = centers.begin(); it != centers.end(); ++it) {
            if ((*it).size() > 0)
            {
                std::vector<std::string> top_terms;
                // Reproject center back into term space
                // Centers come out as column vectors, so U [D x N] * center [N x 1] = cent [D x 1]
                ArrayXd cent = ((U * (*it).matrix()) - tdm_mean).array().abs();
            
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
        return centers_offset_top_terms;
    };
    
    VEC_OF_STR_VECS generate_bases_terms(int n_terms = -1)
    {
        // Default to all terms
        int n_top_terms = n_terms;
        if (n_terms < 0)
        {
            n_top_terms = U.rows();
        }
        
        VEC_OF_STR_VECS bases_top_terms;
        int center_count = 0;
        for(std::vector<ArrayXXd>::iterator it = bases.begin(); it != bases.end(); ++it) 
        {
            if ((*it).size() > 0)
            {
                // NOTE: doing n_terms for each basis dimension!
                std::vector<std::string> top_terms;
                int n_basis_vecs = (*it).rows();
                for (int bb = 0; bb < n_basis_vecs; bb++)
                {
                    // Reproject center back into term space
                    // Bases come out as row vectors, so U [D x N] * center [N x 1] = cent [D x 1]
                    ArrayXd projected = ((*it).row(bb).matrix() * U.transpose()).array().abs().transpose();
            
                    // sort columns independently (only one here)
                    int dim = 1;
                    // sort descending order
                    int ascending = false;
                    // Sorted output matrix
                    ArrayXd Y;
                    // sorted indices for sort dimension
                    ArrayXi IX;
            
                    igl::sort(projected,1,ascending,Y,IX);
            
                    for (int ii = 0; ii < n_top_terms; ii++)
                    {
                        // WARNING: pushing all terms from all bases onto a single vector!!
                        top_terms.push_back(terms.at(IX(ii)));
                    }
                }
                bases_top_terms.push_back(top_terms);
            }
            center_count++;
        }
        
        return bases_top_terms;
    };

    VEC_OF_STR_VECS generate_bases_nomean_terms(std::vector<std::string> stop_terms, int n_top_terms)
    {
        // Copy stop terms over to new map for quick lookup
        std::map<std::string, bool> stopwords_map;
        for (int ss = 0; ss < stop_terms.size(); ss++)
        {
            stopwords_map.insert( std::pair<std::string,bool>(stop_terms.at(ss), true));
        }
        
        VEC_OF_STR_VECS bases_top_terms;
        for(std::vector<ArrayXXd>::iterator it = bases.begin(); it != bases.end(); ++it) 
        {
            // Prepare vector of sorted index arrays for all bases so can iterate through them to pick terms
            if ((*it).size() > 0)
            {
                std::vector<std::string> top_terms;
                int n_basis_vecs = (*it).rows();
                MatrixXi sorted_idxs = MatrixXi::Zero(U.rows(), n_basis_vecs);
                for (int bb = 0; bb < n_basis_vecs; bb++)
                {
                    // Reproject basis vector back into term space
                    // Bases come out as row vectors, so basis [1 x d] * U.T [d x D] = proj [1 x D]
                    ArrayXd projected = ((*it).row(bb).matrix() * U.transpose()).array().abs().transpose();
            
                    // sort columns independently (only one here)
                    int dim = 1;
                    // sort descending order
                    int ascending = false;
                    // Sorted output matrix
                    ArrayXd Y;
                    // sorted indices for sort dimension
                    ArrayXi IX;
            
                    igl::sort(projected,1,ascending,Y,IX);
                    sorted_idxs.col(bb) = IX;
                }
                // reshape to row array so iterating through will automatically iterate across columns then down rows
                sorted_idxs.resize(1,sorted_idxs.size());
                
                // Now iterate through top vectors until have enough terms
                // while also keeping track to not repeat terms
                std::map<int,bool> used_idxs_map;
                for (int ii = 0; ii < sorted_idxs.size(); ii++)
                {
                    int idx = sorted_idxs(ii);
                    std::string term = terms.at(idx);
                    // Only count terms not in stopwords list and not used before
                    if ((stopwords_map.find(term) == stopwords_map.end()) && (used_idxs_map.find(idx) == used_idxs_map.end())) 
                    {
                        top_terms.push_back(term);
                        used_idxs_map.insert( std::pair<int,bool>(idx,true) );
                    }
                    // Stop with this cluster if have enough terms
                    if (top_terms.size() == n_top_terms)
                    {
                        break;
                    }
                }
                bases_top_terms.push_back(top_terms);
            }
        }
        
        return bases_top_terms;
    };

    VEC_OF_STR_VECS generate_centers_nomean_terms(std::vector<std::string> stop_terms, int n_top_terms)
    {
        // Copy stop terms over to new map for quick lookup
        std::map<std::string, bool> stopwords_map;
        for (int ss = 0; ss < stop_terms.size(); ss++)
        {
            stopwords_map.insert( std::pair<std::string,bool>(stop_terms.at(ss), true));
        }
        
        // Run through all centers, reproject, and then store in a matrix where (abs) centers are in the columns. 
        // Simultaneously create a matrix of columns containing cluster index.
        // Also create a matrix of columns containing original indices
        // Form col or row vector of all results by resize to 1 x n_clusters*D
        // Sort
        // Fill in order of sorting, checking against stop terms, and checking whether each cluster's terms are full
        // Keep track of how many total terms have been gathered and break out when all are full n_top_terms * n_clusters
        
        VEC_OF_STR_VECS centers_top_terms;
        int D = U.rows();
        int n_clusters = centers.size();
        MatrixXd coeffs_mat(D,n_clusters);
        MatrixXi idxs_mat(D,n_clusters);
        MatrixXi center_idxs_mat(D,n_clusters);
        for(std::vector<ArrayXd>::size_type cc = 0; cc != centers.size(); cc++) {
            
            std::vector<std::string> top_terms;
            // Reproject center back into term space
            // Centers come out as column vectors, so U [D x d] * center [d x 1] = cent [D x 1]
            ArrayXd cent = (U * centers.at(cc).matrix()).array().abs();
            
            coeffs_mat.col(cc) = cent;
            idxs_mat.col(cc) = ArrayXi::LinSpaced(D,0,D-1);
            center_idxs_mat.col(cc) = ArrayXi::Constant(D,cc);

            centers_top_terms.push_back(top_terms);
        }

        // NOTE: I'm sure we could find a more memory-efficient way of doing this...
        coeffs_mat.resize(n_clusters*D,1);
        idxs_mat.resize(n_clusters*D,1);
        center_idxs_mat.resize(n_clusters*D,1);
        
        ArrayXd coeffs = coeffs_mat;
        ArrayXi idxs = idxs_mat;
        ArrayXi center_idxs = center_idxs_mat;

        // sort columns independently (only one here)
        int dim = 1;
        // sort descending order
        int ascending = false;
        // Sorted output matrix
        ArrayXd Y;
        // sorted indices for sort dimension
        ArrayXi IX;
    
        igl::sort(coeffs,1,ascending,Y,IX);
        
        // Keep track of both total terms accumulated for exit, and used terms so won't repeat with other clusters
        // Algorithm should only assign term to cluster with the most representation of that term
        int total_terms = 0;
        std::map<std::string, bool> usedwords_map;
        for (int ii = 0; ii < IX.size(); ii++)
        {
            std::string term = terms.at(idxs(IX(ii)));
            int center_idx = center_idxs(IX(ii));
            // Only count terms not in stopwords list
            if (   (stopwords_map.find(term) == stopwords_map.end()) \
                && (usedwords_map.find(term) == usedwords_map.end()) \
                && (centers_top_terms.at(center_idx).size() < n_top_terms) )
            {
                centers_top_terms.at(center_idx).push_back(term);
                usedwords_map.insert( std::pair<std::string,bool>(term, true));
                total_terms++;
            }
            // Stop with this cluster if have enough terms
            if (total_terms == (n_top_terms * n_clusters))
            {
                break;
            }
        }
        
        return centers_top_terms;
    };


}; // class def

} // namespace MAPA

#endif

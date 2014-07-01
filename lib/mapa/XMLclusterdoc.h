#ifndef XMLCLUSTERDOC_H
#define XMLCLUSTERDOC_H

/* XMLclusterdoc.h

Generate Jigsaw cluster description XML output document

Eric E Monson – 2014
Duke University

*/

#include "tinyxml2.h"

using namespace tinyxml2;

#include <iostream>
#include <string>
#include <map>
#include <vector>

#include <Eigen/Core>

#include "TDMgenerator.h"
#include "Mapa.h"

using namespace Eigen;


namespace MAPA {

class XMLclusterdoc {

public:

    XMLclusterdoc(MAPA::TDMgenerator *tdm_gen, MAPA::Mapa *mapa, MAPA::SvdlibcSVD *svds, VectorXd *tdm_mean, std::string clusters_name)
    {
        // ---------------------------------------------
        // Convert MAPA output to document labels and terms
    
        ArrayXi labels = mapa->GetLabels();
        std::vector<ArrayXd> centers = mapa->GetPlaneCenters();
        std::vector<ArrayXXd> bases = mapa->GetPlaneBases();

        std::vector<std::string> docIDs = tdm_gen->getDocIDs();
        std::vector<std::string> terms = tdm_gen->getTerms();
    
        // TODO: may need to figure out doc closest to center since center not a doc...
    
        int n_top_terms = 3;
    
        // ---------------------------------------------
        // TDM mean vector terms
    
        ArrayXd cent = tdm_mean->array().abs();
    
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
                ArrayXd cent = (svds->matrixU() * (*it).matrix()).array().abs();

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
                ArrayXd cent = ((svds->matrixU() * (*it).matrix()) - (*tdm_mean)).array().abs();
            
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
                    ArrayXd projected = ((*it).row(bb).matrix() * svds->matrixU().transpose()).array().abs().transpose();
            
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
        name->InsertFirstChild( doc->NewText( clusters_name.c_str() ) );
        XMLNode* type = jigsawcluster->InsertAfterChild( name, doc->NewElement( "type" ) );
        type->InsertFirstChild( doc->NewText( "text-based" ) );
        
        
        center_count = 0;
        // TODO: change this loop definition!!
        for(std::vector<ArrayXXd>::iterator it = bases.begin(); it != bases.end(); ++it) {
        
            // XML
            XMLNode* cluster = jigsawcluster->InsertEndChild( doc->NewElement( "cluster" ) );
            
            // Centers
            std::stringstream terms_ss;
            for (int ii = 0; ii < n_top_terms; ii++)
            {
                terms_ss << centers_top_terms.at(center_count).at(ii);
                if (ii < n_top_terms-1)
                {
                    terms_ss << ",";
                }
            }
            XMLNode* label = cluster->InsertFirstChild( doc->NewElement( "label" ) );
            label->InsertFirstChild( doc->NewText( terms_ss.str().c_str() ) );
        
            // docIDs
            std::stringstream docIDs_ss;
            int n_ids = cluster_docIDs.at(center_count).size();
            for (int ii = 0; ii < n_ids; ii++)
            {
                docIDs_ss << cluster_docIDs.at(center_count).at(ii);
                if (ii < n_ids-1)
                {
                    docIDs_ss << ",";
                }
            }
            XMLNode* documents = cluster->InsertAfterChild( label, doc->NewElement( "documents" ) );
            documents->InsertFirstChild( doc->NewText( docIDs_ss.str().c_str() ) );
            
            center_count++;
        }
        
        doc->Print();
        doc->SaveFile( "pretty.xml" );
        delete doc;

    };
    
private:

    
}; // class def

} // namespace MAPA

#endif

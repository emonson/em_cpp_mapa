#ifndef JIGTOKENIZER_H
#define JIGTOKENIZER_H

/* JIGtokenizer.h

Generate TDM from Jigsaw .jig files of documents

Eric E Monson – 2014
Duke University

*/

#include "tinyxml2.h"

using namespace tinyxml2;

#include <iostream>
#include <string>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "TDMgenerator.h"

using namespace Eigen;


namespace MAPA {

class JIGtokenizer {

public:

    JIGtokenizer(std::string jigfile, MAPA::TDMgenerator *tdm_generator)
    {
        tdm_gen = tdm_generator;
        
        // Load the document to parse
        XMLDocument doc;
        doc.LoadFile( jigfile.c_str() );
        if (doc.ErrorID() > 0)
        {
            std::cerr << "Error Loading " << jigfile << std::endl;
            std::cerr << "Error ID: " << doc.ErrorID() << std::endl;
        }
        else
        {
            for (XMLElement* documentElement = doc.FirstChildElement("documents")->FirstChildElement("document"); 
                 documentElement; 
                 documentElement = documentElement->NextSiblingElement("document")) 
            {
                // Extract the document ID from the XML
                XMLElement* docID = documentElement->FirstChildElement("docID");
                std::string id_str(docID->GetText());
        
                // Extract the document text from the XML
                XMLElement* docText = documentElement->FirstChildElement("docText");
                std::string text_str(docText->GetText());

                // Add document to generator
                tdm_gen->addDocument(id_str, text_str);
            }
        }
    };
    
private:

    MAPA::TDMgenerator *tdm_gen;
    
}; // class def

} // namespace MAPA

#endif

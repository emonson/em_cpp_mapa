#include "tinyxml2.h"
#include "xmltest.h"

#include <cstdlib>
#include <cstring>
#include <ctime>

#if defined( _MSC_VER )
#include <direct.h>		// _mkdir
#include <crtdbg.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
_CrtMemState startMemState;
_CrtMemState endMemState;
#else
#include <sys/stat.h>	// mkdir
#endif

using namespace tinyxml2;

#include <iostream>
#include <string>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "TDMgenerator.h"

int main( int argc, const char** argv )
{
    #if defined( _MSC_VER ) && defined( DEBUG )
        _CrtMemCheckpoint( &startMemState );
    #endif
    
    const char* filename = "/Users/emonson/Programming/em_cpp_mapa/tools/tokenize_test/InfovisVAST-papers.jig";

	// Load the document to parse
	XMLDocument doc;
    doc.LoadFile( filename );
    printf("Error ID %d\n", doc.ErrorID());
    if (doc.ErrorID() > 0)
    {
        printf("Error Loading %s\n", filename);
        return doc.ErrorID();
    }

    int min_term_length = 3;
    int min_term_freq = 5;
    MAPA::TDMgenerator tdm_gen(min_term_length, min_term_freq);
    
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
        tdm_gen.addDocument(id_str, text_str);
    }
    
    // Generate TDM from counts
    tdm_gen.generateTDM();
    
    Eigen::SparseMatrix<double,0,long> tdm = tdm_gen.getTFIDF();
    
    std::cout << "TDM: " << tdm.rows() << " x " << tdm.cols() << ", " << tdm.nonZeros() << std::endl << std::endl;

    return EXIT_SUCCESS;
}

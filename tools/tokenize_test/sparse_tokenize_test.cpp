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

// char_sep_example_3.cpp
// http://www.boost.org/doc/libs/1_40_0/libs/tokenizer/char_separator.htm

#include <iostream>
#include <fstream>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <string>
#include <map>

#include <Eigen/Core>
#include <Eigen/SparseCore>


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

	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep(" \t\n¡!¿?⸘‽“”‘’‛‟.,‚„'\"′″´˝^°¸˛¨`˙˚ªº…:;&_¯­–‑—§#⁊¶†‡@%‰‱¦|/\\ˉˆ˘ˇ-‒~*‼⁇⁈⁉$€¢£‹›«»<>{}[]()=+|01234567890");
    
    // read in stopwords from text file
    std::ifstream stopfile("/Users/emonson/Programming/em_cpp_mapa/tools/tokenize_test/tartarus_org_stopwords.txt", std::ios_base::in);

    // load stopwords into hash map
    std::map<std::string, bool> stopwords_map;
    std::string s;
    while (stopfile >> s) {
        if (stopwords_map.find(s) == stopwords_map.end())
        {
            stopwords_map[s] = true;
            std::cout << s << " · ";
        }
    }
    std::cout << std::endl;
    stopfile.close();
    
    std::map<std::string, int> term_count_map;
    std::map<std::string, int>::iterator term_count_it;

    std::map<std::string, std::vector<int> > term_docIndexVec_map;
    std::map<std::string, std::vector<int> >::iterator term_indexVec_it;
    
    std::map<int, std::string> index_docID_map;
    std::map<int, std::string> termIndex_term_map;
    
    // CONSTANTS
    int MIN_TERM_LENGTH = 2;
    int MIN_TERM_COUNT = 2;
    
    int docIndex = 0;
    long n_terms_counted = 0;

    for (XMLElement* documentElement = doc.FirstChildElement("documents")->FirstChildElement("document"); 
         documentElement; 
         documentElement = documentElement->NextSiblingElement("document")) 
    {
        // Extract the document ID from the XML
        XMLElement* docID = documentElement->FirstChildElement("docID");
        std::string id_str(docID->GetText());
        
        // Set up hash map of docID string and index keys which will be used in count vectors
        index_docID_map[docIndex] = id_str;
        
        // Extract the document text from the XML
        XMLElement* docText = documentElement->FirstChildElement("docText");
        std::string text_str(docText->GetText());
        tokenizer tokens(text_str, sep);
        for (tokenizer::iterator tok_iter = tokens.begin(); tok_iter != tokens.end(); ++tok_iter)
        {
            // std::cout << "<" << *tok_iter << "> ";
            std::string tmp = *tok_iter;
            // NOTE: Right now doing a rough length check
            if (tmp.length() >= MIN_TERM_LENGTH) 
            {
                // Only count terms not in stopwords list
                if (stopwords_map.find(tmp) == stopwords_map.end()) 
                {
                    // Check for all caps, otherwise convert to lowercase
                    //   (maybe should just be turning everything to lowercase...)
                    if (!boost::all(tmp, boost::is_upper())) 
                    {
                        boost::to_lower(tmp);
                    }
                    if (term_count_map.find(tmp) == term_count_map.end()) 
                    {
                        // Initialize term count and doc index vector maps for new term
                        term_count_map[tmp] = 0;
                        std::vector<int> newvec;
                        term_docIndexVec_map[tmp] = newvec;
                    }
                    term_count_map[tmp]++;
                    term_docIndexVec_map[tmp].push_back(docIndex);
                    n_terms_counted++;
                }
            }
            else
            {
                std::cout << "-" << tmp;
            }
        }
        docIndex++;
        // std::cout << "\n";
    }
    
    // Now that we have the terms and the documents they came from, we need to 
    // create the actual term-document vectors out of Eigen data structures
    
	int n_docs = docIndex;    // is (and has to be) one more than the last zero-based doc index
    int n_terms = int(term_count_map.size());
    
    // Filling sparse Eigen matrix with triplets.
    // Will rely on Eigen to do the sums when it forms the sparse matrix out of these triplets.
    // TODO: Should check whether it's faster to do the sums with int rather than double...
    typedef Eigen::Triplet<double,long> T;
    std::vector<T> count_triplets_vector;
    count_triplets_vector.reserve(n_terms_counted);
        
    int term_idx = 0;
    
    // Run through all of the entries in the term totals and correpsonding doc index vectors
    for ( term_count_it=term_count_map.begin(); term_count_it != term_count_map.end(); term_count_it++ )
    {
        std::string term = (*term_count_it).first;
        int term_count = (*term_count_it).second;
        
        // First, check if count passes threshold (could base this on percentiles in future...)
        if (term_count < MIN_TERM_COUNT)
        {
            continue;
        }
        
        // NOTE: Could set up here some sort of entropy thresholds
        
        // Record the term with its index as key
        termIndex_term_map[term_idx] = term;
        
        // Print
        std::cout << term << " => " << term_count << std::endl;
        
        // Convenience vector so iteration and access are more clear
        std::vector<int> docIndex_vec = term_docIndexVec_map[term];
        
        // Run through doc index vectors and increment counts in real data arrays
        std::cout << "    ";
        for ( int ii = 0; ii < docIndex_vec.size(); ii++ )
        {
            std::cout << docIndex_vec[ii] << " ";
            // Increment count sums
            count_triplets_vector.push_back(T(term_idx, docIndex_vec[ii], 1));
        }
        std::cout << std::endl;
        term_idx++;
    }
    
    // Create the actual Term-Document Matrix
    
    // NOTE: the "long" specification for the indices is necessary to match the 
    //   pointer type for the SVDLIBC matrices...
    // NOTE: using the fact that term_idx will get incremented one beyond the last
    //   index value, so equal to the number of terms...
    
    Eigen::SparseMatrix<double,0,long> tdm(term_idx, n_docs);
    tdm.setFromTriplets(count_triplets_vector.begin(), count_triplets_vector.end());

//     std::cout << "Original (sparse) matrix" << std::endl;
//     std::cout << tdm << std::endl << std::endl;
    
    std::cout << std::endl << term_count_map.size() << " terms in dictionary, " << term_idx << " terms used" << std::endl << std::endl;

    return EXIT_SUCCESS;
}
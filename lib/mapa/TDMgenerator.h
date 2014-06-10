#ifndef TDMGENERATOR_H
#define TDMGENERATOR_H

/* TDMgenerator.h

Tokenize strings from documents and generate term-document matrix (TDM)

char_sep_example_3.cpp
http://www.boost.org/doc/libs/1_40_0/libs/tokenizer/char_separator.htm

The method I'm using here is not optimal for memory usage, but gives more
flexibility. The maps of term counts and documents are persistently stored,
and can be updated at any time by adding more documents. The TDM is
generated lazily, upon request. The terms are only filtered for non-letter
characters at first, not for length or frequency or stopwords. Those things
are only taken into account upon TDM generation so parameters can be changed
or stopwords added without having to reingest documents, but just recalculate
the TDM.

Eric E Monson – 2014
Duke University

*/

    // TODO: Add way to return term and document ID vectors!!
    // TODO: Should check whether it's faster to do the sums with int triplet values rather than double...
    
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


using namespace Eigen;


namespace MAPA {

class TDMgenerator {

public:

    TDMgenerator(int min_term_length = 2, int min_term_count = 2)
    {
        tdm_current = false;
        tfidf_current = false;
        VERBOSE = false;
            
        MIN_TERM_LENGTH = min_term_length;
        MIN_TERM_COUNT = min_term_count;
    
        docIndex = 0;
        n_terms_counted = 0;

        // std::string stop_file_name = "/Users/emonson/Programming/em_cpp_mapa/tools/tokenize_test/tartarus_org_stopwords.txt";
        // load_stopwords_from_file(stop_file_name);
        generate_stopwords();
    };
    
    // ---------------------------------
    // Parameters updates
    // All of these dirty the TDM, so it has to be regenerated after changes.
    // 
    void setMinTermLength(int min_term_length)
    {
        MIN_TERM_LENGTH = min_term_length;
        tdm_current = false;
    };

    void setMinTermCount(int min_term_count)
    {
        MIN_TERM_COUNT = min_term_count;
        tdm_current = false;
    };
    
    void setVerbose(bool verbose)
    {
        VERBOSE = verbose;
    };

    int addStopwords(std::string space_separated_words)
    {
        std::stringstream ss(space_separated_words);
        std::string s;
        int new_stopwords_count = 0;
        
        // load stopwords into hash map
        while (std::getline(ss, s, ' ')) 
        {
            if (!s.empty())
            {
                stopwords_map.insert( std::pair<std::string,int>(s, true));
                new_stopwords_count++;
                if (VERBOSE) std::cout << s << " · ";
            }
        }
        if (VERBOSE) std::cout << std::endl;

        tdm_current = false;
        return new_stopwords_count;
    };
    
    void regenerateStopwordsList()
    {
        generate_stopwords();
    }

    // ---------------------------------
    void addDocument(std::string id_str, std::string text_str)
    {
        tdm_current = false;
        tfidf_current = false;

	    // NOTE: not sure how to keep from recreating this each time...
	    boost::char_separator<char> sep(" \t\n¡!¿?⸘‽“”‘’‛‟.,‚„'\"′″´˝^°¸˛¨`˙˚ªº…:;&_¯­–‑—§#⁊¶†‡@%‰‱¦|/\\ˉˆ˘ˇ-‒~*‼⁇⁈⁉$€¢£‹›«»<>{}[]()=+|01234567890");
        
        // Set up hash map of docID string and index keys which will be used in count vectors
        index_docID_map[docIndex] = id_str;
        
        tokenizer tokens(text_str, sep);
        for (tokenizer::iterator tok_iter = tokens.begin(); tok_iter != tokens.end(); ++tok_iter)
        {
            std::string term = *tok_iter;
            // Change everything to lowercase
            boost::to_lower(term);
            
            // Initialize term count and doc index vector maps for new term
            if (term_count_map.find(term) == term_count_map.end()) 
            {
                term_count_map[term] = 0;
                std::vector<int> newvec;
                term_docIndexVec_map[term] = newvec;
            }
            term_count_map[term]++;
            term_docIndexVec_map[term].push_back(docIndex);
            n_terms_counted++;
        }
        docIndex++;
        return;
    };
    
    // ---------------------------------
    void generateTDM()
    {
        // Now that we have the terms and the documents they came from, we need to 
        // create the actual term-document vectors out of Eigen data structures
    
        int n_docs = docIndex;    // is (and has to be) one more than the last zero-based doc index
        int n_terms = int(term_count_map.size());
    
        // Filling sparse Eigen matrix with triplets.
        // Will rely on Eigen to do the sums when it forms the sparse matrix out of these triplets.
        count_triplets_vector.clear();
        count_triplets_vector.reserve(n_terms_counted);
        
        int term_idx = 0;
    
        // Run through all of the entries in the term totals and correpsonding doc index vectors
        for ( term_count_it=term_count_map.begin(); term_count_it != term_count_map.end(); term_count_it++ )
        {
            std::string term = (*term_count_it).first;
            int term_count = (*term_count_it).second;
        
            // First, check if count and length pass thresholds
            if (term_count < MIN_TERM_COUNT || term.length() < MIN_TERM_LENGTH)
            {
                continue;
            }
            // Only count terms not in stopwords list
            if (stopwords_map.find(term) != stopwords_map.end()) 
            {
                continue;
            }
            
            // Record the term with its index as key
            termIndex_term_map[term_idx] = term;
        
            // Print
            if (VERBOSE) std::cout << term << " => " << term_count << std::endl;
        
            // Convenience vector so iteration and access are more clear
            std::vector<int> docIndex_vec = term_docIndexVec_map[term];
        
            // Run through doc index vectors and increment counts in real data arrays
            if (VERBOSE) std::cout << "    ";
            for ( int ii = 0; ii < docIndex_vec.size(); ii++ )
            {
                if (VERBOSE) std::cout << docIndex_vec[ii] << " ";
                // Increment count sums
                count_triplets_vector.push_back(T(term_idx, docIndex_vec[ii], 1));
            }
            if (VERBOSE) std::cout << std::endl;
            term_idx++;
        }
        
        // NOTE: the "long" specification for the indices is necessary to match the 
        //   pointer type for the SVDLIBC matrices...
        // NOTE: using the fact that term_idx will get incremented one beyond the last
        //   index value, so equal to the number of terms...
    
        // Create the actual Term-Document Matrix
        tdm.resize(term_idx, n_docs);
        tdm.setFromTriplets(count_triplets_vector.begin(), count_triplets_vector.end());

        if (VERBOSE) std::cout << std::endl << term_count_map.size() << " terms in dictionary, " << term_idx << " terms used" << std::endl << std::endl;

        tdm_current = true;
        return;
    };
    
    void calculateTFIDF()
    {
        std::cout << "WARNING: calculateTFIDF() not implemented yet!" << std::endl;
        
        tfidf_current = true;
        return;
    };
    
    // ---------------------------------
    // Accessors
    // 
    Eigen::SparseMatrix<double,0,long> getTDM()
    {
        if (!tdm_current)
        {
            generateTDM();
        }
        return tdm;
    };
    
    Eigen::SparseMatrix<double,0,long> getTFIDF()
    {
        if (!tdm_current)
        {
            generateTDM();
        }
        if (!tfidf_current)
        {
            calculateTFIDF();
        }
        return tfidf;
    };
    
private:

    bool tdm_current;
    bool tfidf_current;
    Eigen::SparseMatrix<double,0,long> tdm;
    Eigen::SparseMatrix<double,0,long> tfidf;

	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
    std::map<std::string, bool> stopwords_map;

    std::map<std::string, int> term_count_map;
    std::map<std::string, int>::iterator term_count_it;

    std::map<std::string, std::vector<int> > term_docIndexVec_map;
    std::map<std::string, std::vector<int> >::iterator term_indexVec_it;
    
    std::map<int, std::string> index_docID_map;
    std::map<int, std::string> termIndex_term_map;
    
    int MIN_TERM_LENGTH;
    int MIN_TERM_COUNT;
    bool VERBOSE;
    
    int docIndex;
    long n_terms_counted;

    typedef Eigen::Triplet<double,long> T;
    std::vector<T> count_triplets_vector;


    void generate_stopwords()
    {
        stopwords_map.clear();
        
        // http://stackoverflow.com/questions/236129/how-to-split-a-string-in-c
        
        std::string stopwords = "a about above after again against all also am an and another \
                        any are aren as at back be because been before being below \
                        between both but by can can cannot could couldn d daren did \
                        didn do does doesn doing don down during each even ever \
                        every few first five for four from further get go goes had \
                        hadn has hasn have haven having he her here here hers \
                        herself him himself his how how however i if in into is isn \
                        it its itself just least less let like ll m made make many \
                        may me might mightn more most must mustn my myself needn \
                        never no nor not now of off on once one only or other ought \
                        oughtn our ours ourselves out over own put re s said same \
                        say says second see seen shall shan she should shouldn since \
                        so some still such t take than that that the their theirs \
                        them themselves then there there these they this those three \
                        through to too two under until up us ve very was wasn way we \
                        well were weren what what when when where where whether \
                        which while who who whom why why will with won would wouldn \
                        you your yours yourself yourselves";
        std::stringstream ss(stopwords);
        std::string s;
        
        // load stopwords into hash map
        while (std::getline(ss, s, ' ')) 
        {
            if (!s.empty())
            {
                stopwords_map.insert( std::pair<std::string,int>(s, true));
                if (VERBOSE) std::cout << s << " · ";
            }
        }
        if (VERBOSE) std::cout << std::endl;
        
        return;
    };
    

    void load_stopwords_from_file(std::string stopfile_name)
    {
        stopwords_map.clear();
        
        // read in stopwords from text file
        std::ifstream stopfile(stopfile_name.c_str());
        
        if (stopfile) // Verify that the file was open successfully
        {
            // load stopwords into hash map
            for (std::string s; std::getline(stopfile, s); ) 
            {
                stopwords_map.insert( std::pair<std::string,int>(s, true));
                if (VERBOSE) std::cout << s << " · ";
            }
            if (VERBOSE) std::cout << std::endl;
        }
        else
        {
             std::cerr << "File could not be opened!\n";
             std::cerr << "Error code: " << strerror(errno);
        }
        
        stopfile.close();
        return;
    };
    
}; // class def

} // namespace MAPA

#endif

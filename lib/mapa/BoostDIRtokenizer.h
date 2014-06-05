#ifndef BOOSTDIRTOKENIZER_H
#define BOOSTDIRTOKENIZER_H

/* DIRtokenizer.h

Generate TDM from directory of text files of documents

Eric E Monson – 2014
Duke University

*/

// Skeleton of code taken from boost docs:
// http://www.boost.org/doc/libs/1_55_0/libs/filesystem/example/simple_ls.cpp

#define BOOST_FILESYSTEM_NO_DEPRECATED

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
namespace fs = boost::filesystem;

#include <iostream>
#include <fstream>
#include <string>

#include <Eigen/SparseCore>

#include "TDMgenerator.h"

using namespace Eigen;


namespace MAPA {

class BoostDIRtokenizer {

public:

    BoostDIRtokenizer(std::string dirpath, int min_term_length = 2, int min_term_count = 2)
    {
        tdm_gen.setMinTermLength(min_term_length);
        tdm_gen.setMinTermCount(min_term_count);

        fs::path p(fs::current_path());

        p = fs::system_complete(dirpath.c_str());
    
        if (!fs::exists(p))
        {
            std::cout << "\nNot found: " << p << std::endl;
        }
        else if (!fs::is_directory(p))
        {
            std::cout << "\nNot a directory: " << p << std::endl;    
        }
        else
        {
            std::cout << "\nIn directory: " << p << "\n\n";
            fs::directory_iterator end_iter;
            for (fs::directory_iterator dir_itr(p);
                dir_itr != end_iter;
                ++dir_itr)
            {
                try
                {
                    if (fs::is_regular_file(dir_itr->status()))
                    {
                        // Found an actual file
                        std::cout << dir_itr->path().filename() << "\n";

                        // http://stackoverflow.com/questions/2602013/read-whole-ascii-file-into-c-stdstring
                        std::ifstream t(dir_itr->path().c_str());
                        std::stringstream buffer;
                        buffer << t.rdbuf();

                        // Add document to generator
                        // addDocument(ID_string, text_string)
                        tdm_gen.addDocument(dir_itr->path().filename().native(), buffer.str());
                    }
                }
                catch (const std::exception & ex)
                {
                    std::cout << dir_itr->path().filename() << " " << ex.what() << std::endl;
                }
            }
        }

    };
    
    Eigen::SparseMatrix<double,0,long> getTDM()
    {
        return tdm_gen.getTDM();
    };
    
    Eigen::SparseMatrix<double,0,long> getTFIDF()
    {
        return tdm_gen.getTFIDF();
    };
    

private:

    MAPA::TDMgenerator tdm_gen;
    
}; // class def

} // namespace MAPA

#endif

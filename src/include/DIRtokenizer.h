#ifndef DIRTOKENIZER_H
#define DIRTOKENIZER_H

/* DIRtokenizer.h

Generate TDM from directory of text files of documents

Eric E Monson – 2014
Duke University

*/

// Directory navigation code taken from dirent.h ls.c example
// http://www.softagalleria.net/download/dirent/dirent-1.20.1.zip


#include <iostream>
#include <fstream>
#include <string>

#include <Eigen/SparseCore>

#include "dirent.h"     /* directory navigation */

#include "UtilityCalcs.h"
#include "TDMgenerator.h"

using namespace Eigen;


namespace MAPA {

class DIRtokenizer {

public:

    DIRtokenizer(std::string dirpath, MAPA::TDMgenerator *tdm_generator)
    {
        tdm_gen = tdm_generator;
        
        DIR *dir;
        struct dirent *ent;
            
        /* Open directory stream */
        dir = opendir(dirpath.c_str());
        if (dir != NULL) {

            /* Print all files and directories within the directory */
            while ((ent = readdir (dir)) != NULL) {
                switch (ent->d_type) {
                case DT_REG:
                    // printf ("%s\n", ent->d_name);

                    try
                    {
                        // Found an actual file
                        // std::cout << ent->d_name << "\n";

                        // http://stackoverflow.com/questions/2602013/read-whole-ascii-file-into-c-stdstring
                        std::string whole_file_path = MAPA::UtilityCalcs::PathAppend(dirpath, ent->d_name);
                        std::ifstream t(whole_file_path.c_str());
                        std::stringstream buffer;
                        buffer << t.rdbuf();

                        // Add document to generator
                        // addDocument(ID_string, text_string)
                        tdm_gen->addDocument(ent->d_name, buffer.str());
                    }
                    catch (const std::exception & ex)
                    {
                        std::cout << ent->d_name << " " << ex.what() << std::endl;
                    }

                    break;

                default:
                    break;
                }
            }

            closedir(dir);
        }

    };    

private:

    MAPA::TDMgenerator *tdm_gen;

}; // class def

} // namespace MAPA

#endif

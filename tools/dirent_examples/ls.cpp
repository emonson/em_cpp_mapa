/* 
 * An example demonstrating basic directory listing.
 *
 * Compile this file with Visual Studio 2008 project vs2008.sln and run the
 * produced command in console with a directory name argument.  For example,
 * command
 *
 *     ls "c:\Program Files"
 *
 * might output something like
 *
 *     ./
 *     ../
 *     7-Zip/
 *     Internet Explorer/
 *     Microsoft Visual Studio 9.0/
 *     Microsoft.NET/
 *     Mozilla Firefox/
 *
 * The ls command provided by this file is only an example.  That is, the
 * command does not have any fancy options like "ls -al" in Linux and the
 * command does not support file name matching like "ls *.c".
 */
 
#include <iostream>
#include <string>
#include "dirent.h"

void list_directory( const char *dirname )
{
    DIR *dir;
    struct dirent *ent;
                
    /* Open directory stream */
    dir = opendir(dirname);
    if (dir != NULL) {

        /* Print all files and directories within the directory */
        while ((ent = readdir (dir)) != NULL) {
            switch (ent->d_type) {
            case DT_REG:
                printf ("%s\n", ent->d_name);
                break;

            case DT_DIR:
                printf ("%s/\n", ent->d_name);
                break;

            case DT_LNK:
                printf ("%s@\n", ent->d_name);
                break;

            default:
                printf ("%s*\n", ent->d_name);
            }
        }

        closedir(dir);

    }
}

int main( int argc, const char** argv )
{
    for (int i = 1; i < argc; i++) {
        list_directory(argv[i]);
    }

    /* List current working directory if no arguments on command line */
    if (argc == 1) {
        list_directory(".");
    }
    return EXIT_SUCCESS;
}

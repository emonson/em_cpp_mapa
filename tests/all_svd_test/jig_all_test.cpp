#include <iostream>
#include <string>
#include <time.h>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/SparseCore>

#include "mapa_config.h"
#include "UtilityCalcs.h"
#include "TDMgenerator.h"
#include "JIGtokenizer.h"
#include "SvdlibcSVD.h"

// #include "redsvd.hpp"

#include "EigenRandomSVD.h"
#include "LinalgIO.h"
#include "Precision.h"
#include "RandomSVD.h"
#include "SVD.h"

int main( int argc, const char** argv )
{
	std::string data_dir = MAPA::UtilityCalcs::PathAppend(MAPA_SOURCE_DIR, "data");
	std::string filename = MAPA::UtilityCalcs::PathAppend(data_dir, "InfovisVAST-papers.jig");

    int min_term_length = 3;
    int min_term_count = 5;
    MAPA::TDMgenerator tdm_gen(min_term_length, min_term_count);
	MAPA::JIGtokenizer jig_tok(filename, &tdm_gen);
    
    // Eigen sparse matrix
    Eigen::SparseMatrix<double,0,long> tdm = tdm_gen.getTDM();
    // Eigen dense matrix
    Eigen::MatrixXd tdm_dense = tdm;
    // Sam dense matrix
    FortranLinalg::DenseMatrix<Precision> tdm_sam(tdm.rows(), tdm.cols());
    tdm_sam.setDataPointer(tdm_dense.data());
    
    // DEBUG - write data out for further testing
    // FortranLinalg::LinalgIO<Precision>::writeMatrix("jig_tdm", tdm_sam);

    std::cout << std::endl << "TDM (sparse input matrix): " << tdm.rows() << " x " << tdm.cols() << ", " << tdm.nonZeros() << " nonzeros" << std::endl << std::endl;
    
    int rank = 100;
    int power_iterations = 3;

    // --------------------------------
    // SVDLIBC (sparse)
    
    printf("SVDLIBC ");
    clock_t t = clock();    
    MAPA::SvdlibcSVD svds(tdm, rank);    
    t = clock() - t;
    printf("Elapsed time: %.10f sec.\n", (double)t/CLOCKS_PER_SEC );
    
    std::cout << "U: " << svds.matrixU().rows() << " x " << svds.matrixU().cols() << std::endl;
    // std::cout << svds.matrixU() << std::endl << std::endl;
    std::cout << "V: " << svds.matrixV().rows() << " x " << svds.matrixV().cols() << std::endl;
    // std::cout << svds.matrixV() << std::endl << std::endl;
    std::cout << "S: ";
    std::cout << svds.singularValues().transpose() << std::endl;

    Eigen::MatrixXd Xred = svds.matrixV() * svds.singularValues().asDiagonal();
    std::cout << "X reduced: " << Xred.rows() << " x " << Xred.cols() << std::endl << std::endl;

    // --------------------------------
    // Eigen standard JacobiSVD (dense)
    
    printf("Eigen standard JacobiSVD ");
    t = clock();
    JacobiSVD<MatrixXd> svd_e(tdm_dense, Eigen::ComputeThinU | Eigen::ComputeThinV);
    t = clock() - t;
    printf("Elapsed time: %.10f sec.\n", (double)t/CLOCKS_PER_SEC );
    
    std::cout << "U: " << svd_e.matrixU().rows() << " x " << svd_e.matrixU().cols() << std::endl;
    // std::cout << svd_e.matrixU() << std::endl << std::endl;
    std::cout << "V: " << svd_e.matrixV().rows() << " x " << svd_e.matrixV().cols() << std::endl;
    // std::cout << svd_e.matrixV() << std::endl << std::endl;
    std::cout << "S: ";
    std::cout << svd_e.singularValues().head(rank).transpose() << std::endl;

    Eigen::MatrixXd Xred_e = svd_e.matrixV().leftCols(rank) * svd_e.singularValues().head(rank).asDiagonal();
    std::cout << "X reduced Eigen: " << Xred_e.rows() << " x " << Xred_e.cols() << std::endl << std::endl;
    
    // --------------------------------
    // RedSVD test (row-major sparse)
    
//     REDSVD::SMatrixXf tdm_r = tdm.cast<float>();
// 
//     printf("RedSVD ");
//     t = clock();
//     REDSVD::RedSVD svd_r(tdm_r, rank);
//     t = clock() - t;
//     printf("Elapsed time: %.10f sec.\n", (double)t/CLOCKS_PER_SEC );
// 
//     std::cout << "U: " << svd_r.matrixU().rows() << " x " << svd_r.matrixU().cols() << std::endl;
//     // std::cout << svd_r.matrixU() << std::endl << std::endl;
//     std::cout << "V: " << svd_r.matrixV().rows() << " x " << svd_r.matrixV().cols() << std::endl;
//     // std::cout << svd_r.matrixV() << std::endl << std::endl;
//     std::cout << "S: ";
//     std::cout << svd_r.singularValues().head(rank).transpose() << std::endl;
// 
//     Eigen::MatrixXf Xred_r = svd_r.matrixV().leftCols(rank) * svd_r.singularValues().head(rank).asDiagonal();
//     std::cout << "X reduced redsvd: " << Xred_r.rows() << " x " << Xred_r.cols() << std::endl << std::endl;
    
    // --------------------------------
    // Eigen random SVD (dense – Sam)
    
    printf("Eigen Random SVD ");
    t = clock();
    EigenLinalg::RandomSVD svd_er(tdm_dense, rank, 3);
    t = clock() - t;
    printf("Elapsed time: %.10f sec.\n", (double)t/CLOCKS_PER_SEC );
    
    std::cout << "U: " << svd_er.U.rows() << " x " << svd_er.U.cols() << std::endl;
    // std::cout << svd_er.matrixU() << std::endl << std::endl;
    std::cout << "S: ";
    std::cout << svd_er.S.head(rank).transpose() << std::endl << std::endl;

    // --------------------------------
    // Sam dense standard Fortran version
    
    printf("LAPACK standard SVD ");
    t = clock();
    FortranLinalg::SVD<Precision> svd_fs(tdm_sam, false);
    t = clock() - t;
    printf("Elapsed time: %.10f sec.\n", (double)t/CLOCKS_PER_SEC );

    std::cout << "U: " << svd_fs.U.M() << " x " << svd_fs.U.N() << std::endl;
    std::cout << "S: ";
    for (int ii = 0; ii < rank; ii++)
    {
        std::cout << svd_fs.S(ii) << " ";
    }
    std::cout << std::endl << std::endl;

    // --------------------------------
    // Sam dense random Fortran version
    
    printf("LAPACK random SVD ");
    t = clock();
    FortranLinalg::RandomSVD<Precision> svd_fr(tdm_sam, rank, power_iterations, false);
    t = clock() - t;
    printf("Elapsed time: %.10f sec.\n", (double)t/CLOCKS_PER_SEC );

    std::cout << "U: " << svd_fr.U.M() << " x " << svd_fr.U.N() << std::endl;
    std::cout << "S: ";
    for (int ii = 0; ii < rank; ii++)
    {
        std::cout << svd_fr.S(ii) << " ";
    }
    std::cout << std::endl << std::endl;

    // --------------------------------
    // Matlab

    // f = fopen('jig_tdm','rb');
    // [data,count] = fread(f, 'float64');
    // tdm = reshape(data, [2074,578]);
    // [U,S,V] = svd(tdm);
    // diag(S)
    // 
    // 120.2292, 55.7848, 49.3458, 43.6356, 40.3711, 33.7784, 33.3014, 31.9810, 31.0635, 
    // 30.3060, 29.0700, 28.5012, 28.0047, 27.5122, 27.0972, 26.8200, 26.2419, 25.8579, 
    // 25.7415, 25.0880, 24.9734, 24.6606, 24.4398, 24.1955, 23.8598, 23.7656, 23.4163, 
    // 23.3179, 23.1779, 23.0825, 22.6901, 22.6844, 22.3345, 22.2170, 22.0876, 21.8041,
    // 21.7062, 21.6118, 21.4724, 21.0920, 21.0333, 20.9851, 20.9017, 20.7311, 20.5557, 
    // 20.3701, 20.2316, 20.0848, 20.0000, 19.8401, 19.7174, 19.5898, 19.5232, 19.4532, 
    // 19.3561, 19.1832, 19.1378, 19.0323, 18.9137, 18.8020, 18.7371, 18.4977, 18.3716, 
    // 18.3004, 18.2333, 18.1608, 18.0775, 17.9601, 17.8811, 17.7346, 17.6749, 17.6081, 
    // 17.5227, 17.4908, 17.4475, 17.2730, 17.2121, 17.1433, 17.0005, 16.9598, 16.8710, 
    // 16.8006, 16.7648, 16.7112, 16.5711, 16.5388, 16.4580, 16.3859, 16.2340, 16.1329, 
    // 16.1175, 16.0663, 15.9624, 15.8754, 15.8620, 15.8004, 15.7327, 15.6686, 15.5794, 
    // 15.5487, 15.5021, 15.4758, 15.4171, 15.3297, 15.2767, 15.2235, 15.1880, 15.1530, 
    // 15.0531, 15.0272, 14.9650, 14.9472, 14.8773, 14.8602, 14.7703, 14.7053, 14.6570, 
    // 14.6051, 14.5290, 14.4815, 14.4594, 14.4097, 14.3850, 14.2873, 14.2663, 14.2534, 
    // 14.1648, 14.1588, 14.0699, 14.0454, 14.0104, 13.9636, 13.9084, 13.8496, 13.8221, 
    // 13.7818, 13.7380, 13.6942, 13.6681, 13.5988, 13.5555, 13.5254, 13.4942, 13.3893, 
    // 13.3848, 13.3380, 13.3098, 13.2649, 13.2178, 13.1930, 13.1572, 13.1125, 13.0828, 
    // 13.0072, 12.9543, 12.9229, 12.8864, 12.8120, 12.7926, 12.7557, 12.7441, 12.6484, 
    // 12.5935, 12.5677, 12.5334, 12.5194, 12.4726, 12.4367, 12.3963, 12.3684, 12.3433, 
    // 12.3233, 12.2870, 12.2449, 12.2060, 12.1253, 12.1218, 12.0719, 12.0461, 12.0198, 
    // 11.9649, 11.9084, 11.8904, 11.8651, 11.8456, 11.7883, 11.7697, 11.7509, 11.7302, 
    // 11.6800, 11.6368, 11.6025, 11.5484, 11.5231, 11.4895, 11.4662, 11.4354, 11.3910, 
    // 11.3816, 11.3424, 11.3040, 11.2942, 11.2270, 11.2204, 11.1910, 11.1858, 11.1155, 
    // 11.1124, 11.0856, 11.0541, 11.0090, 10.9940, 10.9765, 10.9331, 10.9219, 10.8841, 
    // 10.8687, 10.7876, 10.7808, 10.7670, 10.7494, 10.7174, 10.7090, 10.6882, 10.6297, 
    // 10.5898, 10.5798, 10.5689, 10.5218, 10.4954, 10.4537, 10.4245, 10.4128, 10.3522, 
    // 10.3422, 10.3259, 10.3025, 10.2797, 10.2476, 10.2226, 10.1899, 10.1705, 10.1447, 
    // 10.1308, 10.0866, 10.0844, 10.0336, 9.9918, 9.9802, 9.9611, 9.9342, 9.9152, 9.9033, 
    // 9.8667, 9.8237, 9.8001, 9.7710, 9.7615, 9.7412, 9.7078, 9.6874, 9.6684, 9.6358, 
    // 9.6184, 9.5952, 9.5701, 9.5559, 9.5453, 9.5200, 9.4750, 9.4384, 9.4246, 9.3919, 
    // 9.3742, 9.3567, 9.3421, 9.3122, 9.2905, 9.2716, 9.2426, 9.2327, 9.1785, 9.1670, 
    // 9.1502, 9.1335, 9.1255, 9.0830, 9.0635, 9.0151, 8.9892, 8.9823, 8.9443, 8.9420, 
    // 8.8915, 8.8620, 8.8420, 8.8322, 8.8281, 8.7788, 8.7589, 8.7457, 8.6931, 8.6685, 
    // 8.6583, 8.6407, 8.6180, 8.5909, 8.5788, 8.5560, 8.5457, 8.5119, 8.4812, 8.4760, 
    // 8.4456, 8.4415, 8.4140, 8.3912, 8.3666, 8.3512, 8.3284, 8.3007, 8.2714, 8.2635, 
    // 8.2585, 8.2461, 8.2028, 8.1964, 8.1723, 8.1671, 8.1286, 8.1039, 8.0856, 8.0754, 
    // 8.0492, 8.0396, 8.0238, 8.0101, 7.9802, 7.9732, 7.9493, 7.9317, 7.8932, 7.8603, 
    // 7.8513, 7.8186, 7.7837, 7.7690, 7.7474, 7.7156, 7.7017, 7.6975, 7.6667, 7.6457, 
    // 7.6388, 7.6168, 7.5841, 7.5659, 7.5331, 7.5158, 7.4964, 7.4849, 7.4642, 7.4428, 
    // 7.4298, 7.4260, 7.4069, 7.3801, 7.3613, 7.3507, 7.3106, 7.2965, 7.2825, 7.2784, 
    // 7.2456, 7.2028, 7.1860, 7.1653, 7.1536, 7.1407, 7.1247, 7.0983, 7.0856, 7.0777, 
    // 7.0672, 7.0444, 7.0040, 6.9813, 6.9570, 6.9314, 6.9190, 6.9069, 6.8988, 6.8783, 
    // 6.8345, 6.8136, 6.7922, 6.7824, 6.7612, 6.7565, 6.7429, 6.7168, 6.7023, 6.6878, 
    // 6.6689, 6.6550, 6.6148, 6.6052, 6.5985, 6.5886, 6.5438, 6.5279, 6.5119, 6.4905, 
    // 6.4741, 6.4608, 6.4560, 6.4453, 6.4365, 6.4043, 6.3765, 6.3442, 6.3325, 6.3114, 
    // 6.2959, 6.2793, 6.2732, 6.2608, 6.2404, 6.2289, 6.1949, 6.1846, 6.1664, 6.1543, 
    // 6.1365, 6.1149, 6.1024, 6.0690, 6.0461, 6.0333, 6.0209, 6.0015, 5.9932, 5.9810, 
    // 5.9577, 5.9254, 5.9132, 5.8870, 5.8758, 5.8663, 5.8452, 5.8290, 5.7966, 5.7753, 
    // 5.7645, 5.7466, 5.7244, 5.7017, 5.6975, 5.6756, 5.6548, 5.6421, 5.6329, 5.6293, 
    // 5.5946, 5.5800, 5.5619, 5.5426, 5.5096, 5.4875, 5.4765, 5.4577, 5.4421, 5.4326, 
    // 5.4107, 5.3930, 5.3701, 5.3549, 5.3269, 5.3191, 5.3046, 5.3021, 5.2904, 5.2618, 
    // 5.2524, 5.2306, 5.2113, 5.1906, 5.1794, 5.1547, 5.1404, 5.1111, 5.0947, 5.0813, 
    // 5.0729, 5.0549, 5.0437, 5.0192, 5.0124, 4.9950, 4.9829, 4.9582, 4.9333, 4.9188, 
    // 4.8656, 4.8502, 4.8421, 4.8334, 4.8228, 4.8144, 4.7881, 4.7543, 4.7246, 4.7166, 
    // 4.6925, 4.6663, 4.6620, 4.6425, 4.6059, 4.5923, 4.5685, 4.5571, 4.5521, 4.5353, 
    // 4.5196, 4.4913, 4.4758, 4.4644, 4.4484, 4.4081, 4.4045, 4.3799, 4.3717, 4.3236, 
    // 4.3019, 4.2864, 4.2584, 4.2365, 4.2188, 4.1930, 4.1773, 4.1686, 4.1363, 4.1095, 
    // 4.0914, 4.0759, 4.0573, 4.0319, 3.9868, 3.9738, 3.9393, 3.9347, 3.9103, 3.8895, 
    // 3.8725, 3.8463, 3.8212, 3.7732, 3.7559, 3.7542, 3.7084, 3.6943, 3.6681, 3.6499, 
    // 3.6326, 3.5844, 3.5718, 3.5676, 3.5334, 3.4970, 3.4577, 3.3847, 3.3451, 3.2999, 
    // 3.2587, 3.2080, 3.1090, 2.8373, 1.7443

    // >> tdm_s = sparse(tdm);
    // >> nnz(tdm_s)
    // 
    // ans =
    // 
    //        34833
    // 
    // >> [Us,Ss,Vs] = svds(tdm_s,5);
    // >> diag(Ss)
    // 
    // ans =
    // 
    //   120.2292, 55.7848, 49.3458, 43.6356, 40.3711
    // 
    // >> [Us,Ss,Vs] = svds(tdm_s,10);
    // >> diag(Ss)
    // 
    // ans =
    // 
    //   120.2292, 55.7848, 49.3458, 43.6356, 40.3711, 33.7784, 33.3014, 31.9810, 31.0635, 30.3060
    // 
    // >> [Us,Ss,Vs] = svds(tdm_s,100);
    // >> diag(Ss)
    // 
    // ans =
    // 
    //   120.2292, 55.7848, 49.3458, 43.6356, 40.3711, 33.7784, 33.3014, 31.9810, 31.0635, 30.3060, 29.0700, 28.5012, 28.0047, 27.5122, 27.0972, 26.8200, 26.2419, 25.8579, 25.7415, 25.0880, 24.9734, 24.6606, 24.4398, 24.1955, 23.8598, 23.7656, 23.4163, 23.3179, 23.1779, 23.0825, 22.6901, 22.6844, 22.3345, 22.2170, 22.0876, 21.8041, 21.7062, 21.6118, 21.4724, 21.0920, 21.0333, 20.9851, 20.9017, 20.7311, 20.5557, 20.3701, 20.2316, 20.0848, 20.0000, 19.8401, 19.7174, 19.5898, 19.5232, 19.4532, 19.3561, 19.1832, 19.1378, 19.0323, 18.9137, 18.8020, 18.7371, 18.4977, 18.3716, 18.3004, 18.2333, 18.1608, 18.0775, 17.9601, 17.8811, 17.7346, 17.6749, 17.6081, 17.5227, 17.4908, 17.4475, 17.2730, 17.2121, 17.1433, 17.0005, 16.9598, 16.8710, 16.8006, 16.7648, 16.7112, 16.5711, 16.5388, 16.4580, 16.3859, 16.2340, 16.1329, 16.1175, 16.0663, 15.9624, 15.8754, 15.8620, 15.8004, 15.7327, 15.6686, 15.5794, 15.5487
    
    return EXIT_SUCCESS;
}

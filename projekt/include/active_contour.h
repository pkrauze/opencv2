#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//C
#include <stdio.h>
#include <tuple>
//C++

using namespace cv;
using namespace std;

Mat neumann_boundary_condition(const Mat &in);
Mat curvature_central(Mat u);
pair<Mat, Mat> exchange(Mat f1, Mat f2, int isExchange);
tuple<Mat_<double>, Mat_<double>, Mat_<double>> ACM(Mat_<double> u, Mat_<double> Img,
                                                    Mat_<double> Ksigma, Mat_<double> KI, Mat_<double> KI2, Mat_<double> KONE,
                                                    double nu, double timestep, double mu, double epsilon, double lambda1,
                                                    double lambda2, int CV, int RSF, int LRCV, int LIF, int LGD, int isExchange);

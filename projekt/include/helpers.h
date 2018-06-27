#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//C
#include <stdio.h>
#include <iostream>
#include <fstream>
//C++

using namespace cv;
using namespace std;

Mat get_initial_contour(Size size, string file);
void write_contours_to_file(vector<vector<Point> > contours, string filename);

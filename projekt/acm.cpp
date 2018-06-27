//opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//C
#include <stdio.h>
//C++
#include <iostream>
#include <fstream>
#include <cmath>
#include <limits>

using namespace cv;
using namespace std;

#include "active_contour.h"
#include "helpers.h"

int main(int argc, char** argv) {
  string options =
    "{help | | prints help }"
    "{image | img/2.bmp | image file }"
    "{contour | <none> | initial contour type }"
    "{save-images | <none> | dir for saving images }"
    "{save-final-contours | <none> | file with final contours }"
    "{save-final-image | <none> | file with final image }"
    "{mu | 1.0 | param}"
    "{nu | 65.025 | param}"
    "{lambda1 | 1.0 | param}"
    "{lambda2 | 1.0 | param}"
    "{epsilon | 1.0 | param}"
    "{timestep | 0.05 | param}"
    "{iter | 200 | param}"
    "{sigma | 3.0 | param}"
    "{RSF | 1 | param}"
    "{LRCV | 0 | param}"
    "{LIF | 0 | param}"
    "{LGD | 0 | param}"
    "{CV | 0 | param}"
    "{exchange | 1 | param}"
    ;
  CommandLineParser parser(argc, argv, options);

  if(parser.has("help")) {
    parser.printMessage();
    return 0;
  }

  string image_file = parser.get<String>("image");
  Mat Img = imread(image_file, IMREAD_GRAYSCALE);

  Mat_<double> initial_lsf;
  if(parser.has("contour")) {
    string contour = "contours/" + parser.get<String>("contour");
    initial_lsf = get_initial_contour(Img.size(), contour);
  } else {
    initial_lsf = Mat::ones(Img.rows, Img.cols, CV_64F) * 1;
    Rect rec(Point(40, 15), Size(20, 20));
    initial_lsf(rec).setTo(-1.0);
  }
  Mat_<double> u = initial_lsf;

  double mu = parser.get<double>("mu");
  double nu = parser.get<double>("nu");
  double lambda1 = parser.get<double>("lambda1");
  double lambda2 = parser.get<double>("lambda2");
  double epsilon = parser.get<double>("epsilon");
  double timestep = parser.get<double>("timestep");
  double iterNum = parser.get<double>("iter");
  double sigma = parser.get<double>("sigma");

  Mat_<double> gauss_kernel_1d = getGaussianKernel(round(2*sigma)*2+1, sigma, CV_64FC1);
  Mat_<double> gauss_kernel_2d = gauss_kernel_1d * gauss_kernel_1d.t();
  Mat_<double> Ksigma = gauss_kernel_2d;

  Mat_<double> KONE;
  filter2D((Mat::ones(Img.size(), CV_64FC1)), KONE, CV_64FC1, Ksigma);

  Mat_<double> KI;
  filter2D(Img, KI, CV_64FC1, Ksigma, Point(-1,-1), 0, BORDER_REPLICATE);

  Mat_<double> KI2;
  filter2D(Img.mul(Img), KI2, CV_64FC1, Ksigma, Point(-1, -1), 0, BORDER_REPLICATE);

  int RSF = parser.get<int>("RSF");
  int LRCV = parser.get<int>("LRCV");
  int LIF = parser.get<int>("LIF");
  int LGD = parser.get<int>("LGD");
  int CV = parser.get<int>("CV");

  // isExchange = 1; % '1' for bright object and dark backgroud;
  //                 % '-1' for dark object and bright backgroud;
  //                 % '0' represent original model.
  int isExchange = parser.get<int>("exchange");

  Mat_<double> f1, f2, img_copy;
  vector<vector<Point> > final_contours;
  Mat final_image;
  Img.convertTo(img_copy, CV_64FC1);

  for(int n=0; n<iterNum; n++) {
    auto acm_result = ACM(u,img_copy,Ksigma,KI,KI2,KONE,nu,timestep,mu,epsilon,
                          lambda1,lambda2,CV,RSF,LRCV,LIF,LGD,isExchange);
    u = get<0>(acm_result);
    f1 = get<1>(acm_result);
    f2 = get<2>(acm_result);

    if(n%5 == 0) {
      Mat new_image = Img.clone();
      Mat contour_mat;

      double limit = 0.0;
      threshold(u, contour_mat, limit, 1, THRESH_BINARY);

      contour_mat.convertTo(contour_mat, CV_8UC1, 255.0);

      // swap negative/positive
      contour_mat = contour_mat * -1 + 255;

      vector<vector<Point> > contours;
      findContours(contour_mat, contours, RETR_LIST, CHAIN_APPROX_NONE);

      cvtColor(new_image, new_image, CV_GRAY2BGR);
      drawContours(new_image, contours, -1, Scalar(0, 0, 255));
      resize(new_image, new_image, Size(400,400));
      putText(new_image, "Iter: "+to_string(n), Point(20,20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

      if(parser.has("save-images")) {
        string dir = parser.get<String>("save-images");
        char iter_num[4];
        if(n==0) {
          string folderCreateCommand = "mkdir " + dir;
          system(folderCreateCommand.c_str());
        }
        sprintf(iter_num, "%04d", n);
        string filename = dir + "/" + "iter_" + iter_num + ".jpg";
        imwrite(filename, new_image);
      }

      if(parser.has("save-final-contours"))
        final_contours = contours;

      if(parser.has("save-final-image"))
        final_image = new_image;

      imshow("window", new_image);

      int wait_time = 200;
      waitKey(wait_time);
    }
  }

  if(parser.has("save-final-contours")) {
    string final_contours_filename = parser.get<String>("save-final-contours");
    if(final_contours.size() > 0)
      write_contours_to_file(final_contours, final_contours_filename);
  }

  if(parser.has("save-final-image")) {
    string final_image_filename = parser.get<String>("save-final-image");
    imwrite(final_image_filename, final_image);
  }

  while(true)
    if (waitKey(1) == 27)
      break;

  return 0;
}

#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <ctime>
using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
  VideoCapture cap(0);

  int low_h = 34;
  int high_h = 82;
  int low_s = 65;
  int high_s = 253;
  int low_v = 75;
  int high_v = 154;
  int last_x = -1;
  int last_y = -1;

  Mat imgTmp;
  cap.read(imgTmp);

  namedWindow("Control");
  createTrackbar("LowH", "Control", &low_h, 179);
  createTrackbar("HighH", "Control", &high_h, 179);
  createTrackbar("LowS", "Control", &low_s, 255);
  createTrackbar("HighS", "Control", &high_s, 255);
  createTrackbar("LowV", "Control", &low_v, 255);
  createTrackbar("HighV", "Control", &high_v, 255);

  Mat imgLines = Mat::zeros( imgTmp.size(), CV_8UC3 );;

  while (true)
    {
      Mat img_original;

      cap.read(img_original); // read a new frame from video

      Mat imgHSV;

      cvtColor(img_original, imgHSV, COLOR_BGR2HSV);

      Mat img_thresholded;

      inRange(imgHSV, Scalar(low_h, low_s, low_v), Scalar(high_h, high_s, high_v), img_thresholded);
      erode(img_thresholded, img_thresholded, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)) );
      dilate( img_thresholded, img_thresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
      dilate( img_thresholded, img_thresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
      erode(img_thresholded, img_thresholded, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)) );

      Moments oMoments = moments(img_thresholded);
      double dM01 = oMoments.m01;
      double dM10 = oMoments.m10;
      double area = oMoments.m00;

      int pos_x = dM10 / area;
      int pos_y = dM01 / area;

      if (last_x >= 0 && last_y >= 0 && pos_x >= 0 && pos_y >= 0)
        line(imgLines, Point(pos_x, pos_y), Point(last_x, last_y), Scalar(0,0,255), 1);

      last_x = pos_x;
      last_y = pos_y;
      imshow("Thresholded Image", img_thresholded);

      if (waitKey(50) == 27)
        break;
    }

  cvDestroyWindow("Thresholded Image");

  return 0;
}

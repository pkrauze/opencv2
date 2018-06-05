#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <ctime>
using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
  VideoCapture cap(0);

  int iLastX = -1;
  int iLastY = -1;
  clock_t t;
  int i=2;

  Mat img_tmp;
  cap.read(img_tmp);
  Mat imgLines = Mat::zeros( img_tmp.size(), CV_8UC3 );;

  while (true)
    {
      Mat img_original;
      cap.read(img_original);

      Mat img_HSV;
      cvtColor(img_original, img_HSV, COLOR_BGR2HSV);

      Mat img_new;
      int high_h_arr[3] = {41, 82, 147};
      int high_s_arr[3] = {226, 253, 318};
      int high_v_arr[3] = {219, 154, 195};
      int low_h_arr[3] = {0, 34, 89};
      int low_s_arr[3] = {116, 65, 144};
      int low_v_arr[3] = {171, 75, 45};

      t = clock();

      inRange(img_HSV,
              Scalar(low_h_arr[i], low_s_arr[i], low_v_arr[i]),
              Scalar(high_h_arr[i], high_s_arr[i], high_v_arr[i]),
              img_new);

      erode(img_new, img_new, Mat(), Point(-1, -1), 2);
      dilate( img_new, img_new, Mat(), Point(-1,-1), 2);
      erode(img_new, img_new, Mat(), Point(-1, -1), 2);
      erode(img_new, img_new, Mat(), Point(-1, -1), 2);
      dilate( img_new, img_new, Mat(), Point(-1,-1), 2);
      dilate( img_new, img_new, Mat(), Point(-1,-1), 2);

      if(t/CLOCKS_PER_SEC%5 == 0)
      {
        putText(img_original, "ZMIANA!", Point(70,70), FONT_HERSHEY_PLAIN, 2,  Scalar(0,0,255,255));
        i++;
        if (i>3)
          i = 0;
      }
      
      vector<vector<Point> > contours;
      findContours(img_new, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

      drawContours(img_HSV, contours, 0, Scalar(100,100,100));

      if(contours.size() >= 1)
        {
      Moments oMoments = moments(contours[0]);
      double dM01 = oMoments.m01;
      double dM10 = oMoments.m10;
      double area = oMoments.m00;

      int pos_x = dM10 / area;
      int pos_y = dM01 / area;

      if (iLastX >= 0 && iLastY >= 0 && pos_x >= 0 && pos_y >= 0)
        line(imgLines, Point(pos_x, pos_y), Point(iLastX, iLastY), Scalar(0,0,255), 1);

      iLastX = pos_x;
      iLastY = pos_y;

      circle(img_original,Point(pos_x,pos_y), 63, Scalar(0,0,255), 3);
      line(img_original, Point(pos_x-75,pos_y), Point(pos_x-10, pos_y), Scalar(0,0,255), 3);
      line(img_original, Point(pos_x+10,pos_y), Point(pos_x+75, pos_y), Scalar(0,0,255), 3);
      line(img_original, Point(pos_x,pos_y-75), Point(pos_x, pos_y-10), Scalar(0,0,255), 3);
      line(img_original, Point(pos_x,pos_y+10), Point(pos_x, pos_y+75), Scalar(0,0,255), 3);
      string str = to_string(pos_x).append(", ").append(to_string(pos_y));
      putText(img_original, str, Point(pos_x-70,pos_y+105), FONT_HERSHEY_PLAIN, 2,  Scalar(0,0,255,255));
        }
      imshow("Original", img_original);
      imshow("in range", img_new);

      if (waitKey(1) == 27)
        break;
    }

  cvDestroyWindow("Original");

  return 0;
}

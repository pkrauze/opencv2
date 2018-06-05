//opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//C
#include <stdio.h>
//C++
#include <iostream>

using namespace cv;
using namespace std;
Mat original;

int main( int argc, char** argv )
{
  VideoCapture cap(0);
  cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
  cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
  int camOpen = cap.open(CV_CAP_ANY);
  vector<Point2f> points[2];
 

  while (true)
    {
      cap >> original;
      // cvtColor(original, original, COLOR_RGB2GRAY );
      Mat new_gray;
      // original.copyTo(new_gray);
      vector<uchar> status;
      vector<float> err;
      double qualityLevel = 0.01;
      double minDistance = 10;
      int blockSize = 3, gradiantSize = 3;
      bool useHarrisDetector = false;
      double k = 0.04;
      Mat original;
      int maxCorners = 50;
      int maxTrackbar = 100;

      // if(points[0].size() < 10)
        // goodFeaturesToTrack(original, points[0], maxCorners, qualityLevel, minDistance, Mat(), blockSize, gradiantSize, useHarrisDetector, k);

      // calcOpticalFlowPyrLK(new_gray, original, points[0], points[1], status, err, winSize, 3, termcrit, 0, 0.001);

      // for( int i = 0; i < points[0].size(); ++i )
      //   {
      //     line(original, points[0][i], points[1][i], Scalar(255, 0, 0), 10, 1, 0);

      //     if(points[0][i] == points[1][i])
      //       points[0].erase(i);
      //   }

      if(!original.empty())
        imshow("Montion", original);

      if (waitKey(1) == 27)
        break;
    }
  return 0;
}
read(original);

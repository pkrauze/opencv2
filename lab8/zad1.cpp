
//opencv
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
//C
#include <stdio.h>
//C++
#include <iostream>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
  VideoCapture cap(0);
  Ptr<BackgroundSubtractorMOG2> pMOG2 = createBackgroundSubtractorMOG2();;
  namedWindow("Background", 1);
  namedWindow("Montion", 1);
  namedWindow("settings", 1);
  int substractor_history = 10;
  createTrackbar("history", "settings", &substractor_history, 100);
  int substractor_mixtures = 100;
  createTrackbar("mixtures", "settings", &substractor_mixtures, 100);
  int shadows = 0;
  createTrackbar("shadows", "settings", &shadows, 1);
  Mat original, foreground, background;
  
  while (true)
    {
      cap.read(original);

      pMOG2->apply(original, foreground);

      pMOG2->setHistory(substractor_history);
      pMOG2->setNMixtures(substractor_mixtures);
      pMOG2->setDetectShadows(shadows);

      pMOG2->getBackgroundImage(background);

      morphologyEx(foreground, foreground, MORPH_OPEN, Mat::ones(9, 9, CV_8UC1));

      vector<vector<Point> > contours;
      findContours(foreground, contours, RETR_LIST, CHAIN_APPROX_NONE);

      for(int i=0; i<contours.size(); i++)
        drawContours(original, contours, i, Scalar(100, 100, 100), 2);

      imshow("Background", background);
      imshow("Montion", original);

      if (waitKey(1) == 27)
        break;
    }
  return 0;
}


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
  int line_length = 50;
  createTrackbar("line length", "settings", &line_length, 100);
  int line_gap = 10;
  createTrackbar("line gap", "settings", &line_gap, 100);
  int threshold = 50;
  createTrackbar("threshold", "settings", &threshold, 100);
  Mat original, foreground, background, edges, img_lines;
  
  while (true)
    {
      cap.read(original);

      Canny(original, edges, 50, 200, 3);

      vector<Vec4i> lines;
      HoughLinesP(edges, lines, 1, CV_PI/180, threshold, line_length, line_gap);
      for(int i=0; i<lines.size(); i++) {
        Vec4i l = lines[i];
        line(original, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
      }

      imshow("Montion", original);

      if(waitKey(30)>=0)
        break;
    }
  cvDestroyAllWindows();
  cap.release();
  return 0;
}

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

void draw(vector<vector<Point> > contours, Mat frame2) {
  if(contours.size() > 0) {
    vector<Point> max_contour;
    int largest_area=-1;

    Rect bounding_rect;
    for( int i = 0; i< contours.size(); i++ ) // iterate through each contour. 
      {
        double a=contourArea( contours[i]);  //  Find the area of contour
        if(a>largest_area){
          largest_area=a;
          max_contour = contours[i];
        }
      }
    bounding_rect=boundingRect(max_contour);

    Mat hull;
    convexHull(max_contour, hull);

    vector<vector<Point> > drawing = {hull};
    rectangle(frame2, bounding_rect, Scalar(255, 255, 0));
    drawContours(frame2, drawing, -1, Scalar(255, 255, 0));
  }
}

int k;
int option = 1;

int main( int argc, char** argv )
{
  VideoCapture cap(0);
  bool flag = true;
  bool track = false;
  bool track2 = false;
  namedWindow("Montion", 1);
  Mat frame1, frame2;
  Ptr<BackgroundSubtractorMOG2> pMOG2 = createBackgroundSubtractorMOG2();;

  pMOG2->setHistory(30);
  pMOG2->setNMixtures(100);
  pMOG2->setDetectShadows(0);

  cap.read(frame2);
  
  while (true)
    {
      frame1 = frame2.clone();
       cap.read(frame2);

      Mat gray_frame1, gray_frame2;
      cvtColor(frame2, gray_frame2, CV_BGR2GRAY);
      cvtColor(frame1, gray_frame1, CV_BGR2GRAY);
      Mat diff;
      absdiff(gray_frame1, gray_frame2, diff);

      Mat blurr_diff;
      GaussianBlur(diff, blurr_diff, Size(5,5), 20);

      Mat thresh;
      threshold(blurr_diff, thresh, 20.0, 255.0, THRESH_BINARY);

      vector<vector<Point> > contours;
      vector<Vec4i> hierarchy;
      findContours(thresh, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
      Mat foreground;
      Mat gauss = frame2.clone();
 
      k = waitKey(1);
      if(k == 49 || k == 50 || k == 51 || k == 52)
        option = k;
      switch(option)
        {
        case 49:
          flag = true;
          draw(contours, frame2);
          break;
        case 50:
          flag = false;
          draw(contours, frame2);
          break;
        case 51:
          flag = true;
          pMOG2->apply(gauss, foreground);
          pMOG2->getBackgroundImage(gauss);

          morphologyEx(foreground, foreground, MORPH_OPEN, Mat::ones(4, 4, CV_8UC1));
          findContours(foreground, contours, RETR_LIST, CHAIN_APPROX_NONE);
          draw(contours, frame2);
          break;
        case 52:
          if(track)
            draw(contours, frame2);
          break;
        }

      //imshow("Tresh", thresh);
      imshow("Montion", frame2);
      if(waitKey(1) == 27)
        break;
    }
  return 0;
}

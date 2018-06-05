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

bool montionDetector(Mat frame1, Mat frame2) {
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
  findContours(thresh, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_NONE);

  if(contours.size() >= 20) {
    return true;
  }
  return false;
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
  chrono::steady_clock sc;   // create an object of `steady_clock` class
  auto start = sc.now();
  auto end = sc.now();
  auto time_span = static_cast<chrono::duration<double> >(end - start);

  cap.read(frame2);

  char data_label_temp[50];

  double width = cap.get(CAP_PROP_FRAME_WIDTH);
  double height = cap.get(CAP_PROP_FRAME_HEIGHT);
  double fps = cap.get(CAP_PROP_FPS);

  cout << "resolution: " << width << " x " << height << endl;
  cout << "fps: " << fps << endl;

  bool pause = true;
  bool newFile = true;
  while (true)
    {
      frame1 = frame2.clone();
      if(pause)
        cap.read(frame2);
      String moveText = "MOVEMENT DETECTED";
      Point textPoint = Point2i(30, frame2.rows-30);
      Point textPoint2 = Point2i(30, 30);
      time_t t = time(0);
      tm* now = localtime(&t);
      sprintf(data_label_temp, "%d-%02d-%02d-%02d-%02d-%02d", (now->tm_year + 1900), (now->tm_mon + 1), (now->tm_mday), (now->tm_hour), (now->tm_min), (now->tm_sec));
      String label(data_label_temp);
      putText(frame2, label, textPoint, FONT_HERSHEY_PLAIN, 1, Scalar(0,200,0), 2);
      Size frame_size(width, height);
      String fileName;

      if(newFile && label != fileName)
        fileName = label + ".avi";
      newFile = false;

      VideoWriter oVideoWriter(fileName, -1, fps, frame_size, true);
      cout << fileName << endl;

      k = waitKey(1);
      if(k == 49 || k == 50 || k == 51)
        option = k;

      switch(option)
        {
        case 49:
          pause = true;
          if(montionDetector(frame1, frame2)) {
            start = sc.now();     // start timer
          }
          end = sc.now();
          time_span = static_cast<chrono::duration<double> >(end - start);
          if(time_span.count() < 10)
            oVideoWriter.write(frame2);
          putText(frame2, "RECORDING", textPoint2, FONT_HERSHEY_PLAIN, 1, Scalar(0,200,0), 2);
          break;
        case 50:
          pause = false;
          putText(frame2, "PAUSED", textPoint2, FONT_HERSHEY_PLAIN, 1, Scalar(0,200,0), 2);
          break;
        case 51:
          pause = true;
          newFile = true;
          option = 1;
          putText(frame2, "NEW FILE", textPoint2, FONT_HERSHEY_PLAIN, 1, Scalar(0,200,0), 2);
          break;
        }

      //imshow("Tresh", thresh);
      imshow("Montion", frame2);
      if(waitKey(1) == 27)
        break;
    }
  cap.release();
  cvDestroyAllWindows();
  return 0;
}

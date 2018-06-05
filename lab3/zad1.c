#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main() {
  VideoCapture stream("robot_no_loop.avi");

  if (!stream.isOpened()) {
    cout << "cannot open camera";
  }

  //unconditional loop
  Mat firstFrame, secondFrame, result;
  Mat firstFrame2, secondFrame2, result2;

  stream.read(firstFrame2);
  cvtColor(firstFrame2, firstFrame2, CV_RGB2GRAY);

  while (true) {
    stream >> firstFrame;
    stream >> secondFrame;

    imshow("cam1", secondFrame);

    stream.read(secondFrame2);
    cvtColor(secondFrame2, secondFrame2, CV_RGB2GRAY);



    cvtColor(firstFrame, firstFrame, CV_RGB2GRAY);
    cvtColor(secondFrame, secondFrame, CV_RGB2GRAY);
    absdiff(firstFrame, secondFrame, result);
    absdiff(firstFrame2, secondFrame2, result2);
    threshold(result, result, 100, 255, THRESH_BINARY);
    threshold(result2, result2, 100, 255, THRESH_BINARY);

    imshow("cam2", result);
    imshow("cam3", result2);

    if (waitKey(30) == 27)
      break;
  }
  return 0;
}

#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main() {
  VideoCapture stream1(0);

  if (!stream1.isOpened()) {
    cout << "cannot open camera";
  }

  //unconditional loop
  while (true) {
    Mat cameraFrame, converted;

    stream1.read(cameraFrame);
    cvtColor(cameraFrame, converted, CV_RGB2GRAY);
    bitwise_not(converted, converted);

    imshow("cam", cameraFrame);
    imshow("cam2", converted);

    if (waitKey(30) == 27)
      break;
  }
  return 0;
}

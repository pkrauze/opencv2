#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <ctime>
using namespace cv;
using namespace std;

int main() {
  VideoCapture cap(0);
  VideoCapture cap2("bike.avi");

  if (!cap.isOpened()) {
    cout << "cannot open camera";
  }

  int w = cap.get(CV_CAP_PROP_FRAME_WIDTH) +1;
  int h = cap.get(CV_CAP_PROP_FRAME_HEIGHT) +1;

  VideoWriter video("out.mp4", CV_FOURCC('m','p','4','v'), 30, Size(w,h), true);

  Mat frame, secondFrame;
  clock_t t;
  t = clock();
  char buf[100];
  int i = 0;
  while (true) {
    cap >> frame;
    cap2 >> secondFrame;
    i++;

    video.write(frame);
    video.write(secondFrame);

    imshow( "Frame", frame );

    if (waitKey(300) == 27)
      break;
  }
  return 0;
}

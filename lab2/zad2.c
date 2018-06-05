#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <ctime>
using namespace cv;
using namespace std;
using namespace this_thread; // sleep_for, sleep_until
using namespace chrono;

int main() {
  VideoCapture cap(0);

  if (!cap.isOpened()) {
    cout << "cannot open camera";
  }

  int w = cap.get(CV_CAP_PROP_FRAME_WIDTH) +1;
  int h = cap.get(CV_CAP_PROP_FRAME_HEIGHT) +1;

  //VideoWriter writer("out.mp4", CV_FOURCC('m','p','4','v'), fps, Size(w,h), true);

  Mat frame;
  clock_t t;
  t = clock();
  char buf[100];
  int i = 0;
  while (true) {
    cap >> frame;
    i++;
    sprintf(buf, "frames/%03d.jpg", i); // Make sure directory 'frames' exists.
    imwrite(buf, frame);
    imshow( "Frame", frame );

    if (waitKey(30) == 27)
      break;
  }
  return 0;
}

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main() {
  VideoCapture stream(0);
  Mat firstFrame, secondFrame, detectorFrame, thirdFrame, result;

  if (!stream.isOpened()) {
    cout << "cannot open file";
  }

  stream.read(firstFrame);
  secondFrame = firstFrame.clone();
  cvtColor(firstFrame, firstFrame, CV_BGR2GRAY);
  cvtColor(secondFrame, secondFrame, CV_BGR2GRAY);

  firstFrame.copyTo(detectorFrame);
  detectorFrame = detectorFrame - detectorFrame;

  int alpha_slider = 1;
  int threshold_slider = 1;
  namedWindow("settings", 1);
  createTrackbar("Alpha", "settings", &alpha_slider, 100);
  createTrackbar("threshold", "settings", &threshold_slider, 255);

  while (true) {
    stream.read(secondFrame);
    stream.read(thirdFrame);

    cvtColor(secondFrame, secondFrame, CV_RGB2GRAY);
    cvtColor(thirdFrame, thirdFrame, CV_RGB2GRAY);
    absdiff(secondFrame, thirdFrame, result);
    threshold(result, result, threshold_slider, 255, THRESH_BINARY);

    double alpha = (double) alpha_slider / 100;
    detectorFrame = (1-alpha) * detectorFrame + alpha * result;

    imshow("cam", detectorFrame);

    if (waitKey(200) == 27)
      break;
  }

  cvDestroyWindow("settings");
  cvDestroyWindow("cam");
  firstFrame.release();
  secondFrame.release();
  thirdFrame.release();
  detectorFrame.release();
  result.release();

  return 0;
}

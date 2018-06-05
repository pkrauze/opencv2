#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <ctime>
using namespace cv;
using namespace std;

class TrackColor {
private:
  vector<Point> contour;
  Point center;

  int h, s, v;

  string color;

  // Scalar low_yellow(0, 116, 171);
  // Scalar high_yellow(41, 226, 219);

  // Scalar low_green(34, 65, 75);
  // Scalar high_green(82, 253, 154);

  // Scalar low_blue(89, 144, 45);
  // Scalar high_blue(147, 318, 195);

public:
  TrackColor(vector<Point> contour) {
    this->contour = contour;
  }

  void setColor(string color) {
    this->color = color;
  }

  Point getCenter() {
    return this->center;
  }

  void setCenter(Point center) {
    this->center = center;
  }

  void setH(int value) {
    this->h = value;
  }

  void setS(int value) {
    this->s = value;
  }

  void setV(int value) {
    this->v = value;
  }

  int getH() {
    return this->h;
  }

  int getS() {
    return this->s;
  }

  int getV() {
    return this->v;
  }
};

#include "helpers.h"

Mat get_initial_contour(Size size, string file) {
    ifstream in(file, ifstream::in);

    string contour_type;
    in >> contour_type;

    Mat_<double> initial_lsf = Mat::ones(size, CV_64FC1);

    if(contour_type == "rectangle") {
      Point top_left, bot_right;
      in >> top_left.x >> top_left.y >> bot_right.x >> bot_right.y;
      initial_lsf(Rect(top_left, bot_right)).setTo(-1.0);
    }
    else if(contour_type == "ellipse") {
      Point center;
      Size size;
      in >> center.x >> center.y;
      in >> size.height >> size.width;

      ellipse(initial_lsf, center, size, 0, 0, 360, Scalar(-1.0), CV_FILLED);
    }
    else if(contour_type == "circle") {
      Point center;
      int radius;
      in >> center.x >> center.y;
      in >> radius;
      circle(initial_lsf, center, radius, Scalar(-1.0), CV_FILLED);
    }
    else if(contour_type == "contour") {
      vector<Point> contour;
      int x,y;
      while(in >> x >> y) {
        Point point(x,y);
        contour.push_back(point);
      }

      vector<vector<Point> > contours;
      contours.push_back(contour);

      drawContours(initial_lsf, contours, -1, Scalar(-1.0), CV_FILLED);
    }
    else {
      cout << "wrong contour type" << endl;
      exit(1);
    }

    return initial_lsf;
}

void write_contours_to_file(vector<vector<Point> > contours, string filename) {
  ofstream fout(filename);
  int i;
  for(i=0; i<contours.size(); i++) {
    vector<Point> contour = contours[i];
    fout << "contour " << i+1 << endl;

    for(Point p : contour) {
      fout << p.x << " " << p.y << endl;
    }
  }
}

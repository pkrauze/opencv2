#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main() {
  VideoCapture stream(0);
  Mat src, src_gray, gauss, sobel_x, sobel_y, detected_edges, dst;
  Mat countours, gradient, dst_phase;
  Mat new_gradient, new_angle, gradient_binary;
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;

  if (!stream.isOpened()) {
    cout << "cannot open file";
  }
  stream.read(src);
  // int alpha_slider = 1;
  int threshold_slider = 1;
  namedWindow("settings", 1);
  // createTrackbar("Alpha", "settings", &alpha_slider, 100);
  createTrackbar("threshold", "settings", &threshold_slider, 100);

  while (true) {
    stream.read(src);

    Mat sobel_x_bin, sobel_y_bin;
    cvtColor(src, src_gray, CV_BGR2GRAY );
    GaussianBlur( src_gray, gauss, Size(3,3), 2.0);
    Sobel(gauss, sobel_x, -1, 1.0, 0.0, 3);
    threshold(sobel_x, sobel_x_bin, 20, 250, THRESH_BINARY);
    Sobel(gauss, sobel_y, -1, 0.0, 1.0, 3);
    threshold(sobel_y, sobel_y_bin, 20, 250, THRESH_BINARY);

    Canny( gauss, detected_edges, 60, 100, 3 );

    sobel_x.convertTo(sobel_x, CV_32F, 1.0/255);
    sobel_y.convertTo(sobel_y, CV_32F, 1.0/255);
    cartToPolar(sobel_x, sobel_y, new_gradient, new_angle, true);

    threshold(new_gradient, gradient_binary, 20, 250, THRESH_BINARY);

    Mat gradient_colored = gradient_binary.clone();
    cvtColor(gradient_colored, gradient_colored, CV_GRAY2BGR);
    gradient_colored.convertTo(gradient_colored, CV_8UC3, 255);
    float edge;

    for( int i = 0; i< gradient_binary.rows; i++ )
    {
      for( int j=0; j< gradient_binary.cols; j++)
      {
        edge = gradient_colored.at<float>(i,j);
        Vec3b color;

        if (edge < 0 ) {
          float angle_value = new_angle.at<float>(i,j);
          if(angle_value>45 && angle_value<=135)
            color = Vec3b( 255, 255, 255);
          if(angle_value>135 && angle_value<=255)
            color = Vec3b( 255, 0, 0);
          if(angle_value>255 && angle_value<=315)
            color = Vec3b(0,255,0);
          if((angle_value>315 && angle_value<=360) || (angle_value>0 && angle_value<=45))
            color = Vec3b(0,0,255);
        } else {
          color = Vec3b(0,0,0);
        }

        gradient_colored.at<Vec3b>(i,j) = color;
      }
    }

    imshow("gauss", gauss);
    imshow("sobel x", sobel_x);
    imshow("sobel y", sobel_y);
    imshow("canny", detected_edges);
    imshow("gradient", gradient_binary);
    imshow( "Contours", gradient_colored);

    if (waitKey(200) == 27)
      break;
  }

  cvDestroyWindow("settings");
  cvDestroyWindow("cam");

  return 0;
}

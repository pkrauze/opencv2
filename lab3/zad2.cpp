#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include<vector>

using namespace std;
using namespace cv;

int main(){
    //VideoCapture cap("robot_no_loop.avi");
    VideoCapture cap(0);

    if(!cap.isOpened()){
        cerr << "Error opening the webcam!" << endl;
        return -1;
    }

    int size = 70;

    Mat image = imread("logo.bmp",0);
    resize(image,image,Size(size,size));
    Mat frame;
    Mat newFrame;

    while(true){
      cap>>newFrame;

      cvtColor(newFrame, newFrame, CV_BGR2GRAY);
      Rect react(newFrame.cols - (size + 50), newFrame.rows - (size + 50), size, size);
      newFrame(react) = image + newFrame(react);
      imshow("frame", newFrame);

      if (waitKey(30) == 27)
        break;
    }

    cvDestroyWindow("frame");
    image.release();
    newFrame.release();

    return 0;
}

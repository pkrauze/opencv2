//opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/objdetect/objdetect.hpp"
//C
#include <stdio.h>
//C++
#include <iostream>

using namespace cv;
using namespace std;


int main( int argc, char** argv )
{
  VideoCapture cap(0);

  std::vector<Rect> faces;
  Mat frame_gray;
  String face_cascade_name = "haarcascade_frontalface_alt.xml";
  String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
  CascadeClassifier face_cascade;
  CascadeClassifier eyes_cascade;
  namedWindow("settings", 1);
  int scaleFactor = 3;
  createTrackbar("scaleFactor", "settings", &scaleFactor, 100);
  int minNeighbors = 3;
  createTrackbar("minNeighbors", "settings", &minNeighbors, 100);
  int minSize = 10;
  createTrackbar("minSize", "settings", &minSize, 100);
  Mat frame;
  int k;

  while (true)
    {
      cap >> frame;

      if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
      if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

      cvtColor( frame, frame_gray, CV_BGR2GRAY );
      equalizeHist( frame_gray, frame_gray );

      //-- Detect faces
      face_cascade.detectMultiScale( frame_gray, faces, 1.1, 3, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
      Rect first_face_area;
      Mat first_face;

      for( size_t i = 0; i < faces.size(); i++ )
        {
          Rect face_area = faces[i];

          Mat faceROI = frame_gray( faces[i] );
          std::vector<Rect> eyes;

          //-- In each face, detect eyes
          eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

          Point left_side(face_area.x, face_area.y + face_area.height/2);
          Rect eyes_area(left_side - Point(0, face_area.height/3), Size(face_area.width, face_area.height/3));

          //option = COPY;
          switch(k) {
          case 49:
            GaussianBlur(frame(face_area), frame(face_area), Size(15, 15), 20);
            break;
          case 50:
            rectangle(frame, eyes_area, Scalar(0,0,0), -1);
            break;
          case 51:
            if(i==0) {
              first_face_area = face_area;
              first_face = frame(first_face_area).clone();
            }
            else {
              resize(first_face, first_face, face_area.size());
              first_face.copyTo(frame(face_area));
            }
            break;
          case 52:
            break;
          case 27:
            return 0;
          };
        }
      //-- Show what you got
      if(!frame.empty())
        imshow("Montion", frame);

      k = waitKey(1);

      if(waitKey(1) == 27)
        break;
    }
  return 0;
}

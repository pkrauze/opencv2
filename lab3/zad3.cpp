#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include<math.h>
using namespace std;
using namespace cv;

Mat img, img1;

int main()
{
   img=imread("logo.bmp",CV_LOAD_IMAGE_UNCHANGED);
    cvtColor(img,img,CV_RGB2GRAY);
    if (img.empty()) //check whether the image is loaded or not
    {
        cout << "Error : Image cannot be loaded..!!" << endl;
    }
    else
    {
     img.convertTo(img,CV_32F,(1.0),(0.0));
     dct(img,img,0);
     img.copyTo(img1);
    }

   Mat f1,f2,f3,f4;
   VideoCapture cap(0);

  if(!cap.isOpened())
  {
    cout<<"File is Missing"<<endl;   //File is not opened
    return -1;
  }

  Mat frame;
  while(true)
  {
    bool f=cap.read(frame);
    if (!f) //if not success, break loop
    {
      cout << "Cannot read the frame from video file" << endl;
      break;
    }
    cvtColor(frame,f1,CV_RGB2GRAY);
    resize(f1,f2,Size(512,512));
    f2.convertTo(f2,CV_32F);
    dct(f2,f2,0);
    f2.copyTo(f3);


     float p,q,r;
     p=f2.at<float>(50,50);
     q=img1.at<float>(50,50);
     r=p+q;
     f3.at<float>(50,50)=r;

    idct(f3,f3);
    f3.convertTo(f3,CV_8U);
    imshow("Watermarked Video",f3);
    waitKey(10);
  }
  cvDestroyWindow("Input Video");
  cvDestroyWindow("WaterMarked Video");
  return 0;
}


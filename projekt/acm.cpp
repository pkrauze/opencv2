//opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//C
#include <stdio.h>
//C++
#include <iostream>
#include <cmath>
#include <limits>

using namespace cv;
using namespace std;

Mat_<double> decorate_with_contours_from_acm_matrix
(const Mat_<double> &image, const Mat_<double> &LSF) {
  // draw contour based on LSF
  // (contours are drawn around areas with negative values)

  // it tries to imitate Matlab behaviour for drawing contours

  Mat new_image = image.clone();

  Mat contour_mat;

  // only take negative
  double limit = 0.0;
  threshold(LSF, contour_mat, limit, 1, THRESH_BINARY);

  // convert to uchars
  contour_mat.convertTo(contour_mat, CV_8UC1, 255.0);

  // swap negative/positive
  contour_mat = contour_mat * -1 + 255;

  // find contours (positive areas)
  vector<vector<Point> > contours;
  findContours(contour_mat, contours, RETR_LIST, CHAIN_APPROX_NONE);

  Mat_<uchar> clean_img = Mat::zeros(image.size(), CV_8UC1);

  int which_contour = -1;     // means 'all'
  Scalar colour(250);
  drawContours(new_image, contours, which_contour, colour);

  return new_image;
}

// function g = NeumannBoundCond(f)
// % Neumann boundary condition
// [nrow,ncol] = size(f);
// g = f;
// g([1 nrow],[1 ncol]) = g([3 nrow-2],[3 ncol-2]);
// g([1 nrow],2:end-1) = g([3 nrow-2],2:end-1);
// g(2:end-1,[1 ncol]) = g(2:end-1,[3 ncol-2]);
void wypisz(Mat_<double> u) {
  cout.precision(3);
  cout << "WYPISUJE" << endl;
  for(int i=0; i<40; i++) {
    for(int j=0; j<40; j++) {
      cout << fixed << u.at<double>(i,j) << " ";
    }
    cout << endl;
  }
  cout << endl;
}

Mat neumann_boundary_condition(const Mat &in) {

    // copies values that are 2 pixels away from borders onto borders

    Mat out = in.clone();

    // 4 corners

    out.at<float>(Point(0, 0)) = out.at<float>(Point(2, 2));
    out.at<float>(Point(out.cols - 1, 0)) = out.at<float>(Point(out.cols - 3, 2));
    out.at<float>(Point(0, out.rows - 1)) = out.at<float>(Point(2, out.rows - 3));
    out.at<float>(Point(out.cols - 1, out.rows - 1)) =
            out.at<float>(Point(out.cols - 3, out.rows - 3));

    // top/bottom edges (without corners)

    Rect top_edge(Point(1, 0), Size(out.cols - 2, 1));
    Rect bottom_edge(Point(1, out.rows - 1), Size(out.cols - 2, 1));
    Rect one_of_top_rows(Point(1, 2), Size(out.cols - 2, 1));
    Rect one_of_bottom_rows(Point(1, out.rows - 3), Size(out.cols - 2, 1));

    out(one_of_top_rows).copyTo(out(top_edge));
    out(one_of_bottom_rows).copyTo(out(bottom_edge));

    // left/right edges (without corners)

    Rect left_edge(Point(0, 1), Size(1, out.rows - 2));
    Rect right_edge(Point(out.cols - 1, 1), Size(1, out.rows - 2));
    Rect one_of_left_cols(Point(2, 1), Size(1, out.rows - 2));
    Rect one_of_right_cols(Point(out.cols - 3, 1), Size(1, out.rows - 2));

    out(one_of_left_cols).copyTo(out(left_edge));
    out(one_of_right_cols).copyTo(out(right_edge));

    return out;
}

// function k = curvature_central(u)
// % compute curvature
// [ux,uy] = gradient(u);
// normDu = sqrt(ux.^2+uy.^2+1e-10);
// Nx = ux./normDu;
// Ny = uy./normDu;
// [nxx,~] = gradient(Nx);
// [~,nyy] = gradient(Ny);
// k = nxx+nyy;
Mat curvature_central(Mat u) {
  Mat K(u.size(), u.type());

  Mat ux, uy;
  Sobel(u, ux, -1, 1, 0, 1, 0.5);
  Sobel(u, uy, -1, 0, 1, 1, 0.5);

  Mat normDu = (ux.mul(ux) + uy.mul(uy)) + 1e-10;
  sqrt(normDu, normDu);

  Mat Nx = ux.mul(1 / normDu);
  Mat Ny = uy.mul(1 / normDu);

  Mat nxx, nyy;

  Sobel(Nx, nxx, -1, 1, 0, 1, 0.5);
  Sobel(Ny, nyy, -1, 0, 1, 1, 0.5);

  K = nxx + nyy;

  return K;
}

// function [f1,f2]=exchange(f1,f2,isExchange)
// %exchange f1 and f2
// if isExchange==0
//     return;
// end
// if isExchange==1
//     f1=min(f1,f2);
//     f2=max(f1,f2);
// end
// if isExchange==-1
//     f1=max(f1,f2);
//     f2=min(f1,f2);
// end
pair<Mat_<double>, Mat> exchange(Mat f1, Mat f2, int isExchange) {
  Mat f1_min(f1.size(), CV_64FC1), f2_max(f2.size(), CV_64FC1);
  if(isExchange == 0) {
  }
  if(isExchange == 1) {
    // f1 = min(f1, f2);
    for(int i = 0; i < f1.rows; i++) {
      for(int j = 0; j < f1.cols; j++) {
        double f1_val = f1.at<double>(Point(j,i));
        double f2_val = f2.at<double>(Point(j,i));
        double elem = f1_val;
        if (f2_val > f1_val) {
          elem = f2_val;
        }
        f1_min.at<double>(Point(j,i)) = elem;
      }
    }

    //f2 = max(f1, f2);
    for(int i = 0; i < f2.rows; i++) {
      for(int j = 0; j < f2.cols; j++) {
        double f1_val = f1.at<double>(Point(j,i));
        double f2_val = f2.at<double>(Point(j,i));
        double elem = f2_val;
        if (f1_val < f2_val) {
          elem = f1_val;
        }
        f2_max.at<double>(Point(j,i)) = elem;
      }
    }
  }
  if(isExchange == -1) {
    max(f1, f2, f1);
    min(f1, f2, f2);
  }

  return make_pair(f1_min, f2_max);
}

// function [u,f1,f2]= ACM(u,Img,Ksigma,KI,KI2,KONE,nu,timestep,mu,epsilon,lambda1,lambda2,CV,RSF,LRCV,LIF,LGD,isExchange)
tuple<Mat, Mat, Mat> ACM(Mat_<double> u, Mat Img, Mat Ksigma, Mat KI, Mat KI2, Mat KONE, double nu, double timestep,
                         int mu, double epsilon,int lambda1, int lambda2,
                         int CV, int RSF, int LRCV, int LIF,int LGD, int isExchange) {
  // u = NeumannBoundCond(u);
  // K = curvature_central(u);
  u = neumann_boundary_condition(u);
  Mat_<double> K = curvature_central(u);
  // Hu=0.5*(1+(2/pi)*atan(u./epsilon));
  Mat Hu(u.size(), CV_64FC1, Scalar::all(0));

  for(int i=0; i<Hu.rows; i++){
    for(int j=0; j<Hu.cols; j++){
	    double u_elem = u.at<double>(Point(j,i));
      Hu.at<double>(Point(j, i)) =  0.5 * (1 + (2/M_PI) * atan(u_elem / epsilon));
    }
  }
  // DrcU=(epsilon/pi)./(epsilon^2.+u.^2);
  Mat_<double> DrcU = 1 / (u.mul(u) + epsilon * epsilon) * epsilon / M_PI;

  // // // KIH= imfilter((Hu.*Img),Ksigma,'replicate');
  Mat_<double> KIH;
  filter2D(Hu.mul(Img), KIH, CV_64FC1, Ksigma, Point(-1,-1), 0, BORDER_REPLICATE);

  // // // KH= imfilter(Hu,Ksigma,'replicate');
  Mat_<double> KH;
  filter2D(Hu, KH, CV_64FC1, Ksigma, Point(-1,-1), 0, BORDER_REPLICATE);

  // // // f1=KIH./KH;
  Mat_<double> f1 = KIH / KH;

  // // // f2=(KI-KIH)./(KONE-KH);
  Mat_<double> f2 =  (KI-KIH)/(KONE-KH);
  auto ret = exchange(f1, f2, isExchange);
  f1 = ret.first;
  f2 = ret.second;

  // if CV~=0
  //     c= Hu.*Img;
  //     C1 = sum(c(:))/sum(Hu(:));
  //     c1=(1-Hu).*Img;
  //     c2=1-Hu;
  //     C2 = sum(c1(:))/sum(c2(:));
  //     CVterm=CV*(DrcU.*(-lambda1*(Img-C1).^2+lambda2*(Img-C2).^2));
  // else
  //     CVterm=0;
  // end
  Mat_<double> CVterm;
  if(CV!=0) {
    Mat_<double> c = Hu.mul(Img);
    auto C1 = sum(c) / sum(Hu); /* TO DO: mat sum */
    Mat_<double> c1 = (1 - Hu).mul(Img);
    Mat_<double> c2 = 1 - Hu;
    auto C2 = sum(c1) / sum(c2);
    Mat_<double> tempC1 = Img-C1;
    Mat_<double> tempC2 = Img-C2;
    CVterm = CV * (DrcU.mul((-lambda1 * tempC1.mul(tempC1) + lambda2 * tempC2.mul(tempC2))));
  } else {
    CVterm = 0;
  }

  // if LRCV~=0
  //     LRCVterm=LRCV*DrcU.*(-lambda1*(Img-f1).^2+lambda2*(Img-f2).^2);
  // else
  //     LRCVterm=0;
  // end
  Mat_<double> LRCVterm;
  if(LRCV!=0) {
    Mat_<double> tempf1 = Img-f1;
    Mat_<double> tempf2 = Img-f2;
    LRCVterm = LRCV * DrcU.mul((-lambda1 * tempf1.mul(tempf1) + lambda2 * tempf2.mul(tempf2)));
  }
  else
    LRCVterm = 0;

  // if LIF~=0
  //     LIFterm=DrcU.*((Img - f1.*Hu - f2.*(1 - Hu)).*(f1 - f2));
  // else
  //     LIFterm=0;
  // end
  Mat_<double> LIFterm;
  if(LIF!=0)
    LIFterm = DrcU.mul(((Img - f1.mul(Hu) - f2.mul((1 - Hu))).mul((f1 - f2))));
  else
    LIFterm = 0;

  // if RSF~=0
  //     s1=lambda1.*f1.^2-lambda2.*f2.^2;
  //     s2=lambda1.*f1-lambda2.*f2;
  //     dataForce=(lambda1-lambda2)*KONE.*Img.*Img+imfilter(s1,Ksigma,'replicate')-2.*Img.*imfilter(s2,Ksigma,'replicate');
  //     RSFterm=-RSF*DrcU.*dataForce;
  // else
  //     RSFterm=0;
  // end
  Mat_<double> RSFterm;
  if(RSF!=0) {
    Mat_<double> s1 = lambda1*(f1.mul(f1)) - lambda2*(f2.mul(f2));
    Mat_<double> s2 = lambda1*f1 - lambda2*f2;
    Mat_<double> dataForce, filtered_s1, filtered_s2;
    filter2D(s1, filtered_s1, CV_64FC1, Ksigma, Point(-1,-1), 0, BORDER_REPLICATE);
    filter2D(s2, filtered_s2, CV_64FC1, Ksigma, Point(-1,-1), 0, BORDER_REPLICATE);
    dataForce = (lambda1 - lambda2) * KONE.mul(Img).mul(Img) + filtered_s1 - 2 * Img.mul(filtered_s2);
    RSFterm =- RSF * DrcU.mul(dataForce);
  } else {
    RSFterm = 0;
  }
  //wypisz(RSFterm);
  // if LGD~=0
  //     KI2H = imfilter(Img.^2.*Hu,Ksigma,'replicate');
  //     sigma1 = (f1.^2.*KH - 2.*f1.*KIH + KI2H)./(KH);
  //     sigma2 = (f2.^2.*KONE - f2.^2.*KH - 2.*f2.*KI + 2.*f2.*KIH + KI2 - KI2H)./(KONE-KH);

  // else
  //     LGDterm=0;
  // end
  Mat_<double> LGDterm;
  if(LGD!=0) {
    Mat_<double> ImgHu2 = Img.mul(Img).mul(Hu);
    Mat_<double> KI2H;
    filter2D(ImgHu2, KI2H, -1, Ksigma, Point(-1,-1), 0, BORDER_REPLICATE);
    Mat_<double> sigma1 = (f1.mul(f1).mul(KH) - (2 * f1.mul(KI2H))) / KH;
    Mat_<double> sigma2 = (f2.mul(f1).mul(KONE) - f2.mul(f2).mul(KH) - (2 * f2.mul(KI)) + (2 * f2.mul(KIH)) + KI2 - KI2H) / (KONE - KH);

    // //     localForce = (lambda1 - lambda2).*KONE.*log(sqrt(2*pi)) ...
    // //         + imfilter(lambda1.*log(sqrt(sigma1)) - lambda2.*log(sqrt(sigma2)) + lambda1.*f1.^2./(2.*sigma1) - lambda2.*f2.^2./(2.*sigma2) ,Ksigma,'replicate')...
    // //         + Img.*imfilter(lambda2.*f2./sigma2 - lambda1.*f1./sigma1,Ksigma,'replicate')...
    // //         + Img.^2.*imfilter(lambda1.*1./(2.*sigma1) - lambda2.*1./(2.*sigma2),Ksigma,'replicate');
    // //     LGDterm = -LGD*DrcU.*localForce;
    Mat_<double> c_sigma2, c_sigma1;
    sqrt(sigma2, c_sigma2);
    log(c_sigma2, c_sigma2);
    sqrt(sigma1, c_sigma1);
    log(c_sigma1, c_sigma1);

    Mat_<double> calculated_lambda1 = (lambda1 * c_sigma1) - (lambda2 * c_sigma2) + (lambda1 * f1.mul(f1))/(2*sigma1) - (lambda2*f2.mul(f2)/(2*sigma2));
    Mat_<double> filtered_lambda1, filtered_lambda1_sig1, filtered_lambda2_sig2;
    filter2D(calculated_lambda1, filtered_lambda1, -1, Ksigma, Point(-1,-1), 0, BORDER_REPLICATE);
    filter2D((lambda2*f2/sigma2 - lambda1*f1/sigma1), filtered_lambda1_sig1, -1, Ksigma, Point(-1,-1), 0, BORDER_REPLICATE);
    filter2D(((lambda1*1/(2*sigma1) - lambda2*1/(2*sigma2))), filtered_lambda2_sig2, -1, Ksigma, Point(-1,-1), 0, BORDER_REPLICATE);
    Mat_<double> localForce = ((lambda1 - lambda2) * KONE * log(sqrt(2*M_PI)));// + filtered_lambda1;// + Img * filtered_lambda1_sig1 + Img.mul(Img) * filtered_lambda2_sig2;
    localForce += filtered_lambda1 + Img.mul(filtered_lambda1_sig1) + Img.mul(Img).mul(filtered_lambda2_sig2);
    LGDterm = -1 * LGD * DrcU.mul(localForce);
  } else {
    LGDterm = 0;
  }

  // // PenaltyTerm=mu*(4*del2(u)-K);
  Mat_<double> laplacian_u;
  Laplacian(u, laplacian_u, u.depth(), 3, 1, 0, BORDER_DEFAULT);
  laplacian_u.convertTo(laplacian_u, CV_64F);
  //cout << K.type() << endl;
  //cout << (4 * laplacian_u).type() << endl;
  Mat_<double> PenaltyTerm = (4 * laplacian_u - K) * mu;

  // // LengthTerm=nu.*DrcU.*K;
  Mat_<double> LengthTerm = nu * DrcU.mul(K);

  // // u=u+timestep*(LengthTerm+PenaltyTerm+RSFterm+LRCVterm+LIFterm+LGDterm+CVterm);
  u = (u + timestep).mul(LengthTerm + PenaltyTerm + RSFterm + LRCVterm + LIFterm + LGDterm + CVterm);

  // cout << "NIE WYCHODZE" << endl;
  return make_tuple(u, f1, f2);
}


int main() {
  // clc; clear all; close all;
  Mat Img = imread("1.bmp", IMREAD_GRAYSCALE);
  // Img = imread('4.bmp');
  // Img = double(Img(:,:,1));

  // % -----set initial contour-----
  // c0 = 1;
  // initialLSF = ones(size(Img(:,:,1))).*c0;
  // initialLSF(15:35,40:60) = -c0;
  Mat_<double> initialLSF = Mat::ones(Img.rows, Img.cols, CV_64F) * 1;

  Rect rec(Point(40, 15), Size(20, 20));
  initialLSF(rec).setTo(-1.0);

  Mat_<double> u = initialLSF;
  // // u = initialLSF;
  // // h1 = figure(1);
  // // imagesc(Img, [0, 255]);
  // // colormap(gray);
  // // hold on;
  // // axis off,axis equal
  // // contour(initialLSF,[0 0],'g','linewidth',1.5);
  // // hold off
  // // pause(0.1);

  // // % -----set parameters-----
  // // mu = 1; % the distance regularization term
  // // nu = 0.001*255*255; % the length term. 
  // // lambda1 = 1; 
  // // lambda2 = 1; 
  // // epsilon = 1.0;
  // // timestep = 0.05;
  // // iterNum = 200; % the number of iterations. 
  // // sigma=3; % control the local size
  int mu = 1;
  double nu = 0.001*255*255;
  int lambda1 = 1;
  int lambda2 = 1;
  double epsilon = 1.0;
  double timestep = 0.05;
  int iterNum = 200;
  int sigma = 3;

  // // Ksigma= fspecial('gaussian',round(2*sigma)*2+1,sigma);
  // int WinSize = round(2*sigma)*2+1;
  // Mat_<double> Ksigma(WinSize, WinSize, CV_64F);
  // for (int i=0;i<Ksigma.rows;i++){
  //   for (int j=0;j<Ksigma.cols;j++){
  //     double x = (double) j - (double) WinSize/2.0;
  //     double y = (double) i - (double) WinSize/2.0;
  //     Ksigma.at<double>(j,i) = (1.0 /(M_PI*pow(sigma,4))) * (1 - (x*x+y*y)/(sigma*sigma))* (pow(2.718281828, - (x*x + y*y) / 2*sigma*sigma));
  //   }
  // }
  Mat_<double> gauss_kernel_1d = getGaussianKernel(round(2*sigma)*2+1, sigma, CV_64FC1);
  Mat_<double> gauss_kernel_2d = gauss_kernel_1d * gauss_kernel_1d.t();
  Mat_<double> Ksigma = gauss_kernel_2d;

  // KONE = imfilter(ones(size(Img)),Ksigma,'replicate');
  Mat_<double> KONE;
  filter2D((Mat::ones(Img.size(), CV_64FC1)), KONE, CV_64FC1, Ksigma);

  // KI = imfilter(Img,Ksigma,'replicate');
  Mat_<double> KI;
  filter2D(Img, KI, CV_64FC1, Ksigma, Point(-1,-1), 0, BORDER_REPLICATE);

  // KI2 = imfilter(Img.^2,Ksigma,'replicate');
  Mat_<double> KI2;
  filter2D(Img.mul(Img), KI2, CV_64FC1, Ksigma, Point(-1, -1), 0, BORDER_REPLICATE);
  // % --- model weight ---
  // RSF = 1;
  // LRCV = 0;
  // LIF = 0;
  // LGD = 0;
  // CV = 0;
  int RSF = 1;
  int LRCV = 0;
  int LIF = 0;
  int LGD = 0;
  int CV = 0;

  // isExchange = 1; % '1' for bright object and dark backgroud; 
  //                 % '-1' for dark object and bright backgroud;
  //                 % '0' represent original model.
  int isExchange = 1;

  // % -----start level set evolution-----
  // h2=figure(2);
  // tic
  // for n=1:iterNum
  //     [u,f1,f2]= ACM(u,Img,Ksigma,KI,KI2,KONE,nu,timestep,mu,epsilon,lambda1,lambda2,CV,RSF,LRCV,LIF,LGD,isExchange);
  //     if mod(n,10)==0
  //         imagesc(Img, [0, 255]);
  //         colormap(gray);
  //         hold on;
  //         axis off,axis equal
  //         contour(u,[0 0],'r');
  //         title([num2str(n), ' iterations ']);
  //         hold off;
  //         pause(.1);
  //     end
  // end
  // toc
 
  //resize(Img, Img, Size(250,250));
  imshow("image", Img);
  Mat_<double> f1, f2;
  Mat_<double> img_copy;
  Img.convertTo(img_copy, CV_64FC1);

  for(int n=0; n<iterNum; n++) {
    //u.convertTo(u, CV_64F);

    auto acm_result = ACM(u,img_copy,Ksigma,KI,KI2,KONE,nu,timestep,mu,epsilon,lambda1,lambda2,CV,RSF,LRCV,LIF,LGD,isExchange);

    // //cout << get<0>(acm_result).type() << endl;
    // //cout << u.type() << endl;
    u = get<0>(acm_result);
    f1 = get<1>(acm_result);
    f2 = get<2>(acm_result);
    //wypisz(u);

    if(n%10 == 0) {
      Mat display_image = decorate_with_contours_from_acm_matrix(Img, u);
      imshow("deb", Img);
      imshow("deb2", img_copy);
      resize(display_image, display_image, Size(250,250));

      // text: iteration number X
      putText(display_image, "Iter: "+to_string(n), Point(20,20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(250));

      imshow("debug_window", display_image);
      imshow("tak", u);

      int wait_time = 5000;
      waitKey(wait_time);
    }
  }

  // % -----display result-----
  // imagesc(Img, [0, 255]);colormap(gray);hold on;axis off,axis equal
  // [c,h] = contour(u,[0 0],'r','linewidth',1.5);
  // % figure;mesh(u);colorbar;title('Final level set function');hold on, contour(u,[0 0],'r','linewidth',1.5);

  return 0;
}

#include "active_contour.h"

Mat neumann_boundary_condition(const Mat &in) {
    Mat mat = in.clone();
    // corners
    mat.at<float>(Point(0, 0)) = mat.at<float>(Point(2, 2));
    mat.at<float>(Point(mat.cols - 1, 0)) = mat.at<float>(Point(mat.cols - 3, 2));
    mat.at<float>(Point(0, mat.rows - 1)) = mat.at<float>(Point(2, mat.rows - 3));
    mat.at<float>(Point(mat.cols - 1, mat.rows - 1)) = mat.at<float>(Point(mat.cols - 3, mat.rows - 3));

    // edges
    Rect top_edge(Point(1, 0), Size(mat.cols - 2, 1));
    Rect bottom_edge(Point(1, mat.rows - 1), Size(mat.cols - 2, 1));
    Rect one_of_top_rows(Point(1, 2), Size(mat.cols - 2, 1));
    Rect one_of_bottom_rows(Point(1, mat.rows - 3), Size(mat.cols - 2, 1));
    mat(one_of_top_rows).copyTo(mat(top_edge));
    mat(one_of_bottom_rows).copyTo(mat(bottom_edge));
    Rect left_edge(Point(0, 1), Size(1, mat.rows - 2));
    Rect right_edge(Point(mat.cols - 1, 1), Size(1, mat.rows - 2));
    Rect one_of_left_cols(Point(2, 1), Size(1, mat.rows - 2));
    Rect one_of_right_cols(Point(mat.cols - 3, 1), Size(1, mat.rows - 2));

    mat(one_of_left_cols).copyTo(mat(left_edge));
    mat(one_of_right_cols).copyTo(mat(right_edge));

    return mat;
}

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

pair<Mat, Mat> exchange(Mat f1, Mat f2, int isExchange) {
  Mat f1_min(f1.size(), CV_64FC1), f2_max(f2.size(), CV_64FC1);
  if(isExchange == 0) {
  }
  for(int i = 0; i < f1.rows; i++) {
    for(int j = 0; j < f1.cols; j++) {
      double f1_val = f1.at<double>(Point(j,i));
      double f2_val = f2.at<double>(Point(j,i));
      double elem = f1_val;
      if(isExchange == 1) {
        if (f2_val < f1_val) {
          elem = f2_val;
        }
      } else if(isExchange == -1) {
        if (f2_val > f1_val) {
          elem = f2_val;
        }
      }
      f1_min.at<double>(Point(j,i)) = elem;
    }
  }
  for(int i = 0; i < f2.rows; i++) {
    for(int j = 0; j < f2.cols; j++) {
      double f1_val = f1.at<double>(Point(j,i));
      double f2_val = f2.at<double>(Point(j,i));
      double elem = f2_val;
      if(isExchange == 1) {
        if (f1_val > f2_val) {
          elem = f1_val;
        }
      } else if(isExchange == -1) {
        if (f1_val < f2_val) {
          elem = f1_val;
        }
      }
      f2_max.at<double>(Point(j,i)) = elem;
    }
  }
  return make_pair(f1_min, f2_max);
}

tuple<Mat_<double>, Mat_<double>, Mat_<double>> ACM(Mat_<double> u, Mat_<double> Img,
                                                    Mat_<double> Ksigma, Mat_<double> KI, Mat_<double> KI2, Mat_<double> KONE,
                                                    double nu, double timestep, double mu, double epsilon, double lambda1,
                                                    double lambda2, int CV, int RSF, int LRCV, int LIF, int LGD, int isExchange) {
  u = neumann_boundary_condition(u);
  Mat_<double> K = curvature_central(u);
  Mat Hu(u.size(), CV_64FC1, Scalar::all(0));

  for(int i=0; i<Hu.rows; i++){
    for(int j=0; j<Hu.cols; j++){
	    double u_elem = u.at<double>(Point(j,i));
      Hu.at<double>(Point(j, i)) =  0.5 * (1 + (2/M_PI) * atan(u_elem / epsilon));
    }
  }
  Mat_<double> DrcU = 1 / (u.mul(u) + epsilon * epsilon) * epsilon / M_PI;

  Mat_<double> KIH;
  filter2D(Hu.mul(Img), KIH, CV_64FC1, Ksigma, Point(-1,-1), 0, BORDER_REPLICATE);

  Mat_<double> KH;
  filter2D(Hu, KH, CV_64FC1, Ksigma, Point(-1,-1), 0, BORDER_REPLICATE);

  Mat_<double> f1 = KIH / KH;
  Mat_<double> f2 =  (KI-KIH);
  divide(f2, KONE-KH, f2);
  auto ret = exchange(f1, f2, isExchange);
  f1 = ret.first;
  f2 = ret.second;

  Mat_<double> CVterm;
  if(CV!=0) {
    Mat_<double> c = Hu.mul(Img);
    auto C1 = sum(c) / sum(Hu);
    Mat_<double> c1 = (1 - Hu).mul(Img);
    Mat_<double> c2 = 1 - Hu;
    auto C2 = sum(c1) / sum(c2);
    Mat_<double> tempC1 = Img-C1;
    Mat_<double> tempC2 = Img-C2;
    CVterm = CV * (DrcU.mul((-lambda1 * tempC1.mul(tempC1) + lambda2 * tempC2.mul(tempC2))));
  } else {
    CVterm = 0;
  }

  Mat_<double> LRCVterm;
  if(LRCV!=0) {
    Mat_<double> tempf1 = Img-f1;
    Mat_<double> tempf2 = Img-f2;
    LRCVterm = LRCV * DrcU.mul((-lambda1 * tempf1.mul(tempf1) + lambda2 * tempf2.mul(tempf2)));
  }
  else
    LRCVterm = 0;

  Mat_<double> LIFterm;
  if(LIF!=0)
    LIFterm = DrcU.mul(((Img - f1.mul(Hu) - f2.mul((1 - Hu))).mul((f1 - f2))));
  else
    LIFterm = 0;

  Mat_<double> RSFterm;
  if(RSF!=0) {
    Mat_<double> s1 = lambda1*f1.mul(f1) - f2.mul(f2);
    Mat_<double> s2 = lambda1*f1 - lambda2*f2;
    Mat_<double> dataForce, filtered_s1, filtered_s2;
    filter2D(s1, filtered_s1, CV_64FC1, Ksigma, Point(-1,-1), 0, BORDER_REPLICATE);
    filter2D(s2, filtered_s2, CV_64FC1, Ksigma, Point(-1,-1), 0, BORDER_REPLICATE);
    dataForce = (lambda1 - lambda2) * KONE.mul(Img).mul(Img) + filtered_s1 - 2 * Img.mul(filtered_s2);
    RSFterm =- RSF * DrcU.mul(dataForce);
  } else {
    RSFterm = 0;
  }

  Mat_<double> LGDterm;
  if(LGD!=0) {
    Mat_<double> ImgHu2 = Img.mul(Img).mul(Hu);
    Mat_<double> KI2H;
    filter2D(ImgHu2, KI2H, -1, Ksigma, Point(-1,-1), 0, BORDER_REPLICATE);
    Mat_<double> sigma1 = (f1.mul(f1).mul(KH) - (2 * f1.mul(KI2H))) / KH;
    Mat_<double> sigma2 = (f2.mul(f1).mul(KONE) - f2.mul(f2).mul(KH) - (2 * f2.mul(KI)) + (2 * f2.mul(KIH)) + KI2 - KI2H) / (KONE - KH);

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
    Mat_<double> localForce = ((lambda1 - lambda2) * KONE * log(sqrt(2*M_PI)));
    localForce += filtered_lambda1 + Img.mul(filtered_lambda1_sig1) + Img.mul(Img).mul(filtered_lambda2_sig2);
    LGDterm = -1 * LGD * DrcU.mul(localForce);
  } else {
    LGDterm = 0;
  }

  Mat_<double> laplacian_u;
  Laplacian(u, laplacian_u, -1, 1, 0.25);
  laplacian_u.convertTo(laplacian_u, CV_64FC1);

  Mat_<double> PenaltyTerm = (4 * laplacian_u - K) * mu;
  Mat_<double> LengthTerm = nu * DrcU.mul(K);
  u = u + timestep * (LengthTerm + PenaltyTerm + RSFterm + LRCVterm + LIFterm + LGDterm + CVterm);

  return make_tuple(u, f1, f2);
}

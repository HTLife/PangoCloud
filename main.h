#ifndef MAIN_H
#define MAIN_H

#include "PangoCloud.h"
#include <opencv2/opencv.hpp>

void draw();
void
convertMatToCloud(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &ptCloud,
    cv::Mat &depthMat,
    IplImage *rgbimg);
#endif // MAIN_H

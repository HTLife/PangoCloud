/**
 *  Minimal example of showing point cloud with Pangolin
 *
 *  Jacky Liu jjkka132 (at) gmail (dot) com
 *  Mar 12 2017
 */




#include <iostream>
#include <pangolin/pangolin.h>
#include <pangolin/gl/gldraw.h>
#include <pcl/common/transforms.h>

#include <pcl/registration/icp.h>

#include "main.h"


PangoCloud *cloud;
PangoCloud *cloud2;
CAMERA_INTRINSIC_PARAMETERS camera;
FRAME frame1, frame2;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr ptCloud (new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr ptCloud2 (new pcl::PointCloud<pcl::PointXYZRGB>);

struct CustomType
{
    CustomType()
        : x(0), y(0.0f) {}

    CustomType(int x, float y, std::string z)
        : x(x), y(y), z(z) {}

    int x;
    float y;
    std::string z;
};

std::ostream& operator<< (std::ostream& os, const CustomType& o){
    os << o.x << " " << o.y << " " << o.z;
    return os;
}

std::istream& operator>> (std::istream& is, CustomType& o){
    is >> o.x;
    is >> o.y;
    is >> o.z;
    return is;
}

void SampleMethod()
{
    std::cout << "You typed ctrl-r or pushed reset" << std::endl;
    glRotated (30, 30, 30, 30);
}

void init()
{
    // Load configuration data
    pangolin::ParseVarsFile("app.cfg");

    // Create OpenGL window in single line
    pangolin::CreateWindowAndBind("PangoCloud - ICP",1024,768);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
                pangolin::ModelViewLookAt(-0,0.5,-3, 0,0,0, pangolin::AxisY)
                );

    const int UI_WIDTH = 180;

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -640.0f/480.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    // Add named Panel and bind to variables beginning 'ui'
    // A Panel is just a View with a default layout and input handling
    pangolin::CreatePanel("ui")
        .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

    // Safe and efficient binding of named variables.
    // Specialisations mean no conversions take place for exact types
    // and conversions between scalar types are cheap.
    pangolin::Var<bool> a_button("ui.A_Button",false,false);

    pangolin::Var<double> a_double("ui.A_Double",4,1.1,20);

    pangolin::Var<int> an_int("ui.An_Int",2,0,5);
    pangolin::Var<double> a_double_log("ui.Log_scale var",3,1,1E4, true);
    pangolin::Var<bool> a_checkbox("ui.A_Checkbox",false,true);
    pangolin::Var<int> an_int_no_input("ui.An_Int_No_Input",2);
    pangolin::Var<CustomType> any_type("ui.Some_Type", CustomType(0,1.2f,"Hello") );

    pangolin::Var<bool> save_window("ui.Save_Window",false,false);
    pangolin::Var<bool> save_cube("ui.Save_Cube",false,false);

    pangolin::Var<bool> record_cube("ui.Record_Cube",false,false);

    // std::function objects can be used for Var's too. These work great with C++11 closures.
    pangolin::Var<std::function<void(void)> > reset("ui.Reset", SampleMethod);

    // Demonstration of how we can register a keyboard hook to alter a Var
    pangolin::RegisterKeyPressCallback(pangolin::PANGO_CTRL + 'b', pangolin::SetVarFunctor<double>("ui.A Double", 3.5));

    // Demonstration of how we can register a keyboard hook to trigger a method
    pangolin::RegisterKeyPressCallback(pangolin::PANGO_CTRL + 'r', SampleMethod);
    // Default hooks for exiting (Esc) and fullscreen (tab).



    loadImages();
    double dLastTh = 0;
    while( !pangolin::ShouldQuit() )
    {
        // Clear entire screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if( pangolin::Pushed(a_button) )
          std::cout << "You Pushed a button!" << std::endl;

        // Overloading of Var<T> operators allows us to treat them like
        // their wrapped types, eg:
        if( a_checkbox )
          an_int = (int)a_double;

        if( !any_type->z.compare("robot"))
          any_type = CustomType(1,2.3f,"Boogie");

        an_int_no_input = an_int;

        if( pangolin::Pushed(save_window) )
          pangolin::SaveWindowOnRender("window");

        if( pangolin::Pushed(save_cube) )
          d_cam.SaveOnRender("cube");

        if( pangolin::Pushed(record_cube) )
          pangolin::DisplayBase().RecordOnRender("ffmpeg:[fps=50,bps=8388608,unique_filename]//screencap.avi");

        // Activate efficiently by object
        d_cam.Activate(s_cam);

        // Render some stuff
        glColor3f(1.0,1.0,1.0);
        //pangolin::glDrawColouredCube();
        //pangolin::glDrawAxis(1);
        //glTranslatef(0.0f, 0.0f, -100.0f);
        //glColor4f(0,0,1,1);
        //pangolin::glDrawLine(0,0,0, 0,0,1);
        if (dLastTh != a_double)
        {
            dLastTh = a_double;
            calculate(a_double);
        }
        draw();
        //glBegin(GL_LINES);
        //      glPointSize(20.0);
        //      glBegin(GL_POINTS);
        //        //glVertex3f(0.0f, 0.0f, 0.0f);
        //        glVertex3f(5.0f, 5.0f, 5.0f);
        //      glEnd();
        //      glBegin(GL_LINES);
        //        glVertex3f(0.0f, 0.0f, 0.0f);
        //        glVertex3f(4.9f, 4.9f, 4.9f);
        //      glEnd();
        // Swap frames and Process Events
        pangolin::FinishFrame();
    }
}

void draw()
{
    cloud->drawPoints();
    cloud2->drawPoints();
}

void loadImages()
{
    /// RGB
    cv::Mat rgbMat = imread("../1.png", cv::IMREAD_UNCHANGED);
    cv::Mat rgbMat2 = imread("../2.png", cv::IMREAD_UNCHANGED);


    /// Depth
    cv::Mat depthMat = imread("../1d.png", cv::IMREAD_UNCHANGED);
    double depthScale = 0.0001;
    depthMat.convertTo(depthMat, CV_16UC1, 1000 * depthScale);

    cv::Mat depthMat2 = imread("../2d.png", cv::IMREAD_UNCHANGED);
    depthMat2.convertTo(depthMat2, CV_16UC1, 1000 * depthScale);

    //int32_t depthSize = depthMat.total() * depthMat.elemSize();

    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr ptCloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr ptCloud2 (new pcl::PointCloud<pcl::PointXYZRGB>);



    convertMatToCloud(ptCloud, depthMat, rgbMat);
    convertMatToCloud(ptCloud2, depthMat2, rgbMat2);



    frame1.rgb = rgbMat;
    frame1.depth = depthMat;
    frame2.rgb = rgbMat2;
    frame2.depth = depthMat2;


    // 提取特征并计算描述子
    cout<<"extracting features"<<endl;
    string detecter = "SIFT";
    string descriptor = "SIFT";

    computeKeyPointsAndDesp( frame1, detecter, descriptor );
    computeKeyPointsAndDesp( frame2, detecter, descriptor );


}

Eigen::Affine3d create_rotation_matrix(
        double ax,
        double ay,
        double az) {
  Eigen::Affine3d rx =
      Eigen::Affine3d(Eigen::AngleAxisd(ax, Eigen::Vector3d(1, 0, 0)));
  Eigen::Affine3d ry =
      Eigen::Affine3d(Eigen::AngleAxisd(ay, Eigen::Vector3d(0, 1, 0)));
  Eigen::Affine3d rz =
      Eigen::Affine3d(Eigen::AngleAxisd(az, Eigen::Vector3d(0, 0, 1)));
  return rz * ry * rx;
}


void calculate(double th)
{
    RESULT_OF_PNP result = estimateMotion( frame1, frame2, camera, th );
    cout<<result.rvec<<endl<<result.tvec<<endl;

    // 处理result
    // 将旋转向量转化为旋转矩阵
    cv::Mat R;
    cv::Rodrigues( result.rvec, R );
    Eigen::Matrix3d r;
    cv::cv2eigen(R, r);

    // 将平移向量和旋转矩阵转换成变换矩阵
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

    Eigen::AngleAxisd angle(r);
    cout<<"translation"<<endl;
    Eigen::Translation<double,3> trans(result.tvec.at<double>(0,0), result.tvec.at<double>(0,1), result.tvec.at<double>(0,2));
    T = angle;
    T(0,3) = result.tvec.at<double>(0,0);
    T(1,3) = result.tvec.at<double>(0,1);
    T(2,3) = result.tvec.at<double>(0,2);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr ptCloud_new (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud( *ptCloud2, *ptCloud_new, T.matrix() );
    //Eigen::Vector3f v3f(0.0, 0.0, 0.0);
    //Eigen::Quaternionf rotation (0.966, 0, -0.259, 0);


    //pcl::transformPointCloud(*ptCloud2, *ptCloud2, v3f, rotation);
    //pcl::transformPointCloud (
    //    const pcl::PointCloud< PointT > &cloud_in,
    //    pcl::PointCloud< PointT > &cloud_out,
    //    const Eigen::Matrix< Scalar, 3, 1 > &offset,
    //    const Eigen::Quaternion< Scalar > &rotation)


//    Eigen::Affine3d r = create_rotation_matrix(result.rvec[0], result.rvec[1], result.rvec[2]);
//    Eigen::Affine3d t(Eigen::Translation3d(Eigen::Vector3d(result.tvec[0],result.tvec[1],result.tvec[2])));
//    Eigen::Matrix4d m4d = (t * r).matrix();
/*
    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
    icp.setInputCloud(ptCloud_new);
    icp.setInputTarget(ptCloud);

    pcl::PointCloud<pcl::PointXYZRGB> Final;
    icp.align(Final);
    std::cout << "has converged:" << icp.hasConverged() << " score: " <<
    icp.getFitnessScore() << std::endl;
    std::cout << icp.getFinalTransformation() << std::endl;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr ptCloud_new2 (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud( *ptCloud_new, *ptCloud_new2, icp.getFinalTransformation() );
*/

    cloud = new PangoCloud(ptCloud.get());
    cloud2 = new PangoCloud(ptCloud_new.get());

}

void
convertMatToCloud(
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr &ptCloud,
            cv::Mat &depthMat,
            cv::Mat &rgbMat)
//    IplImage *rgbimg)
{
    //pcl::PointCloud<pcl::PointXYZ>::Ptr ptCloud (
    //    new pcl::PointCloud<pcl::PointXYZ>);

    // calibration parameters
    //float const fx_d = 5.9421434211923247e+02;
    //float const fy_d = 5.9104053696870778e+02;
    //float const cx_d = 3.3930780975300314e+02;
    //float const cy_d = 2.4273913761751615e+02;

    float const fx_d = 591.123473;  // focal length x
    float const fy_d = 590.076012;  // focal length y
    float const cx_d = 331.038659;  // optical center x
    float const cy_d = 234.047543;  // optical center y


    unsigned char *rgbImg = (unsigned char*)(rgbMat.data);

    float factor = 5000;

    uint8_t r(255), g(15), b(15);

    unsigned char* p = depthMat.data;
    for (int i = 0; i<depthMat.rows; i++)
    {
        for (int j = 0; j < depthMat.cols; j++)
        {

            unsigned short val = depthMat.at<unsigned short>(i, j);
            float z = static_cast<float>(val);
            //float z = static_cast<float>(*p);
            pcl::PointXYZRGB point;
            point.z = z / factor;
            point.x = point.z*(cx_d - j)  / fx_d;
            point.y = point.z *(cy_d - i) / fy_d;

            b = rgbMat.at<cv::Vec3b>(i,j)[2]; //B
            g = rgbMat.at<cv::Vec3b>(i,j)[1]; //G
            r = rgbMat.at<cv::Vec3b>(i,j)[0]; //R

            uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
                        static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
            point.rgb = *reinterpret_cast<float*>(&rgb);

            ptCloud->points.push_back(point);
            ++p;
        }
    }

    ptCloud->width = (int)depthMat.cols;
    ptCloud->height = (int)depthMat.rows;

    //return ptCloud;
}



//void getIncrementalTransformation(
//        Eigen::Vector3f & trans,
//        Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & rot,
//        const DeviceArray2D<unsigned short> & depth,
//        const DeviceArray2D<PixelRGB> & image,
//        unsigned char * rgbImage,
//        unsigned short * depthData)
//{

//}





// computeKeyPointsAndDesp 同时提取关键点与特征描述子
void computeKeyPointsAndDesp( FRAME& frame, string detector, string descriptor )
{
    cv::Ptr<cv::FeatureDetector> _detector;
    cv::Ptr<cv::DescriptorExtractor> _descriptor;

    cv::initModule_nonfree();
    _detector = cv::FeatureDetector::create( detector.c_str() );
    _descriptor = cv::DescriptorExtractor::create( descriptor.c_str() );

    if (!_detector || !_descriptor)
    {
        cerr<<"Unknown detector or discriptor type !"<<detector<<","<<descriptor<<endl;
        return;
    }

    _detector->detect( frame.rgb, frame.kp );
    _descriptor->compute( frame.rgb, frame.kp, frame.desp );

    return;
}

// estimateMotion 计算两个帧之间的运动
// 输入：帧1和帧2
// 输出：rvec 和 tvec
RESULT_OF_PNP estimateMotion(
        FRAME& frame1,
        FRAME& frame2,
        CAMERA_INTRINSIC_PARAMETERS& camera,
        double th)
{
    static ParameterReader pd;
    vector< cv::DMatch > matches;
    cv::FlannBasedMatcher matcher;
    matcher.match( frame1.desp, frame2.desp, matches );

    cout<<"find total "<<matches.size()<<" matches."<<endl;
    vector< cv::DMatch > goodMatches;
    double minDis = 9999;
    //double good_match_threshold = atof( pd.getData( "good_match_threshold" ).c_str() );
    double good_match_threshold = th;//4
    for ( size_t i=0; i<matches.size(); i++ )
    {
        if ( matches[i].distance < minDis )
            minDis = matches[i].distance;
    }

    cout << "minDis=" << minDis << endl;
    cout << "th=" << good_match_threshold << endl;
    for ( size_t i=0; i<matches.size(); i++ )
    {

        cout << matches[i].distance << endl;
        if (matches[i].distance < good_match_threshold*minDis)
        {
            goodMatches.push_back( matches[i] );
        }
    }

    cout<<"good matches: "<<goodMatches.size()<<endl;
    // 第一个帧的三维点
    vector<cv::Point3f> pts_obj;
    // 第二个帧的图像点
    vector< cv::Point2f > pts_img;

    // 相机内参
    for (size_t i=0; i<goodMatches.size(); i++)
    {
        // query 是第一个, train 是第二个
        cv::Point2f p = frame1.kp[goodMatches[i].queryIdx].pt;
        // 获取d是要小心！x是向右的，y是向下的，所以y才是行，x是列！
        ushort d = frame1.depth.ptr<ushort>( int(p.y) )[ int(p.x) ];
        if (d == 0)
            continue;
        pts_img.push_back( cv::Point2f( frame2.kp[goodMatches[i].trainIdx].pt ) );

        // 将(u,v,d)转成(x,y,z)
        cv::Point3f pt ( p.x, p.y, d );
        cv::Point3f pd = point2dTo3d( pt, camera );
        pts_obj.push_back( pd );
    }

    double camera_matrix_data[3][3] = {
        {camera.fx, 0, camera.cx},
        {0, camera.fy, camera.cy},
        {0, 0, 1}
    };

    cout<<"solving pnp"<<endl;
    // 构建相机矩阵
    cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
    cv::Mat rvec, tvec, inliers;
    // 求解pnp
    cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, 100, inliers );

    RESULT_OF_PNP result;
    result.rvec = rvec;
    result.tvec = tvec;
    result.inliers = inliers.rows;

    return result;
}



cv::Point3f point2dTo3d( cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    cv::Point3f p; // 3D 点
    p.z = double( point.z ) / camera.scale;
    p.x = ( point.x - camera.cx) * p.z / camera.fx;
    p.y = ( point.y - camera.cy) * p.z / camera.fy;
    return p;
}



int main(int argc, char* argv[])
{  


    camera.fx = 525.0;
    camera.fy = 525.0;
    camera.cx = 319.5;
    camera.cy = 239.5;
    camera.scale = 5000;
    init();

    return 0;
}

#include <fstream>
#include <string>

#include <ros/ros.h>
// #include <sensor_msgs/Imu.h>
#include <image_transport/image_transport.h>
// #include "semslam_msgs/ObjectDetection.h"

#include <Eigen/Core>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// #include <boost/date_time/posix_time/posix_time.hpp>

void
subscriberConnectedCallback(const ros::SingleSubscriberPublisher& ssp)
{
    ROS_WARN_STREAM("[KITTI Data] Subscriber " << ssp.getSubscriberName()
                                               << " connected to topic "
                                               << ssp.getTopic() << ".");
}

void
imageSubscriberConnectedCallback(
  const image_transport::SingleSubscriberPublisher& ssp)
{
    ROS_WARN_STREAM("[KITTI Data] Subscriber " << ssp.getSubscriberName()
                                               << " connected to topic "
                                               << ssp.getTopic() << ".");
}

int
main(int argc, char* argv[])
{
    ros::init(argc, argv, "kitti_data");
    ros::NodeHandle nh;

    ros::NodeHandle data_nh("~");

    image_transport::ImageTransport it(nh);

    boost::function<void(const image_transport::SingleSubscriberPublisher&)>
      image_fn = imageSubscriberConnectedCallback;

    image_transport::Publisher cam0_pub =
      it.advertise("cam0/image_raw",
                   1000,
                   (image_transport::SubscriberStatusCallback)
                     imageSubscriberConnectedCallback);
    ros::Publisher cam0_info_pub =
      nh.advertise<sensor_msgs::CameraInfo>("cam0/camera_info", 10);

    image_transport::Publisher cam1_pub =
      it.advertise("cam1/image_raw", 1000, image_fn, image_fn);
    ros::Publisher cam1_info_pub =
      nh.advertise<sensor_msgs::CameraInfo>("cam1/camera_info", 10);

    // ros::Publisher car_det_pub =
    // nh.advertise<semslam_msgs::ObjectDetection>("/object_detector/car/detections",
    // 1000, (ros::SubscriberStatusCallback)subscriberConnectedCallback);
    // ros::Publisher window_det_pub =
    // nh.advertise<semslam_msgs::ObjectDetection>("/object_detector/window/detections",
    // 1000);

    // ros::Publisher imu_pub = nh.advertise<sensor_msgs::Imu>("imu0", 1000);

    // delay
    double delay_secs = 1.0;
    data_nh.getParam("delay", delay_secs);

    std::string kitti_dir;
    data_nh.getParam("data_dir", kitti_dir);

    ROS_INFO_STREAM("KITTI cam0 has " << cam0_pub.getNumSubscribers()
                                      << " subscribers.");

    ROS_INFO_STREAM("Waiting for " << delay_secs
                                   << " seconds before publishing data.");
    ros::Duration(delay_secs).sleep();

    ROS_INFO_STREAM("KITTI cam0 has " << cam0_pub.getNumSubscribers()
                                      << " subscribers.");

    double rate;
    if (!data_nh.getParam("rate", rate)) {
        rate = 1.0;
    }

    double t_start, t_end;

    if (!data_nh.getParam("t_start", t_start)) {
        t_start = 0;
    }

    if (!data_nh.getParam("t_end", t_end)) {
        t_end = std::numeric_limits<double>::infinity();
    }

    // double t_start = 11.8;
    // double t_end = 13.3;

    // std::string
    // kitti_dir("/home/sean/data/raw_data/2011_09_30/2011_09_30_drive_0020_sync");
    // std::string
    // kitti_dir("/home/sean/data/raw_data/2011_09_26/2011_09_26_drive_0014_sync");
    // std::string kitti_dir("/home/sean/data/kitti/sequences/05");
    // std::string kitti_dir("/media/sean/My
    // Book/ubuntu_home/data/dataset/sequences/05"); std::string
    // kitti_dir("/media/sean/My Book/ubuntu_home/data/dataset/sequences/05");
    // std::string kitti_dir("/media/sean/My
    // Book/ubuntu_home/data/dataset/sequences/07");

    std::string cam_ts_filename = kitti_dir + "/times.txt";
    std::ifstream cam_ts_file(cam_ts_filename);

    if (!cam_ts_file) {
        ROS_ERROR_STREAM("unable to open file " << cam_ts_filename);
        exit(1);
    }

    // read camera times
    // std::vector<ros::Time> cam_ts;
    std::vector<double> cam_ts;
    while (cam_ts_file.good()) {
        double t;
        cam_ts_file >> t;
        cam_ts.push_back(t / rate);

        /*
        std::string ts;
        std::getline(cam_ts_file, ts);

        if (ts.length() > 0) {
                ros::Time t_ros =
        ros::Time::fromBoost(boost::posix_time::time_from_string(ts));
                cam_ts.push_back(t_ros);
        }
        */
    }

    ROS_INFO_STREAM(
      cam_ts.size()
      << " camera msmts"); // and " << imu_ts.size() << " imu msmts.");

    // build camera info msg
    sensor_msgs::CameraInfo info0_msg;
    sensor_msgs::CameraInfo info1_msg;
    // double cam_fx = 721.5377;
    // double cam_fy = 721.5377;
    // double cam_cx = 609.5593;
    // double cam_cy = 172.854;

    info0_msg.height = 370;
    info0_msg.width = 1226;
    info0_msg.distortion_model = "plumb_bob";
    info0_msg.binning_x = 1;
    info0_msg.binning_y = 1;

    // read calibration information
    std::string cam_calib_filename = kitti_dir + "/calib.txt";
    std::ifstream cam_calib_file(cam_calib_filename);

    if (!cam_calib_file) {
        ROS_ERROR_STREAM("[KITTI Data] Unable to open file "
                         << cam_calib_filename);
        exit(1);
    }

    std::string ln;
    std::string token;

    Eigen::Vector3d t0, t1;

    while (std::getline(cam_calib_file, ln)) {
        std::stringstream ss(ln);

        ss >> token;

        /*
                        if (token.find("K_00") == 0) {
                                for (int i = 0; i < 9; ++i) {
                                        ss >> info0_msg.K[i];
                                }
                        } else if (token.find("D_00") == 0) {
                                info0_msg.D.resize(5);
                                for (int i = 0; i < 5; ++i) {
                                        ss >> info0_msg.D[i];
                                }
                        } else if (token.find("P_rect_00") == 0) {
                                for (int i = 0; i < 12; ++i) {
                                        ss >> info0_msg.P[i];
                                }
                        } else if (token.find("K_01") == 0) {
                                for (int i = 0; i < 9; ++i) {
                                        ss >> info1_msg.K[i];
                                }
                        } else if (token.find("D_01") == 0) {
                                info1_msg.D.resize(5);
                                for (int i = 0; i < 5; ++i) {
                                        ss >> info1_msg.D[i];
                                }
                        } else if (token.find("P_rect_01") == 0) {
                                for (int i = 0; i < 12; ++i) {
                                        ss >> info1_msg.P[i];
                                }
                        } else if (token.find("T_00") == 0) {
                                for (int i = 0; i < 3; ++i) {
                                        ss >> t0(i);
                                }
                        } else if (token.find("T_01") == 0) {
                                for (int i = 0; i < 3; ++i) {
                                        ss >> t1(i);
                                }
                        }
        */
        if (token.find("P2") == 0) {
            for (int i = 0; i < 12; ++i) {
                ss >> info0_msg.P[i];
            }
        } else if (token.find("P3") == 0) {
            for (int i = 0; i < 12; ++i) {
                ss >> info1_msg.P[i];
            }
        }
    }

    std::cout << "Camera 0 P = {";
    for (double x : info0_msg.P) {
        std::cout << x << ", ";
    }
    std::cout << "}" << std::endl;
    std::cout << "Camera 1 P = {";
    for (double k : info1_msg.P) {
        std::cout << k << ", ";
    }
    std::cout << "}" << std::endl;
    /*
            std::cout << "Camera 0 D = {";
            for (double k : info0_msg.D) {
                    std::cout << k << ", ";
            }
            std::cout << "}" << std::endl;

            std::cout << "Camera 0 T = " << t0.transpose() << std::endl;


            std::cout << "Camera 1 D = {";
            for (double k : info1_msg.D) {
                    std::cout << k << ", ";
            }
            std::cout << "}" << std::endl;
            std::cout << "Camera 1 T = " << t1.transpose() << std::endl;
    */

    std::string car_fmt_str = kitti_dir + "/det_2/car/%06d.txt";
    std::string window_fmt_str = kitti_dir + "/det_2/window/%06d.txt";

    std::string cam0_fmt_str = kitti_dir + "/image_2/%06d.png";
    std::string cam1_fmt_str = kitti_dir + "/image_3/%06d.png";
    size_t cam_next = 0;

    // wait until there's a subscriber
    // while (ros::ok() && (cam0_pub.getNumSubscribers() == 0 &&
    // cam1_pub.getNumSubscribers() == 0)) { 	ROS_INFO_STREAM("[KITTI_data_node]
    // No subscribers, trying again in 1 second...");
    // 	ros::Duration(1.0).sleep();
    // }

    // Find first index
    while (cam_ts[cam_next] * rate < t_start)
        cam_next++;

    double t_offset = ros::Time::now().toSec() - cam_ts[cam_next];
    ROS_INFO_STREAM("T OFFSET = " << t_offset);
    while (ros::ok() && cam_next < cam_ts.size()) {

        if (cam_ts[cam_next] * rate > t_end) {
            break;
        }

        // read & publish cam
        char fname[1024];
        sprintf(fname, cam0_fmt_str.c_str(), cam_next);
        cv::Mat img0 = cv::imread(fname);

        sprintf(fname, cam1_fmt_str.c_str(), cam_next);
        cv::Mat img1 = cv::imread(fname);

        cv_bridge::CvImage img0_msg;
        img0_msg.header.stamp = ros::Time(cam_ts[cam_next] * rate);
        img0_msg.image = img0;
        img0_msg.encoding = sensor_msgs::image_encodings::BGR8;

        cv_bridge::CvImage img1_msg;
        img1_msg.header.stamp = ros::Time(cam_ts[cam_next] * rate);
        img1_msg.image = img1;
        img1_msg.encoding = sensor_msgs::image_encodings::BGR8;

        // Read & publish object detections
        // semslam_msgs::ObjectDetection car_det_msg, window_det_msg;
        // car_det_msg.header.stamp = ros::Time(cam_ts[cam_next] * rate);
        // car_det_msg.header.seq = cam_next;
        // window_det_msg.header.stamp = ros::Time(cam_ts[cam_next] * rate);
        // window_det_msg.header.seq = cam_next;

        // sprintf(fname, car_fmt_str.c_str(), cam_next);
        // std::ifstream car_det_file(fname);
        // std::string ln;
        // // each line is [left top right bottom score]
        // while (std::getline(car_det_file, ln)) {
        // 	std::stringstream ss(ln);
        // 	double l, t, r, b, s;

        // 	ss >> l >> t >> r >> b >> s;

        // 	car_det_msg.left.push_back(l);
        // 	car_det_msg.top.push_back(t);
        // 	car_det_msg.right.push_back(r);
        // 	car_det_msg.bottom.push_back(b);
        // 	car_det_msg.score.push_back(s);
        // }

        // sprintf(fname, window_fmt_str.c_str(), cam_next);
        // std::ifstream win_det_file(fname);
        // while (std::getline(win_det_file, ln)) {
        // 	std::stringstream ss(ln);
        // 	double l, t, r, b, s;
        // 	ss >> l >> t >> r >> b >> s;

        // 	window_det_msg.left.push_back(l);
        // 	window_det_msg.top.push_back(t);
        // 	window_det_msg.right.push_back(r);
        // 	window_det_msg.bottom.push_back(b);
        // 	window_det_msg.score.push_back(s);
        // }

        // ROS_INFO_STREAM("Publishing object detections for file " << cam_next
        // << " (t = " << cam_ts[cam_next]*rate << ")");

        // wait appropriate amount of time
        // if (imu_next > 0 || cam_next > 0) {

        double delay = cam_ts[cam_next] + t_offset - ros::Time::now().toSec();
        if (cam_next > 0 && delay > 0) {
            // ROS_INFO_STREAM("Sleeping for " << delay << " seconds.");
            ros::Duration(cam_ts[cam_next] + t_offset -
                          ros::Time::now().toSec())
              .sleep();
        }

        // ROS_INFO_STREAM("[Kitti data node] publishing t = " <<
        // cam_ts[cam_next]); ROS_INFO_STREAM("Data = " << data);

        cam0_pub.publish(img0_msg.toImageMsg());
        cam1_pub.publish(img1_msg.toImageMsg());

        // car_det_pub.publish(car_det_msg);
        // window_det_pub.publish(window_det_msg);

        info0_msg.header.stamp = ros::Time(cam_ts[cam_next] * rate);
        cam0_info_pub.publish(info0_msg);

        info1_msg.header.stamp = ros::Time(cam_ts[cam_next] * rate);
        cam1_info_pub.publish(info1_msg);

        // last_t = cam_ts[cam_next];
        cam_next++;
    }
}
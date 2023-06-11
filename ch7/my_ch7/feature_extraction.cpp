#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "usage : feature_extraction img1 img2" << std::endl;
        return 1;
    }

    // reading image
    //cv::Mat img_1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    //cv::Mat img_2 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
    cv::Mat img_1 = cv::imread(argv[1], 1);
    cv::Mat img_2 = cv::imread(argv[2], 1);

    // initialize
    std::vector<cv::KeyPoint> key_points_1;
    std::vector<cv::KeyPoint> key_points_2;
    cv::Mat descriptors_1;
    cv::Mat descriptors_2;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);

    orb->detect(img_1, key_points_1);
    orb->detect(img_2, key_points_2);

    orb->compute(img_1, key_points_1, descriptors_1);
    orb->compute(img_2, key_points_2, descriptors_2);

    cv::Mat out_img_1;
    cv::drawKeypoints(img_1, key_points_1, out_img_1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("orb特征点", out_img_1);

    std::vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(descriptors_1, descriptors_2, matches);

    double min_dis = 10000.0;
    double max_dis = 0.0;

    for (int i = 0; i < descriptors_1.rows; ++i) {
        double dis = matches[i].distance;
        if (dis < min_dis) min_dis = dis;
        if (dis > max_dis) max_dis = dis;
    }

    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; ++i) {
        if (matches[i].distance < std::max(2 * min_dis, 30.0))  {
            good_matches.emplace_back(matches[i]);
        }
    }

    cv::Mat img_match;
    cv::Mat img_goodmatch;
    drawMatches(img_1, key_points_1, img_2, key_points_2, matches, img_match);
    drawMatches(img_1, key_points_1, img_2, key_points_2, good_matches, img_goodmatch);
    cv::imshow("所有匹配点对", img_match);
    cv::imshow("优化后匹配点对", img_goodmatch);
    cv::waitKey(0);

    return 0;
}

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/version.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/features.hpp>
#else
#include <opencv2/features2d.hpp>
#endif
#include <string>
#include <vector>

namespace lab2 {

struct TrackerConfig {
    int maxFeatures = 2500;
    double ratioThreshold = 0.75;
    double ransacThreshold = 3.0;
    int minMatches = 10;
    int minInliers = 8;
    double minInlierRatio = 0.4;
    bool enableOpticalFlow = true;
};

struct TrackingResult {
    std::vector<cv::Point2f> corners;
    std::string status = "not_found";
    std::string method = "none";
    int matches = 0;
    int inliers = 0;
    double inlierRatio = 0.0;
    double medianError = -1.0;
    double elapsedMs = 0.0;
};

// Fixed-reference re-detection corrects drift and permits recovery after loss.
class PlanarTracker {
public:
    explicit PlanarTracker(const cv::Mat& reference, const TrackerConfig& config = {});
    TrackingResult locate(const cv::Mat& frame);
    int referenceKeypoints() const { return static_cast<int>(referencePoints_.size()); }

private:
    TrackerConfig config_;
    cv::Size referenceSize_;
    cv::Ptr<cv::SIFT> detector_;
    cv::BFMatcher matcher_{cv::NORM_L2};
    std::vector<cv::KeyPoint> referencePoints_;
    cv::Mat referenceDescriptors_;
    std::vector<cv::Point2f> referenceCorners_;
    cv::Mat previousGray_;
    std::vector<cv::Point2f> flowSource_, flowDestination_;
    TrackingResult estimate(const std::vector<cv::Point2f>& source,
                            const std::vector<cv::Point2f>& destination,
                            const cv::Size& frameSize, cv::Mat& homography,
                            bool opticalFlow = false) const;
    void seedFlow(const cv::Mat& gray, const cv::Mat& homography,
                  const std::vector<cv::Point2f>& corners);
};

} // namespace lab2

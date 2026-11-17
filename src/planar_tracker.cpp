#include "planar_tracker.hpp"

#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry.hpp>
#else
#include <opencv2/calib3d.hpp>
#endif
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>
#include <stdexcept>
#include <utility>

namespace lab2 {
namespace {
cv::Mat grayscale(const cv::Mat& image) {
    if (image.empty() || image.depth() != CV_8U)
        throw std::invalid_argument("Expected a nonempty 8-bit image.");
    cv::Mat gray;
    if (image.channels() == 1) gray = image;
    else if (image.channels() == 3) cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else if (image.channels() == 4) cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    else throw std::invalid_argument("Expected 1, 3 or 4 image channels.");
    return gray;
}

double hullArea(const std::vector<cv::Point2f>& points) {
    std::vector<cv::Point2f> hull;
    cv::convexHull(points, hull);
    return std::abs(cv::contourArea(hull));
}

bool validPolygon(const std::vector<cv::Point2f>& corners, const cv::Size& frameSize) {
    for (const auto& p : corners) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) ||
            p.x < -2.0 * frameSize.width || p.x > 3.0 * frameSize.width ||
            p.y < -2.0 * frameSize.height || p.y > 3.0 * frameSize.height)
            return false;
    }
    if (!cv::isContourConvex(corners)) return false;
    // Reject mirrored orientation and implausible quadrilaterals.
    const double area = cv::contourArea(corners, true);
    if (area < 64.0 || area > 4.0 * frameSize.area()) return false;
    for (std::size_t i = 0; i < corners.size(); ++i)
        if (cv::norm(corners[i] - corners[(i + 1) % corners.size()]) < 6.0) return false;
    const std::vector<cv::Point2f> screen = {
        {0, 0}, {static_cast<float>(frameSize.width - 1), 0},
        {static_cast<float>(frameSize.width - 1), static_cast<float>(frameSize.height - 1)},
        {0, static_cast<float>(frameSize.height - 1)}};
    std::vector<cv::Point2f> intersection;
    return cv::intersectConvexConvex(corners, screen, intersection, true) > 0.1 * area;
}
} // namespace

PlanarTracker::PlanarTracker(const cv::Mat& reference, const TrackerConfig& config)
    : config_(config), referenceSize_(reference.size()) {
    if (config.maxFeatures < 16 || !(config.ratioThreshold > 0.0 && config.ratioThreshold < 1.0) ||
        !(config.ransacThreshold > 0.0 && std::isfinite(config.ransacThreshold)) ||
        config.minMatches < 4 || config.minInliers < 4 ||
        !(config.minInlierRatio > 0.0 && config.minInlierRatio <= 1.0))
        throw std::invalid_argument("Invalid tracker configuration.");
    if (reference.cols < 8 || reference.rows < 8)
        throw std::invalid_argument("Reference image is empty or too small.");
    detector_ = cv::SIFT::create(config.maxFeatures, 3, 0.02);
    const cv::Mat referenceGray = grayscale(reference);
    // Downsampled templates preserve the coarse features that survive when the
    // object occupies only a small part of the scene. All points are converted
    // back to the original template coordinate system before fitting H.
    for (const double scale : {1.0, 0.5, 0.25, 0.125}) {
        cv::Mat level;
        if (scale == 1.0) level = referenceGray;
        else cv::resize(referenceGray, level, cv::Size(), scale, scale, cv::INTER_AREA);
        if (level.cols < 24 || level.rows < 24) break;
        std::vector<cv::KeyPoint> levelPoints;
        cv::Mat levelDescriptors;
        detector_->detectAndCompute(level, cv::noArray(), levelPoints, levelDescriptors);
        const float scaleX = static_cast<float>(reference.cols) / level.cols;
        const float scaleY = static_cast<float>(reference.rows) / level.rows;
        for (auto point : levelPoints) {
            // cv::resize maps pixel centres using (x + 0.5) / scale - 0.5.
            point.pt.x = (point.pt.x + 0.5F) * scaleX - 0.5F;
            point.pt.y = (point.pt.y + 0.5F) * scaleY - 0.5F;
            referencePoints_.push_back(point);
        }
        if (!levelDescriptors.empty()) referenceDescriptors_.push_back(levelDescriptors);
    }
    if (referenceDescriptors_.empty() || referenceKeypoints() < config.minMatches)
        throw std::invalid_argument("Too few reference keypoints. Choose a textured object/ROI.");
    referenceCorners_ = {{0, 0}, {static_cast<float>(reference.cols - 1), 0},
                         {static_cast<float>(reference.cols - 1), static_cast<float>(reference.rows - 1)},
                         {0, static_cast<float>(reference.rows - 1)}};
}

TrackingResult PlanarTracker::locate(const cv::Mat& frame) {
    const auto started = std::chrono::steady_clock::now();
    const cv::Mat gray = grayscale(frame);
    std::vector<cv::KeyPoint> points;
    cv::Mat descriptors;
    detector_->detectAndCompute(gray, cv::noArray(), points, descriptors);

    std::vector<std::vector<cv::DMatch>> neighbors;
    if (descriptors.rows >= 2)
        matcher_.knnMatch(referenceDescriptors_, descriptors, neighbors, 2);
    std::vector<cv::DMatch> candidates;
    for (const auto& pair : neighbors)
        if (pair.size() == 2 && pair[0].distance < config_.ratioThreshold * pair[1].distance)
            candidates.push_back(pair[0]);
    std::sort(candidates.begin(), candidates.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
        return a.distance < b.distance;
    });
    std::set<int> usedScenePoints;
    std::vector<cv::Point2f> source, destination;
    for (const auto& match : candidates) {
        // One scene feature cannot vote for several different template points.
        if (!usedScenePoints.insert(match.trainIdx).second) continue;
        source.push_back(referencePoints_[match.queryIdx].pt);
        destination.push_back(points[match.trainIdx].pt);
    }
    cv::Mat homography;
    TrackingResult result = estimate(source, destination, frame.size(), homography);
    if (descriptors.rows < 2) result.status = "no_features";
    if (!result.corners.empty()) result.method = "sift";

    // Fresh SIFT matches always take priority. LK is a measured correspondence
    // fallback when the object becomes too small to re-detect enough SIFT points.
    if (config_.enableOpticalFlow && result.corners.empty() && !previousGray_.empty() && previousGray_.size() == gray.size() &&
        flowSource_.size() >= static_cast<std::size_t>(config_.minMatches)) {
        std::vector<cv::Point2f> forward, backward;
        std::vector<unsigned char> forwardStatus, backwardStatus;
        std::vector<float> forwardError, backwardError;
        const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
        cv::calcOpticalFlowPyrLK(previousGray_, gray, flowDestination_, forward,
            forwardStatus, forwardError, cv::Size(21, 21), 3, criteria);
        cv::calcOpticalFlowPyrLK(gray, previousGray_, forward, backward,
            backwardStatus, backwardError, cv::Size(21, 21), 3, criteria);
        std::vector<cv::Point2f> flowSource, flowDestination;
        for (std::size_t i = 0; i < forward.size(); ++i) {
            const auto& p = forward[i];
            if (forwardStatus[i] && backwardStatus[i] && forwardError[i] < 20.0F &&
                std::isfinite(p.x) && std::isfinite(p.y) && p.x >= 0 && p.y >= 0 &&
                p.x < gray.cols && p.y < gray.rows &&
                cv::norm(flowDestination_[i] - backward[i]) < 1.0) {
                flowSource.push_back(flowSource_[i]);
                flowDestination.push_back(p);
            }
        }
        cv::Mat flowHomography;
        TrackingResult flow = estimate(flowSource, flowDestination, frame.size(), flowHomography, true);
        if (!flow.corners.empty()) {
            flow.method = "lk";
            result = std::move(flow);
            homography = flowHomography;
        }
    }
    if (config_.enableOpticalFlow && !result.corners.empty()) seedFlow(gray, homography, result.corners);
    else {
        previousGray_.release();
        flowSource_.clear();
        flowDestination_.clear();
    }
    result.elapsedMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    return result;
}

TrackingResult PlanarTracker::estimate(const std::vector<cv::Point2f>& source,
    const std::vector<cv::Point2f>& destination, const cv::Size& frameSize,
    cv::Mat& homography, bool opticalFlow) const {
    TrackingResult result;
    auto finish = [&](const std::string& status) { result.status = status; return result; };
    result.matches = static_cast<int>(source.size());
    if (result.matches < config_.minMatches) return finish("few_matches");

    cv::Mat mask;
    const double threshold = opticalFlow ? std::min(1.5, config_.ransacThreshold) : config_.ransacThreshold;
    homography = cv::findHomography(source, destination, cv::RANSAC, threshold, mask, 2000, 0.995);
    if (homography.empty()) return finish("no_homography");
    result.inliers = cv::countNonZero(mask);
    result.inlierRatio = static_cast<double>(result.inliers) / result.matches;
    const double minimumRatio = opticalFlow ? std::max(0.6, config_.minInlierRatio) : config_.minInlierRatio;
    if (result.inliers < config_.minInliers || result.inlierRatio < minimumRatio)
        return finish("few_inliers");

    std::vector<cv::Point2f> corners, projected, inlierSource, inlierDestination;
    cv::perspectiveTransform(referenceCorners_, corners, homography);
    if (!validPolygon(corners, frameSize)) return finish("invalid_polygon");
    cv::perspectiveTransform(source, projected, homography);
    std::vector<double> errors;
    for (std::size_t i = 0; i < source.size(); ++i) {
        if (!mask.at<unsigned char>(static_cast<int>(i))) continue;
        inlierSource.push_back(source[i]);
        inlierDestination.push_back(destination[i]);
        errors.push_back(cv::norm(projected[i] - destination[i]));
    }
    // Reject a homography estimated from a tiny patch of the object.
    if (hullArea(inlierSource) < 0.02 * referenceSize_.area() ||
        hullArea(inlierDestination) < 0.08 * std::abs(cv::contourArea(corners)))
        return finish("clustered_inliers");
    std::sort(errors.begin(), errors.end());
    const std::size_t middle = errors.size() / 2;
    result.medianError = errors.size() % 2 ? errors[middle] : (errors[middle - 1] + errors[middle]) / 2;
    if (!std::isfinite(result.medianError) || result.medianError > threshold)
        return finish("large_error");
    result.corners = std::move(corners);
    return finish("tracked");
}

void PlanarTracker::seedFlow(const cv::Mat& gray, const cv::Mat& homography,
    const std::vector<cv::Point2f>& corners) {
    cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1);
    std::vector<cv::Point> polygon;
    for (const auto& p : corners) polygon.emplace_back(cvRound(p.x), cvRound(p.y));
    cv::fillConvexPoly(mask, polygon, cv::Scalar(255));
    cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
    cv::goodFeaturesToTrack(gray, flowDestination_, 150, 0.01, 4.0, mask);
    flowSource_.clear();
    if (!flowDestination_.empty()) {
        cv::Mat inverse;
        if (cv::invert(homography, inverse) != 0.0)
            cv::perspectiveTransform(flowDestination_, flowSource_, inverse);
        else flowDestination_.clear();
    }
    previousGray_ = gray.clone();
}
} // namespace lab2

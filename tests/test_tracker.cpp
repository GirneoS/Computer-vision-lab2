#include "planar_tracker.hpp"

#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry.hpp>
#endif
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kReferenceWidth = 420;
constexpr int kReferenceHeight = 300;
const cv::Size kFrameSize(800, 600);

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// All scene content is generated locally. Different seeds produce unrelated
// textures, including the text, so a distractor does not reuse reference labels.
cv::Mat makeTexture(std::uint64_t seed) {
    cv::RNG random(seed);
    cv::Mat image(kReferenceHeight, kReferenceWidth, CV_8UC3);
    random.fill(image, cv::RNG::NORMAL, cv::Scalar::all(125), cv::Scalar::all(14));
    cv::GaussianBlur(image, image, cv::Size(3, 3), 0.6);

    for (int index = 0; index < 180; ++index) {
        const cv::Point center(random.uniform(12, image.cols - 12),
                               random.uniform(12, image.rows - 12));
        const cv::Scalar color(random.uniform(15, 240), random.uniform(15, 240),
                               random.uniform(15, 240));
        const int radius = random.uniform(3, 17);
        switch (index % 3) {
        case 0:
            cv::circle(image, center, radius, color, index % 2 ? 2 : -1,
                       cv::LINE_AA);
            break;
        case 1:
            cv::rectangle(image, center - cv::Point(radius, radius),
                          center + cv::Point(radius, radius), color,
                          index % 2 ? 2 : -1, cv::LINE_AA);
            break;
        default:
            cv::line(image, center - cv::Point(radius, radius),
                     center + cv::Point(radius, -radius), color, 2,
                     cv::LINE_AA);
            break;
        }
    }

    for (int row = 0; row < 7; ++row) {
        std::string label;
        for (int character = 0; character < 15; ++character) {
            label += static_cast<char>(random.uniform('A', 'Z' + 1));
        }
        const cv::Point position(12, 30 + row * 40);
        cv::putText(image, label, position, cv::FONT_HERSHEY_SIMPLEX, 0.62,
                    cv::Scalar(250, 250, 250), 3, cv::LINE_AA);
        cv::putText(image, label, position, cv::FONT_HERSHEY_SIMPLEX, 0.62,
                    cv::Scalar(10, 10, 10), 1, cv::LINE_AA);
    }
    return image;
}

std::vector<cv::Point2f> referenceCorners() {
    return {{0.0F, 0.0F},
            {static_cast<float>(kReferenceWidth - 1), 0.0F},
            {static_cast<float>(kReferenceWidth - 1),
             static_cast<float>(kReferenceHeight - 1)},
            {0.0F, static_cast<float>(kReferenceHeight - 1)}};
}

std::vector<cv::Point2f> regularCorners() {
    return {{140.0F, 90.0F}, {630.0F, 115.0F},
            {610.0F, 460.0F}, {105.0F, 430.0F}};
}

cv::Mat makeScene(const cv::Mat& object,
                  const std::vector<cv::Point2f>& destination) {
    cv::Mat scene(kFrameSize, CV_8UC3, cv::Scalar(55, 65, 70));
    // A background with its own edges makes this more than a black-canvas test.
    for (int y = 25; y < scene.rows; y += 65) {
        cv::line(scene, cv::Point(0, y), cv::Point(scene.cols - 1, y + 20),
                 cv::Scalar(65, 75, 80), 2, cv::LINE_AA);
    }
    const cv::Mat transform =
        cv::getPerspectiveTransform(referenceCorners(), destination);
    cv::Mat warped;
    cv::warpPerspective(object, warped, transform, kFrameSize, cv::INTER_LINEAR);
    cv::Mat mask(object.size(), CV_8UC1, cv::Scalar(255));
    cv::Mat warpedMask;
    cv::warpPerspective(mask, warpedMask, transform, kFrameSize,
                        cv::INTER_NEAREST);
    warped.copyTo(scene, warpedMask);
    return scene;
}

void checkDetection(const lab2::TrackingResult& result,
                    const std::vector<cv::Point2f>& expected,
                    double maximumCornerError = 6.0) {
    require(result.corners.size() == 4,
            "Object was not localized; status=" + result.status);
    const lab2::TrackerConfig defaults;
    require(result.inliers >= defaults.minInliers,
            "Accepted detection has too few inliers");
    require(result.matches >= result.inliers,
            "Inlier count exceeds the match count");
    require(std::isfinite(result.inlierRatio) &&
                result.inlierRatio >= defaults.minInlierRatio &&
                result.inlierRatio <= 1.0,
            "Accepted detection has an invalid inlier ratio");
    require(std::isfinite(result.medianError) && result.medianError >= 0.0,
            "Invalid reprojection-error statistic");
    require(std::isfinite(result.elapsedMs) && result.elapsedMs >= 0.0,
            "Invalid processing-time statistic");

    double worstError = 0.0;
    for (std::size_t corner = 0; corner < expected.size(); ++corner) {
        require(std::isfinite(result.corners[corner].x) &&
                    std::isfinite(result.corners[corner].y),
                "A projected corner is not finite");
        worstError = std::max(
            worstError, cv::norm(result.corners[corner] - expected[corner]));
    }
    std::ostringstream message;
    message << "Maximum corner error " << std::fixed << std::setprecision(3)
            << worstError << " px exceeds " << maximumCornerError << " px";
    require(worstError <= maximumCornerError, message.str());
}

void expectInvalidReference(const cv::Mat& reference) {
    bool rejected = false;
    try {
        const lab2::PlanarTracker tracker(reference);
        (void)tracker;
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "Invalid reference must throw std::invalid_argument");
}

void testInvalidReference() {
    expectInvalidReference(cv::Mat());
    expectInvalidReference(
        cv::Mat(kReferenceHeight, kReferenceWidth, CV_8UC3, cv::Scalar::all(127)));
}

void testKnownHomography() {
    const cv::Mat reference = makeTexture(918273);
    lab2::PlanarTracker tracker(reference);
    const auto corners = regularCorners();
    checkDetection(tracker.locate(makeScene(reference, corners)), corners);
}

void testScaleRotationPerspective() {
    const cv::Mat reference = makeTexture(918273);
    lab2::PlanarTracker tracker(reference);
    const std::vector<cv::Point2f> corners = {
        {330.0F, 85.0F}, {590.0F, 270.0F},
        {430.0F, 495.0F}, {190.0F, 330.0F}};
    checkDetection(tracker.locate(makeScene(reference, corners)), corners);
}

void testBlankAndDistractorRejection() {
    const cv::Mat reference = makeTexture(918273);
    lab2::PlanarTracker tracker(reference);
    const cv::Mat blank(kFrameSize, CV_8UC3, cv::Scalar::all(100));
    require(tracker.locate(blank).corners.empty(),
            "A blank frame produced a false detection");
    const cv::Mat distractor = makeScene(makeTexture(772233), regularCorners());
    require(tracker.locate(distractor).corners.empty(),
            "An unrelated textured object produced a false detection");
}

void testDisappearanceAndReappearance() {
    const cv::Mat reference = makeTexture(918273);
    lab2::PlanarTracker tracker(reference);
    checkDetection(tracker.locate(makeScene(reference, regularCorners())),
                   regularCorners());
    const cv::Mat blank(kFrameSize, CV_8UC3, cv::Scalar::all(100));
    for (int frame = 0; frame < 3; ++frame) {
        require(tracker.locate(blank).corners.empty(),
                "The tracker retained a stale box after disappearance");
    }
    const std::vector<cv::Point2f> returnedCorners = {
        {230.0F, 190.0F}, {650.0F, 150.0F},
        {690.0F, 465.0F}, {200.0F, 510.0F}};
    checkDetection(tracker.locate(makeScene(reference, returnedCorners)),
                   returnedCorners);
}

void testPartialOcclusion() {
    const cv::Mat reference = makeTexture(918273);
    lab2::PlanarTracker tracker(reference);
    const auto corners = regularCorners();
    cv::Mat scene = makeScene(reference, corners);
    // Hide approximately one fifth of the projected object's interior.
    cv::rectangle(scene, cv::Rect(315, 190, 165, 180),
                  cv::Scalar(35, 45, 55), cv::FILLED);
    checkDetection(tracker.locate(scene), corners, 7.0);
}

void testPartlyOutsideFrame() {
    const cv::Mat reference = makeTexture(918273);
    lab2::PlanarTracker tracker(reference);
    const std::vector<cv::Point2f> corners = {
        {-28.0F, 12.0F}, {390.0F, 35.0F},
        {370.0F, 345.0F}, {-42.0F, 310.0F}};
    checkDetection(tracker.locate(makeScene(reference, corners)), corners, 7.0);
}

void testOpticalFlowDuringGradualShrink() {
    const cv::Mat reference = makeTexture(918273);
    lab2::PlanarTracker tracker(reference);
    lab2::TrackerConfig detectionOnlyConfig;
    detectionOnlyConfig.enableOpticalFlow = false;
    lab2::PlanarTracker detectionOnly(reference, detectionOnlyConfig);
    constexpr int frameCount = 110;
    int foundFrames = 0;
    int flowFrames = 0;
    int flowOnlyFrames = 0;
    double worstSiftCornerError = 0.0;
    double worstFlowCornerError = 0.0;
    std::string firstError;
    for (int frame = 0; frame < frameCount; ++frame) {
        // The object shrinks continuously from 419x299 to about 50x36 pixels.
        // A small step between frames remains measurable by optical flow even
        // when independently re-detecting enough SIFT descriptors becomes hard.
        const double fraction = static_cast<double>(frame) / (frameCount - 1);
        const double scale = std::pow(0.12, fraction);
        const double angle = 0.12 + frame * 0.001;
        const cv::Point2f center(400.0F + 20.0F * static_cast<float>(std::sin(frame * 0.03)),
                                 300.0F + 15.0F * static_cast<float>(std::cos(frame * 0.03)));
        std::vector<cv::Point2f> corners;
        for (const auto& corner : referenceCorners()) {
            const double x = (corner.x - (kReferenceWidth - 1) / 2.0) * scale;
            const double y = (corner.y - (kReferenceHeight - 1) / 2.0) * scale;
            corners.emplace_back(
                center.x + static_cast<float>(x * std::cos(angle) - y * std::sin(angle)),
                center.y + static_cast<float>(x * std::sin(angle) + y * std::cos(angle)));
        }
        // warpPerspective's linear interpolation alone does not anti-alias a
        // strong reduction. Filter in reference coordinates before sampling so
        // tiny objects do not acquire artificial high-frequency flicker.
        cv::Mat filteredReference;
        const double sigma = 0.5 * std::sqrt(1.0 / (scale * scale) - 1.0);
        if (sigma > 0.05) {
            cv::GaussianBlur(reference, filteredReference, cv::Size(), sigma);
        } else {
            filteredReference = reference;
        }
        const cv::Mat scene = makeScene(filteredReference, corners);
        const auto result = tracker.locate(scene);
        if (!result.corners.empty()) {
            ++foundFrames;
            double cornerError = 0.0;
            for (std::size_t corner = 0; corner < corners.size(); ++corner) {
                cornerError = std::max(cornerError, cv::norm(result.corners[corner] - corners[corner]));
            }
            if (result.method == "lk") {
                worstFlowCornerError = std::max(worstFlowCornerError, cornerError);
            } else {
                worstSiftCornerError = std::max(worstSiftCornerError, cornerError);
            }
            try {
                checkDetection(result, corners, 7.0);
            } catch (const std::runtime_error& error) {
                if (firstError.empty()) {
                    firstError = "Shrink frame " + std::to_string(frame) +
                                 " (" + result.method + "): " + error.what();
                }
            }
        }
        if (result.method == "lk") {
            ++flowFrames;
            if (detectionOnly.locate(scene).corners.empty()) {
                ++flowOnlyFrames;
            }
        }
    }
    std::cout << "  gradual shrink: " << foundFrames << '/' << frameCount
              << " found, " << flowFrames << " LK, " << flowOnlyFrames
              << " LK detections without independent SIFT detection; worst corner error SIFT="
              << worstSiftCornerError << " px, LK=" << worstFlowCornerError << " px\n";
    require(foundFrames >= frameCount / 2,
            "Too little of the gradual-shrink sequence was tracked");
    require(flowFrames > 0, "The gradual-shrink test did not exercise optical flow");
    require(flowOnlyFrames > 0,
            "No frame demonstrated optical-flow recovery beyond independent SIFT");
    require(firstError.empty(), firstError);

    // Start from a confirmed detection so the blank frame also tests clearing
    // a populated LK state rather than merely staying lost after a tiny object.
    const cv::Mat restored = makeScene(reference, regularCorners());
    checkDetection(tracker.locate(restored), regularCorners());
    const cv::Mat blank(kFrameSize, CV_8UC3, cv::Scalar::all(100));
    require(tracker.locate(blank).corners.empty(),
            "Optical flow retained the object on the first blank frame");
    require(tracker.locate(blank).corners.empty(),
            "Optical flow retained state across consecutive blank frames");
    const auto recovered = tracker.locate(restored);
    checkDetection(recovered, regularCorners());
    require(recovered.method == "sift", "Recovery after loss must use fresh SIFT detection");
}

} // namespace

int main() {
    cv::setRNGSeed(20261117);
    cv::setNumThreads(1);
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"invalid reference", testInvalidReference},
        {"known homography", testKnownHomography},
        {"scale, rotation and perspective", testScaleRotationPerspective},
        {"blank and unrelated distractor", testBlankAndDistractorRejection},
        {"disappearance and reappearance", testDisappearanceAndReappearance},
        {"partial occlusion", testPartialOcclusion},
        {"partly outside frame", testPartlyOutsideFrame},
        {"optical flow during gradual shrink", testOpticalFlowDuringGradualShrink},
    };

    int failures = 0;
    for (const auto& test : tests) {
        try {
            test.second();
            std::cout << "PASS: " << test.first << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL: " << test.first << ": " << error.what() << '\n';
        }
    }
    std::cout << (tests.size() - static_cast<std::size_t>(failures)) << '/'
              << tests.size() << " tests passed\n";
    return failures == 0 ? 0 : 1;
}

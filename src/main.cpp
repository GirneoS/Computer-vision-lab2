#include "planar_tracker.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options {
    fs::path input;
    fs::path outputDir;
    std::string label = "Mona Lisa";
    cv::Rect roi;
    bool hasRoi = false;
    bool display = false;
    lab2::TrackerConfig tracker;
};

void printHelp() {
    std::cout
        << "Planar object tracking with SIFT and RANSAC homography\n\n"
        << "Usage: lab2_tracker --input VIDEO [options]\n\n"
        << "  --output-dir DIR       Results directory (default: results/<video stem>)\n"
        << "  --label TEXT           Object label (default: Mona Lisa; ASCII recommended)\n"
        << "  --roi X Y WIDTH HEIGHT Reference rectangle in the first frame\n"
        << "  --ratio NUMBER         Descriptor ratio threshold (0 < value < 1)\n"
        << "  --ransac NUMBER        RANSAC reprojection threshold in pixels (> 0)\n"
        << "  --min-matches NUMBER   Minimum number of tentative matches (>= 4)\n"
        << "  --min-inliers NUMBER   Minimum number of RANSAC inliers (>= 4)\n"
        << "  --display              Show the result; press Q or Escape to stop\n"
        << "  --no-flow              Disable LK fallback (SIFT-only comparison)\n"
        << "  --help                 Show this help\n\n"
        << "Outputs: annotated.avi (MJPG), tracking.csv, summary.json, contact_sheet.jpg\n";
}

std::string nextValue(int argc, char** argv, int& i, const std::string& option) {
    if (++i >= argc) {
        throw std::invalid_argument("Missing value for " + option);
    }
    return argv[i];
}

int parseInt(const std::string& text, const std::string& option) {
    try {
        std::size_t used = 0;
        const int result = std::stoi(text, &used);
        if (used == text.size()) {
            return result;
        }
    } catch (const std::exception&) {
    }
    throw std::invalid_argument("Invalid integer for " + option + ": " + text);
}

double parseDouble(const std::string& text, const std::string& option) {
    try {
        std::size_t used = 0;
        const double result = std::stod(text, &used);
        if (used == text.size() && std::isfinite(result)) {
            return result;
        }
    } catch (const std::exception&) {
    }
    throw std::invalid_argument("Invalid number for " + option + ": " + text);
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input") {
            options.input = nextValue(argc, argv, i, arg);
        } else if (arg == "--output-dir") {
            options.outputDir = nextValue(argc, argv, i, arg);
        } else if (arg == "--label") {
            options.label = nextValue(argc, argv, i, arg);
        } else if (arg == "--roi") {
            const int x = parseInt(nextValue(argc, argv, i, arg), arg);
            const int y = parseInt(nextValue(argc, argv, i, arg), arg);
            const int width = parseInt(nextValue(argc, argv, i, arg), arg);
            const int height = parseInt(nextValue(argc, argv, i, arg), arg);
            options.roi = cv::Rect(x, y, width, height);
            options.hasRoi = true;
        } else if (arg == "--ratio") {
            options.tracker.ratioThreshold = parseDouble(nextValue(argc, argv, i, arg), arg);
        } else if (arg == "--ransac") {
            options.tracker.ransacThreshold = parseDouble(nextValue(argc, argv, i, arg), arg);
        } else if (arg == "--min-matches") {
            options.tracker.minMatches = parseInt(nextValue(argc, argv, i, arg), arg);
        } else if (arg == "--min-inliers") {
            options.tracker.minInliers = parseInt(nextValue(argc, argv, i, arg), arg);
        } else if (arg == "--display") {
            options.display = true;
        } else if (arg == "--no-flow") {
            options.tracker.enableOpticalFlow = false;
        } else {
            throw std::invalid_argument("Unknown argument: " + arg + ". Use --help.");
        }
    }
    if (options.input.empty()) {
        throw std::invalid_argument("--input VIDEO is required. Use --help.");
    }
    if (!fs::is_regular_file(options.input)) {
        throw std::invalid_argument("Input video does not exist or is not a regular file: " + options.input.string());
    }
    if (!(options.tracker.ratioThreshold > 0.0 && options.tracker.ratioThreshold < 1.0)) {
        throw std::invalid_argument("--ratio must be between 0 and 1 (exclusive)");
    }
    if (!(options.tracker.ransacThreshold > 0.0)) {
        throw std::invalid_argument("--ransac must be positive");
    }
    if (options.tracker.minMatches < 4 || options.tracker.minInliers < 4) {
        throw std::invalid_argument("--min-matches and --min-inliers must be at least 4");
    }
    if (options.label.empty()) {
        throw std::invalid_argument("--label must not be empty");
    }
    if (options.outputDir.empty()) {
        options.outputDir = fs::path("results") / options.input.stem();
    }
    const fs::path canonicalInput = fs::weakly_canonical(options.input);
    for (const char* filename : {"annotated.avi", "tracking.csv", "summary.json", "contact_sheet.jpg"}) {
        const fs::path outputPath = options.outputDir / filename;
        if (canonicalInput == fs::weakly_canonical(outputPath) ||
            (fs::exists(outputPath) && fs::equivalent(options.input, outputPath))) {
            throw std::invalid_argument("Output path would overwrite the input video; choose a different --output-dir");
        }
    }
    return options;
}

void drawResult(cv::Mat& frame, const lab2::TrackingResult& result, const std::string& label,
                int frameIndex) {
    const bool found = result.corners.size() == 4;
    if (found) {
        std::vector<cv::Point> polygon;
        for (const auto& point : result.corners) {
            polygon.emplace_back(cvRound(point.x), cvRound(point.y));
        }
        cv::polylines(frame, polygon, true, cv::Scalar(30, 235, 30), 2, cv::LINE_AA);
        const auto top = *std::min_element(polygon.begin(), polygon.end(),
                                          [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; });
        const int x = std::clamp(top.x, 4, std::max(4, frame.cols - 120));
        const int y = std::clamp(top.y - 7, 22, std::max(22, frame.rows - 7));
        cv::putText(frame, label, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
        cv::putText(frame, label, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(30, 235, 30), 1, cv::LINE_AA);
    }
    std::ostringstream text;
    text << "frame " << frameIndex << " | " << (found ? result.method : "LOST")
         << " | inliers " << result.inliers << "/" << result.matches;
    const int baselineY = std::max(12, frame.rows - 10);
    cv::putText(frame, text.str(), cv::Point(8, baselineY), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(frame, text.str(), cv::Point(8, baselineY), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                found ? cv::Scalar(60, 255, 60) : cv::Scalar(80, 80, 255), 1, cv::LINE_AA);
}

void saveContactSheet(const fs::path& annotatedPath, const fs::path& outputPath,
                      int frameCount, double fps, const std::string& title) {
    cv::VideoCapture video(annotatedPath.string());
    if (!video.isOpened()) {
        throw std::runtime_error("Cannot reopen the annotated video for the contact sheet");
    }
    const int count = std::min(8, frameCount);
    const int thumbWidth = 320;
    const int thumbHeight = 180;
    const int cellHeight = 210;
    const int headerHeight = 35;
    const int rows = (count + 3) / 4;
    cv::Mat sheet(headerHeight + rows * cellHeight, 4 * thumbWidth, CV_8UC3, cv::Scalar(24, 24, 24));
    cv::putText(sheet, title, cv::Point(10, 24), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
    for (int i = 0; i < count; ++i) {
        const int index = count == 1 ? 0 : cvRound(static_cast<double>(i) * (frameCount - 1) / (count - 1));
        video.set(cv::CAP_PROP_POS_FRAMES, index);
        cv::Mat frame;
        if (!video.read(frame) || frame.empty()) {
            throw std::runtime_error("Cannot read output frame " + std::to_string(index) + " for the contact sheet");
        }
        const int x = (i % 4) * thumbWidth;
        const int y = headerHeight + (i / 4) * cellHeight;
        // Keep the original aspect ratio even for a non-16:9 user video.
        const double scale = std::min(static_cast<double>(thumbWidth) / frame.cols,
                                      static_cast<double>(thumbHeight) / frame.rows);
        cv::Mat thumbnail;
        cv::resize(frame, thumbnail, cv::Size(), scale, scale, cv::INTER_AREA);
        thumbnail.copyTo(sheet(cv::Rect(x + (thumbWidth - thumbnail.cols) / 2,
                                       y + (thumbHeight - thumbnail.rows) / 2,
                                       thumbnail.cols, thumbnail.rows)));
        std::ostringstream caption;
        caption << "frame " << index << " | " << std::fixed << std::setprecision(2) << index / fps << " s";
        cv::putText(sheet, caption.str(), cv::Point(x + 8, y + thumbHeight + 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
    }
    if (!cv::imwrite(outputPath.string(), sheet)) {
        throw std::runtime_error("Cannot write contact sheet: " + outputPath.string());
    }
}

int process(const Options& options) {
    cv::setNumThreads(1);
    cv::setRNGSeed(42);
    cv::VideoCapture input(options.input.string());
    if (!input.isOpened()) {
        throw std::runtime_error("Cannot open video: " + options.input.string());
    }
    const double reportedFps = input.get(cv::CAP_PROP_FPS);
    const double fps = std::isfinite(reportedFps) && reportedFps > 0.0 ? reportedFps : 30.0;
    const double reportedFrames = input.get(cv::CAP_PROP_FRAME_COUNT);
    if (fps != reportedFps) {
        std::cerr << "Warning: invalid source FPS; using 30 FPS for output timestamps.\n";
    }
    cv::Mat frame;
    if (!input.read(frame) || frame.empty()) {
        throw std::runtime_error("The input video has no readable first frame");
    }
    const cv::Size size = frame.size();
    const cv::Rect roi = options.hasRoi ? options.roi : cv::Rect(0, 0, size.width, size.height);
    if (roi.x < 0 || roi.y < 0 || roi.width <= 0 || roi.height <= 0 ||
        static_cast<long long>(roi.x) + roi.width > size.width ||
        static_cast<long long>(roi.y) + roi.height > size.height) {
        throw std::invalid_argument("--roi must have positive size and lie inside the first frame");
    }
    lab2::PlanarTracker tracker(frame(roi).clone(), options.tracker);
    fs::create_directories(options.outputDir);
    const fs::path annotatedPath = options.outputDir / "annotated.avi";
    cv::VideoWriter writer(annotatedPath.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, size);
    if (!writer.isOpened()) {
        throw std::runtime_error("Cannot create MJPG output video: " + annotatedPath.string());
    }
    std::ofstream csv(options.outputDir / "tracking.csv");
    if (!csv) {
        throw std::runtime_error("Cannot create tracking.csv in " + options.outputDir.string());
    }
    csv << "frame,time_s,status,method,matches,inliers,inlier_ratio,median_error_px,processing_ms,"
           "x0,y0,x1,y1,x2,y2,x3,y3\n";
    csv << std::fixed << std::setprecision(6);
    int frameIndex = 0;
    int foundFrames = 0;
    int siftFrames = 0, flowFrames = 0;
    double totalProcessingMs = 0.0;
    double totalInliers = 0.0;
    double totalError = 0.0;
    bool stoppedEarly = false;
    const auto wallStart = std::chrono::steady_clock::now();
    do {
        if (frame.size() != size) {
            throw std::runtime_error("Frame size changed at frame " + std::to_string(frameIndex));
        }
        const lab2::TrackingResult result = tracker.locate(frame);
        const bool found = result.corners.size() == 4;
        totalProcessingMs += result.elapsedMs;
        if (found) {
            ++foundFrames;
            if (result.method == "sift") ++siftFrames;
            if (result.method == "lk") ++flowFrames;
            totalInliers += result.inliers;
            totalError += result.medianError;
        }
        csv << frameIndex << ',' << frameIndex / fps << ',' << result.status << ',' << result.method << ',' << result.matches << ','
            << result.inliers << ',' << result.inlierRatio << ',';
        if (result.medianError >= 0.0) {
            csv << result.medianError;
        }
        csv << ',' << result.elapsedMs;
        for (int corner = 0; corner < 4; ++corner) {
            csv << ',';
            if (found) {
                csv << result.corners[corner].x;
            }
            csv << ',';
            if (found) {
                csv << result.corners[corner].y;
            }
        }
        csv << '\n';
        drawResult(frame, result, options.label, frameIndex);
        writer.write(frame);
        ++frameIndex;
        if (frameIndex % 100 == 0) {
            std::cout << "Processed " << frameIndex << " frames; tracked " << foundFrames << '\n';
        }
        if (options.display) {
            cv::imshow("Lab 2 - planar object tracking", frame);
            const int key = cv::waitKey(1) & 0xff;
            if (key == 'q' || key == 'Q' || key == 27) {
                stoppedEarly = true;
                break;
            }
        }
    } while (input.read(frame) && !frame.empty());
    const double wallSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - wallStart).count();
    writer.release();
    input.release();
    csv.flush();
    if (!csv) {
        throw std::runtime_error("Writing tracking.csv failed");
    }
    csv.close();
    if (options.display) {
        cv::destroyAllWindows();
    }
    // Some containers expose an estimated frame count. Report any mismatch
    // explicitly rather than treating this metadata as universally exact.
    const bool hasReportedFrameCount = std::isfinite(reportedFrames) && reportedFrames > 0.0 &&
        std::abs(reportedFrames - std::round(reportedFrames)) < 0.01;
    const bool completeAgainstReportedCount = !hasReportedFrameCount ||
        std::abs(reportedFrames - frameIndex) < 0.01;
    if (!stoppedEarly && !completeAgainstReportedCount) {
        std::cerr << "Warning: decoded " << frameIndex << " frames, but the container reports "
                  << reportedFrames << ". The input may be truncated or the frame count may be inaccurate.\n";
    }
    saveContactSheet(annotatedPath, options.outputDir / "contact_sheet.jpg", frameIndex, fps,
                     options.input.filename().string());
    cv::FileStorage summary((options.outputDir / "summary.json").string(),
                            cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
    if (!summary.isOpened()) {
        throw std::runtime_error("Cannot create summary.json");
    }
    const double coverage = static_cast<double>(foundFrames) / frameIndex;
    const double processingFps = totalProcessingMs > 0.0 ? frameIndex * 1000.0 / totalProcessingMs : 0.0;
    summary << "input" << options.input.lexically_normal().generic_string()
            << "output_video" << annotatedPath.lexically_normal().generic_string()
            << "opencv_version" << CV_VERSION
            << "source_fps" << fps
            << "source_reported_frame_count" << (std::isfinite(reportedFrames) ? reportedFrames : 0.0)
            << "reported_frame_count_available" << static_cast<int>(hasReportedFrameCount)
            << "complete_against_reported_count" << static_cast<int>(completeAgainstReportedCount)
            << "width" << size.width << "height" << size.height
            << "decoded_frames" << frameIndex << "found_frames" << foundFrames
            << "sift_frames" << siftFrames << "flow_frames" << flowFrames
            << "lost_frames" << frameIndex - foundFrames
            << "coverage" << coverage
            << "processing_ms_total" << totalProcessingMs
            << "processing_ms_mean" << totalProcessingMs / frameIndex
            << "processing_fps" << processingFps
            << "wall_seconds" << wallSeconds
            << "wall_fps" << (wallSeconds > 0.0 ? frameIndex / wallSeconds : 0.0)
            << "mean_inliers_when_tracked" << (foundFrames ? totalInliers / foundFrames : 0.0)
            << "mean_median_error_px_when_tracked" << (foundFrames ? totalError / foundFrames : -1.0)
            << "reference_keypoints" << tracker.referenceKeypoints()
            << "stopped_early" << static_cast<int>(stoppedEarly)
            << "reference_roi" << "{" << "x" << roi.x << "y" << roi.y << "width" << roi.width << "height" << roi.height << "}"
            << "parameters" << "{"
            << "max_features" << options.tracker.maxFeatures
            << "reference_pyramid_scales" << "[" << 1.0 << 0.5 << 0.25 << 0.125 << "]"
            << "sift_contrast_threshold" << 0.02
            << "ratio_threshold" << options.tracker.ratioThreshold
            << "ransac_threshold_px" << options.tracker.ransacThreshold
            << "min_matches" << options.tracker.minMatches
            << "min_inliers" << options.tracker.minInliers
            << "min_inlier_ratio" << options.tracker.minInlierRatio
            << "optical_flow_enabled" << static_cast<int>(options.tracker.enableOpticalFlow)
            << "}";
    summary.release();
    std::cout << "Finished: " << foundFrames << '/' << frameIndex << " frames tracked ("
              << std::fixed << std::setprecision(2) << coverage * 100.0 << "%), " << processingFps
              << " processing FPS. Results: " << options.outputDir.string() << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
                printHelp();
                return 0;
            }
        }
        return process(parseOptions(argc, argv));
    } catch (const cv::Exception& error) {
        std::cerr << "OpenCV error: " << error.what() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
    }
    return 1;
}

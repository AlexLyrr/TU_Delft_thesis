/*
	Author: Alexios Lyrakis
 */

#include <cmath>
#include <fstream>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

using namespace cv;

#define SAFE_DISP_LIMIT 5
#define UNCERTAINTY_THRESHOLD 300000
#define PLAN_WITH_UNCERTAINTY
// #define UNCERTAINTY

const int height_tmp = 376, width_tmp = 1241, block_size_tmp = 9, numberOfDisparities = 80;
Point drone_pixel = Point(width_tmp / 2, height_tmp / 2);

Mat cost(height_tmp, width_tmp, CV_8UC3, Scalar(0));
Mat uncertainty_map(height_tmp, width_tmp, CV_8UC1, Scalar(0));

namespace {
    struct ImageParams {
        int width;
        int height;
    };

    class MooreBoundaryTracing {
        public:
        enum MooreNeighborhood {
            P1 = 1,
            P2 = 2,
            P3 = 3,
            P4 = 4,
            P5 = 5,
            P6 = 6,
            P7 = 7,
            P8 = 8,
        };

        MooreBoundaryTracing(ImageParams ip, int safe_disp_limit) : ip(ip), safe_disp_limit(safe_disp_limit), v_boundary_pixels() {}

        // Search for a start point. If no start point is found, we consider the point (0,height/2) as start.
        Point FindStartPoint(Mat &img) {
            // std::cout << "Entered start point search" << std::endl;
            Point start_point;
            bool found = false;
            start_point.y = drone_pixel.y; // We will search only the first row for a start point.
            for (int c = drone_pixel.x; c >= 0; --c) {
                if (img.at<uchar>(drone_pixel.y, c) <= safe_disp_limit) {
                    start_point.x = c + 1;
                    break;
                }
            }
            std::cout << "Start pixels is: " << start_point << std::endl;
            return start_point;
        }

        Point FindNextBoundaryPoint(Mat &img, Point p, MooreNeighborhood &moore_entering_pixel, Point &escape_p) {
            Point next_boundary_point;
            MooreNeighborhood moore_neigh_pixel = moore_entering_pixel;
            bool found = false;
            while (!found) {
                switch (moore_neigh_pixel) {
                case P1: {
                    if (p.x != 0 && p.y != 0) {
                        if (img.at<uchar>(p.y - 1, p.x - 1) > safe_disp_limit) {
                            next_boundary_point.y = p.y - 1;
                            next_boundary_point.x = p.x - 1;
                            found = true;
                            moore_entering_pixel = P6; // Entered from down
                        } else {
                            escape_p = Point(p.x - 1, p.y - 1);
                        }
                    }
                    moore_neigh_pixel = P2;
                    break;
                }
                case P2: {
                    if (p.y != 0) {
                        if (img.at<uchar>(p.y - 1, p.x) > safe_disp_limit) {
                            next_boundary_point.y = p.y - 1;
                            next_boundary_point.x = p.x;
                            found = true;
                            moore_entering_pixel = P8; // Entered from left
                        } else {
                            escape_p = Point(p.x, p.y - 1);
                        }
                    }
                    moore_neigh_pixel = P3;
                    break;
                }
                case P3: {
                    if (p.y != 0 && p.x != img.cols - 1) {
                        if (img.at<uchar>(p.y - 1, p.x + 1) > safe_disp_limit) {
                            next_boundary_point.y = p.y - 1;
                            next_boundary_point.x = p.x + 1;
                            found = true;
                            moore_entering_pixel = P8; // Entered from left
                        } else {
                            escape_p = Point(p.x + 1, p.y - 1);
                            ;
                        }
                    }
                    moore_neigh_pixel = P4;
                    break;
                }
                case P4: {
                    if (p.x != img.cols - 1) {
                        if (img.at<uchar>(p.y, p.x + 1) > safe_disp_limit) {
                            next_boundary_point.y = p.y;
                            next_boundary_point.x = p.x + 1;
                            found = true;
                            moore_entering_pixel = P2; // Entered from up
                        } else {
                            escape_p = Point(p.x + 1, p.y);
                            ;
                        }
                    }
                    moore_neigh_pixel = P5;
                    break;
                }
                case P5: {
                    if (p.x != img.cols - 1 && p.y != img.rows - 1) {
                        if (img.at<uchar>(p.y + 1, p.x + 1) > safe_disp_limit) {
                            next_boundary_point.y = p.y + 1;
                            next_boundary_point.x = p.x + 1;
                            found = true;
                            moore_entering_pixel = P2; // Entered from up
                        } else {
                            escape_p = Point(p.x + 1, p.y + 1);
                            ;
                        }
                    }
                    moore_neigh_pixel = P6;
                    break;
                }
                case P6: {
                    if (p.y != img.rows - 1) {
                        if (img.at<uchar>(p.y + 1, p.x) > safe_disp_limit) {
                            next_boundary_point.y = p.y + 1;
                            next_boundary_point.x = p.x;
                            found = true;
                            moore_entering_pixel = P4; // Entered from right
                        } else {
                            escape_p = Point(p.x, p.y + 1);
                            ;
                        }
                    }
                    moore_neigh_pixel = P7;
                    break;
                }
                case P7: {
                    if (p.x != 0 && p.y != img.rows - 1) {
                        if (img.at<uchar>(p.y + 1, p.x - 1) > safe_disp_limit) {
                            next_boundary_point.y = p.y + 1;
                            next_boundary_point.x = p.x - 1;
                            found = true;
                            moore_entering_pixel = P4; // Entered from right
                        } else {
                            escape_p = Point(p.x - 1, p.y + 1);
                            ;
                        }
                    }
                    moore_neigh_pixel = P8;
                    break;
                }
                case P8: {
                    if (p.x != 0) {
                        if (img.at<uchar>(p.y, p.x - 1) > safe_disp_limit) {
                            next_boundary_point.y = p.y;
                            next_boundary_point.x = p.x - 1;
                            found = true;
                            moore_entering_pixel = P6; // Entered from down
                        } else {
                            escape_p = Point(p.x - 1, p.y);
                            ;
                        }
                    }
                    moore_neigh_pixel = P1;
                    break;
                }
                }
            }
            return next_boundary_point;
        }

        bool BelongToImageBoundary(Mat &img, Point p) {
            if (p.x != 0 && p.x != img.cols - 1 && p.y != 0 && p.y != img.rows - 1)
                return false;
            return true;
        }

        void BoundaryTracing(Mat &img) {
            Point start_point = FindStartPoint(img);
            v_boundary_pixels.push_back(start_point);
            MooreNeighborhood moore_entering_pixel = P8;
            Point current_point = start_point, next_boundary_point(img.cols - start_point.x, img.rows - start_point.y);
            Point escape_p;
            while (start_point != next_boundary_point) {
                next_boundary_point = FindNextBoundaryPoint(img, current_point, moore_entering_pixel, escape_p);
                if (!BelongToImageBoundary(img, next_boundary_point)) {
                    v_boundary_pixels.push_back(next_boundary_point);
                    v_escape_pixels.push_back(escape_p);
                }
                current_point = next_boundary_point;
            }
        }

        std::vector<Point> GetBoundaryPixels() const { return v_boundary_pixels; }
        std::vector<Point> GetEscapePixels() const { return v_escape_pixels; }

        private:
        ImageParams ip;
        int safe_disp_limit; // This may depend on the speed
        std::vector<Point> v_boundary_pixels;
        std::vector<Point> v_escape_pixels;
    };

    bool DetectCollision(Mat &img, int safe_disp_limit) {
        int width = img.cols, height = img.rows;
        if (img.at<uchar>(drone_pixel) > safe_disp_limit) {
            std::cout << "Collision Detected and drone pixel is: " << drone_pixel << std::endl;
            std::cout << "The value of the drone pixel is: " << (int)img.at<uchar>(drone_pixel) << std::endl;
            return true;
        }

        std::cout << "No collision, the algorithm will now finish without finding object boundary" << std::endl;
        std::cout << "The value of the drone pixel is: " << (int)img.at<uchar>(drone_pixel) << std::endl;

        return false;
    }
} // namespace

std::ofstream collision_csv, avoidance_point_csv;
void openFiles() {
    collision_csv.open("csv/collision_detection.csv", std::ofstream::out | std::ofstream::app);
    if (!collision_csv.is_open()) {
        std::cout << "problem with file";
    }
    avoidance_point_csv.open("csv/avoidance_point.csv", std::ofstream::out | std::ofstream::app);
    if (!collision_csv.is_open()) {
        std::cout << "problem with file";
    }
}

int main(int argc, char **argv) {
    Mat disparity_img = imread(argv[1], cv::IMREAD_GRAYSCALE);
    uncertainty_map = imread(argv[2], cv::IMREAD_GRAYSCALE);
    int kernel_size = 100;
    Mat kernel = Mat::ones(kernel_size, kernel_size, CV_32S);
    // std::cout << kernel << std::endl;
    filter2D(uncertainty_map, uncertainty_map, CV_32S, kernel);

    if (disparity_img.empty() || uncertainty_map.empty()) {
        printf("Read input image failed\n");
        return -1;
    }
    std::cout << "Image size: " << disparity_img.size << std::endl;
    ImageParams ip = {
        .width = disparity_img.cols,
        .height = disparity_img.rows,
    };

    openFiles();
    /*
  	CHECK FOR CORRECT COLLISION DETECTION
  	true positives - 0
  	true negatives - 1
  	false positives - 2
  	false negatives - 3
   */
    bool disp_collision = DetectCollision(disparity_img, SAFE_DISP_LIMIT);
#ifdef UNCERTAINTY
    if (uncertainty_map.at<int>(drone_pixel) > UNCERTAINTY_THRESHOLD) {
        disp_collision = true;
    }
#endif
    if (DetectCollision(disparity_img, SAFE_DISP_LIMIT) && (disp_collision == truth_collision)) {
        MooreBoundaryTracing MBT(ip, SAFE_DISP_LIMIT);
        MBT.BoundaryTracing(disparity_img);
        std::vector<Point> v_boundary_pixels = MBT.GetBoundaryPixels();
        std::vector<Point> v_escape_pixels = MBT.GetEscapePixels();

        Point escape_p = v_escape_pixels[1];
        int min_uncertainty = INT_MAX;
#ifdef UNCERTAINTY
        for (size_t i = 0; i < v_escape_pixels.size(); ++i) {
            if (uncertainty_map.at<int>(v_escape_pixels[i]) < min_uncertainty && v_escape_pixels[i].x > kernel_size / 2 && v_escape_pixels[i].x < uncertainty_map.cols - kernel_size / 2 && v_escape_pixels[i].y > kernel_size / 2 && v_escape_pixels[i].y < uncertainty_map.rows - kernel_size / 2) {
                min_uncertainty = uncertainty_map.at<int>(v_escape_pixels[i]);
                escape_p = v_escape_pixels[i];
            }
        }
#endif
    }

    imwrite("image_test.png", disparity_img);

    return 0;
}

#include <iostream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

vector<vector<Point>> getContours(Mat img) {
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    vector<vector<Point>> validContours;
    vector<Point> hull;
    for(int i=0; i<contours.size(); i++) {
        if(contourArea(contours[i]) > 30000)
            validContours.push_back(contours[i]);
    }
    return validContours;
}

void mask_morph(Mat &mask) {
    Mat erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));
    erode(mask, mask, erodeElement);
    erode(mask, mask, erodeElement);
    dilate(mask, mask, dilateElement);
    dilate(mask, mask, dilateElement);
}

int main() {

    VideoCapture cap;
    if(!cap.open(0)) {
        cout << "Cannot open camera." << endl;
        return -1;
    }
    Mat origin;
    Mat output;
    Mat gray_image, yCbCr_image, HSV_image;
    Mat mask;

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    vector<vector<Point>> validContours;
    vector<Point> hull;

    Scalar lower = Scalar(100, 50, 0);
    Scalar upper = Scalar(125, 255, 255);
    while (true)
    {
        cap >> origin;
        medianBlur(origin, origin, 5);
        output = origin;
        cvtColor(origin, gray_image, COLOR_BGR2GRAY);
        cvtColor(origin, yCbCr_image, COLOR_RGB2YCrCb);
        cvtColor(origin, HSV_image, COLOR_RGB2HSV);
        inRange(HSV_image, lower, upper, mask);

        Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
        mask_morph(mask);
        morphologyEx(mask, mask, MORPH_OPEN, element);
        morphologyEx(mask, mask, MORPH_CLOSE, element);

        imshow("mask", mask);
        contours.clear();
        hierarchy.clear();
        validContours.clear();
        hull.clear();
        findContours(mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

        for(int i=0; i<contours.size(); i++) {
            if(fabs(contourArea(Mat(contours[i]))) > 9000)
                validContours.push_back(contours[i]);
        }

        drawContours(output, validContours, -1, Scalar(0, 0, 255));
        for(int i=0; i<validContours.size(); i++) {
            convexHull(Mat(validContours[i]), hull, true);
            int hullcount = (int)hull.size();
            for(int j=0; j<hullcount; j++) {
                line(output, hull[j+1], hull[j], Scalar(255, 0, 0), 2);
            }
            line(output, hull[hullcount-1], hull[0], Scalar(255, 0, 0), 2);
        }
        imshow("Output", output);
        output.release();
        char key = waitKey(1);
        if(key == 'q') {
            cout << "exit" << endl;
            return 0;
        }
    }

    return 0;
}

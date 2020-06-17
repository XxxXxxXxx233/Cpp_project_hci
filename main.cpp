#include <iostream>
#include <opencv2/opencv.hpp>
#include <windows.h>

#define MAX_INT 2147483647
#define mouseLockFrame 50
#define actionCheckFrame 25

using namespace std;
using namespace cv;

void printInfo();

int getDistance(Point p1, Point p2);
int getDistance(POINT p1, POINT p2);
string intToString(int num);

bool checkActionPoint(vector<POINT> mousePosition);
POINT getActionPoint(vector<POINT> mousePosition);
bool doCurrentAction(vector<int> fingerTipsSize);
void morph(Mat &img);

int main() {

    printInfo();

    VideoCapture cap;
    if(!cap.open(0)) {
        cout << "Cannot open camera." << endl;
        return -1;
    }
    Mat origin, allBlack, output, yCrCb_image, mask;

    //Scalar to find skin
    Scalar lower = Scalar(0, 135, 90);
    Scalar upper = Scalar(255, 230, 150);

    //To get the contour of hand
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    vector<vector<Point>> validContours;
    vector<Point> hull;
    vector<int> intHull;
    vector<Vec4i> defectHull;
    vector<Vec4i>::iterator defectHullIter;
    vector<Point> fingerTips;

    //To get the center of hand and track it
    Moments moment;
    bool trackCenter = false;
    vector<Point> centerTrajectory;

    //To analyse the trajectory
    vector<vector<Point>> polyContours;
    vector<Vec4i> polyHierarchy;
    vector<Point> approxPoly;
    double epsilon;
    int corners;
    vector<string> shape;
    int shapeMatched = -1;

    //To control the mouse
    bool controlMouseByHand = false;
    POINT curMouse;
    GetCursorPos(&curMouse);
    int preX = 0;
    int preY = 0;
    int curX = curMouse.x;
    int curY = curMouse.y;
    bool mouseLocked = false;
    vector<POINT> mousePosition;
    int curFinger;
    vector<int> fingerTipsSize;
    POINT actionPoint;
    int curAction;

    //Different colors to draw the picture
    Scalar blackScalar = Scalar(0, 0, 0);
    Scalar whiteScalar = Scalar(255, 255, 255);
    Scalar blueScalar = Scalar(255, 0, 0);
    Scalar redScalar = Scalar(0, 0, 255);
    Scalar greenScalar = Scalar(0, 255, 0);
    Scalar purpleScalar = Scalar(255, 0, 255);
    Scalar cyanScalar = Scalar(255, 255, 0);
    Scalar yellowScalar = Scalar(0, 255, 255);

    int hullCount, index, count, distance, curStart, curEnd, curFar, curDepth;

    //To get the rectangle of hand
    Point leftTop, rightDown;
    int minx, maxx, miny, maxy;

    while (true)
    {
        cap >> origin;

        allBlack.create(Size(origin.cols, origin.rows), CV_8UC3);
        allBlack.setTo(0);

        flip(origin, origin, 1);
        medianBlur(origin, origin, 5);
        output = origin;
        cvtColor(origin, yCrCb_image, COLOR_BGR2YCrCb);
        inRange(yCrCb_image, lower, upper, mask);

        threshold(mask, mask, 20, 255, THRESH_BINARY);
        morph(mask);
        blur(mask, mask, Size(10, 10));
        threshold(mask, mask, 20, 255, THRESH_BINARY);

        contours.clear();
        hierarchy.clear();
        validContours.clear();
        hull.clear();
        defectHull.clear();
        polyContours.clear();
        polyHierarchy.clear();
        approxPoly.clear();
        shape.clear();
        fingerTips.clear();
        minx = miny = MAX_INT;
        maxx = maxy = -1;

        preX = curX;
        preY = curY;
        moment = moments(mask, true);
        Point center(moment.m10/moment.m00, moment.m01/moment.m00);
        curX = center.x;
        curY = center.y;

        if(trackCenter) {
            centerTrajectory.push_back(center);
        }
        for(int i=1; i<centerTrajectory.size(); i++) {
            line(allBlack, centerTrajectory[i-1], centerTrajectory[i], whiteScalar, 2);
            line(output, centerTrajectory[i-1], centerTrajectory[i], greenScalar, 2);
        }
        cvtColor(allBlack, allBlack, COLOR_BGR2GRAY);
        threshold(allBlack, allBlack, 0, 255, THRESH_BINARY);
        findContours(allBlack, polyContours, polyHierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        if(polyContours.size() > 0 && !trackCenter) {
            for(int i=0; i<polyContours.size(); i++) {
                epsilon = 0.02 * arcLength(polyContours[i], true);
                approxPolyDP(polyContours[i], approxPoly, epsilon, true);
                corners = approxPoly.size();
                if(corners == 3) {
                    shape.push_back("Triangle");
                } else if (corners == 4) {
                    shape.push_back("Rectangle");
                } else if (corners > 4 && corners < 10) {
                    shape.push_back("Polygon");
                } else if (corners >= 10) {
                    shape.push_back("Circle");
                }
            }
            shapeMatched = -1;
            for (int i=0; i<shape.size(); i++) {
                if(shape[i] == "Triangle") {
                    ShellExecute(NULL, "open", "explorer.exe", "https://www.sustech.edu.cn/", NULL, SW_SHOW);
                    shapeMatched = 1;
                    break;
                } else if(shape[i] == "Rectangle") {
                    ShellExecute(NULL, "open", "explorer.exe", "https://www.bilibili.com/", NULL, SW_SHOW);
                    shapeMatched = 2;
                    break;
                } else if(shape[i] == "Circle") {
                    ShellExecute(NULL, "open", "explorer.exe", "http://www.baidu.com/", NULL, SW_SHOW);
                    shapeMatched = 3;
                    break;
                }
            }
            if(shapeMatched == 1) {
                cout << "Triangle" << endl;
            } else if (shapeMatched == 2) {
                cout << "Rectangle" << endl;
            } else if (shapeMatched == 3) {
                cout << "Circle" << endl;
            } else {
                cout << "Polygon" << endl;
            }
            centerTrajectory.clear();
            cout << "Clean trajectory" << endl;
            cout << endl;
        }

        circle(output, center, 5, blueScalar, -1);
        circle(allBlack, center, 5, whiteScalar, -1);

        findContours(mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        for(auto & contour : contours) {
            if(fabs(contourArea(Mat(contour))) > 25000)
                validContours.push_back(contour);
        }

        drawContours(output, validContours, -1, redScalar, 3, 8);
        for(auto & validContour : validContours) {
            convexHull(Mat(validContour), hull, true, true);
            hullCount = hull.size();
            for(int j=1; j<hullCount; j++) {
                line(output, hull[j+1], hull[j], blueScalar, 2);
                if((center.y >= hull[j].y && center.y >= hull[index].y) || (abs(center.y - hull[j].y) < 100 && center.x < hull[j].x)) {
                    distance = getDistance(center, hull[j]);
                    if(distance > 40000 && abs(hull[j-1].x - hull[j].x) >= 25) {
                            count = j;
                            index = j;
                            while(abs(hull[count].x - hull[count+1].x) < 25) {
                                count++;
                            }
                            index += (count - index) / 2;
                            fingerTips.push_back(hull[index]);
//                            line(output, center, hull[index], purpleScalar, 3);
                            circle(output, hull[index], 10, greenScalar, -1);
                    }
                }
                if(hull[j].x < minx) {
                    minx = hull[j].x;
                }
                if(hull[j].x > maxx) {
                    maxx = hull[j].x;
                }
                if(hull[j].y < miny) {
                    miny = hull[j].y;
                }
                if(hull[j].y > maxy) {
                    maxy = hull[j].y;
                }
            }
            line(output, hull[hullCount-1], hull[0], blueScalar, 2);
            leftTop.x = minx;
            leftTop.y = maxy;
            rightDown.x = maxx;
            rightDown.y = miny;
            rectangle(output, leftTop, rightDown, cyanScalar, 2);

            convexHull(Mat(validContour), intHull, true, false);
            convexityDefects(Mat(validContour), intHull, defectHull);
            defectHullIter = defectHull.begin();
            while(defectHullIter != defectHull.end()) {
                Vec4i &cur = (*defectHullIter);
                curStart = cur[0];
                Point startPoint(validContour[curStart]);
                curEnd = cur[1];
                Point endPoint(validContour[curEnd]);
                curFar = cur[2];
                Point farPoint(validContour[curFar]);
                curDepth = cur[3] / 256;
                if(curDepth > 30) {
                    line(output, startPoint, farPoint, yellowScalar, 2);
                    line(output, endPoint, farPoint, yellowScalar, 2);
                    circle(output, startPoint, 7, yellowScalar, 2);
                    circle(output, endPoint, 7, yellowScalar, 2);
                    circle(output, farPoint, 7, yellowScalar , 2);
                }
                defectHullIter++;
            }
        }

        curFinger = fingerTips.size();
        fingerTipsSize.push_back(curFinger);

        if(!trackCenter && controlMouseByHand) {
            if(curFinger == 0 && !mouseLocked) {
                mouse_event(MOUSEEVENTF_MOVE, 5 * (curX - preX), 5 * (curY - preY), 0, 0);
                GetCursorPos(&curMouse);
                mousePosition.push_back(curMouse);
            }
            if(mousePosition.size() > mouseLockFrame && !mouseLocked) {
                if(checkActionPoint(mousePosition)) {
                    actionPoint = getActionPoint(mousePosition);
                    mouseLocked = true;
                    mouse_event(MOUSEEVENTF_ABSOLUTE, actionPoint.x, actionPoint.y, 0, 0);
                    cout << "Mouse Locked!" << endl;
                }
            }
            if(fingerTipsSize.size() > actionCheckFrame && mouseLocked) {
                curAction = fingerTipsSize[fingerTipsSize.size()-1];
                if(doCurrentAction(fingerTipsSize)) {
                    switch (curAction) {
                        case 5:
                            mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
                            mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
                            cout << "Left Click" << endl;
                            break;
                        case 4:
                            mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
                            cout << "Left Pressed" << endl;
                            break;
                        case 3:
                            mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
                            cout << "Left Released" << endl;
                            break;
                        case 2:
                            mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
                            mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
                            mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
                            mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
                            cout << "Left Double Click" << endl;
                            break;
                        case 1:
                            mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0);
                            mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0);
                            cout << "Right Click" << endl;
                            break;
                    }
                    if(curAction != 0) {
                        mousePosition.clear();
                        fingerTipsSize.clear();
                        mouseLocked = false;
                    }
                }
            }
        }

        putText(output, intToString(fingerTips.size()), Point(center.x, center.y + 50), 1, 2, redScalar);

        imshow("AllBlack", allBlack);
        imshow("Output", output);

        origin.release();
        output.release();
        mask.release();
        allBlack.release();

        char key = waitKey(1000/60);
        if(key != -1) {
            switch (key) {
                case 'q':
                    cout << "exit" << endl;
                    return 0;
                case 't':
                    cout << "Start to track center" << endl;
                    trackCenter = true;
                    break;
                case 'y' :
                    cout << "Stop tracking" << endl;
                    trackCenter = false;
                    break;
                case 'u':
                    cout << "Clean trajectory" << endl;
                    centerTrajectory.clear();
                    trackCenter = false;
                    break;
                case 'a':
                    cout << "Start to control mouse by hand" << endl;
                    controlMouseByHand = true;
                    break;
                case 's':
                    cout << "Stop controlling mouse" << endl;
                    controlMouseByHand = false;
                    break;
            }
        }
    }
    return 0;
}

int getDistance(Point p1, Point p2) {
    return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}

int getDistance(POINT p1, POINT p2) {
    return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}

string intToString(int num) {
    stringstream ss;
    ss << num;
    return ss.str();
}

bool checkActionPoint(vector<POINT> mousePosition) {
    int size = mousePosition.size();
    POINT actionPoint = mousePosition[size-1];
    for(int i=size-mouseLockFrame; i<size; i++) {
        if(getDistance(mousePosition[i], actionPoint) > 10000) {
            return false;
        }
    }
    return true;
}

POINT getActionPoint(vector<POINT> mousePosition) {
    int size = mousePosition.size();
    POINT p;
    p.x = 0;
    p.y = 0;
    for(int i=size-mouseLockFrame; i<size; i++) {
        p.x += mousePosition[i].x;
        p.y += mousePosition[i].y;
    }
    p.x /= mouseLockFrame;
    p.y /= mouseLockFrame;
    return p;
}

bool doCurrentAction(vector<int> fingerTipsSize) {
    int size = fingerTipsSize.size();
    int curAction = fingerTipsSize[size-1];
    for(int i=size-actionCheckFrame; i<size; i++) {
        if(fingerTipsSize[i] != curAction) {
            return false;
        }
    }
    return true;
}

void morph(Mat &img) {
    Mat erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));
    erode(img, img, erodeElement);
    erode(img, img, erodeElement);
    dilate(img, img, dilateElement);
    dilate(img, img, dilateElement);
}

void printInfo() {
    cout << "Buttons: \n"
            "\tq -> Exit\n"
            "\tt -> Track the trajectory of center\n"
            "\ty -> Stop tracking\n"
            "\tu -> Clean trajectory\n"
            "\ta -> Control mouse by hand\n"
            "\ts -> Stop controlling\n";
    cout << "Usage of the trajectory: \n"
            "\tTriangle -> Open https://www.sustech.edu.cn/\n"
            "\tRectangle -> Open https://www.bilibili.com/\n"
            "\tCircle -> Open http://www.baidu.com/\n"
            "\tPolygon -> Do nothing\n";
    cout << "Controlled by the number of fingers: \n"
            "\t0 -> Move the mouse\n"
            "\t5 -> Left click\n"
            "\t4 -> Left pressed\n"
            "\t3 -> Left released\n"
            "\t2 -> Left double click\n"
            "\t1 -> Right click\n";
}
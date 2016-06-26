
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>
#include <sstream>
#include <string.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <cmath>
#include <vector>
#include "tinyxml.h"

#include "tracker.h"
#include "particle.h"


using namespace cv;
using namespace std;


Mat src;
const char *windowName = "imageWindow"; 

 string file_region="../region/ram.xml";
Scalar color(0,0,255); // red color
vector<Point2f> Region;

int isregionset=0;

void parseRegionXML(string file_region, vector<Point2f> &region){
    TiXmlDocument doc(file_region.c_str());
    if(doc.LoadFile()) // ok file loaded correctly
    {
        TiXmlElement * point = doc.FirstChildElement("point");
        int x, y;
        while (point)
        {
            point->Attribute("x",&x);
            point->Attribute("y",&y);
            Point2f p(x,y);
	    region.push_back(p);
            point = point->NextSiblingElement("point");
        }
    }
    else
        exit(1);
}

bool isPoint(vector<Point2f>  contour,Point test_pt)
{
	if (pointPolygonTest(Mat(contour), test_pt, true) < 0)return TRUE;
	return FALSE;
}


int main(int argc, const char *argv[])
{
	VideoCapture cam;
	int partn = 100;
	bool videoFile=false;
	bool showParticles=false;
	float sd = SD;
	float scale_sd = SCALE_SD;
//	
	RNG rng(12345);
	int count;
	int objMinWidth=30;
	int objMinHeight=30;
	int objMaxWidth=150;
	int objMaxHeight=150;
	Mat frame, fore,imge, prevImg, temp, gray, object_ROI, img_temp;
        BackgroundSubtractorMOG2 bg(500, 25, false);
	vector<vector<Point> > contours;
	vector<Rect> objects;

	cout<<"\nPress Esc Button for Exit"<<endl;
	//parsing command line
	if(argc > 1) {
		for(int j=1; j < argc; j++) {
			if(strcmp(argv[j],"-v") == 0) {
				cam = *new VideoCapture(argv[j+1]);
				if( !cam.isOpened() ) {
					cout << "Failed to open file " << argv[j+1] << endl;
					return 0;
				}
				videoFile=true;
			} else if(strcmp(argv[j],"-p") == 0) {
				partn = atoi(argv[j+1]);
			} else if(strcmp(argv[j],"-sd") == 0) {
				sd = atof(argv[j+1]);
			} else if(strcmp(argv[j],"-sds") == 0) {
				scale_sd = atof(argv[j+1]);
			} else if(strcmp(argv[j],"-sp") == 0) {
				showParticles = true;
			} 
		}	
	}
	if(!videoFile) {
		cam = *new VideoCapture(0);
		if( !cam.isOpened() ) {
			cout << "Failed to open camera" << endl;
			return 0;
		}

	}
	 parseRegionXML(file_region, Region);	
	
	cam >> img_temp;
	

	Mat img;
	Mat hsvImg;
	MatND hist1;
	char key = 10;
	namedWindow("pftracker",0);
	cvSetMouseCallback("pftracker",mouseEventHandler,&img);
		
	particle* pArr;
	particle best;
	bool pInit=false;
	
	cvtColor(img_temp, gray, CV_BGR2GRAY);
	gray.convertTo(temp, CV_8U);
	bilateralFilter(temp, prevImg, 5, 20, 20);

	
	while(key != 27) {
			
	count=0;
		cam >> img;
		frame=img;
		key = cvWaitKey(33);
		cvtColor(frame, gray, CV_BGR2GRAY);
		gray.convertTo(temp, CV_8U);
		bilateralFilter(temp, imge, 5, 20, 20);
		bg.operator()(imge,fore);

	// laplacian filer
	Laplacian( fore,fore, CV_16S, 3, 1, 0, BORDER_DEFAULT );
	convertScaleAbs( fore,fore);
	//thresholding by OTSU algorithm
	threshold( fore,fore, 100, 255, THRESH_OTSU );
	//
		findContours(fore,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
		vector<vector<Point> > contours_poly(contours.size());
		vector<Rect> boundRect(contours.size());

		//if the user select the target object	
		if(areaDefined) {
			cvtColor(img,hsvImg,CV_BGR2HSV);
			//executed just one time, to initalize
			if(!pInit) {
				hist1 = getHistogramHSV(hsvImg(*sArea));
				pArr = init_particles(sArea,&hist1,1,partn);
				pInit = true;
			}
			//update using gaussian random number generator	
			updateParticles(pArr,partn,hsvImg,hist1,sd,scale_sd);
			//select the best particle
			best = getBest(pArr,partn);
			//make copies of best particles and erase worsts
			pArr = resampleParticles(pArr,partn);			
			rectangle(img,
						Point(best.x-best.width/2*best.scale,best.y-best.height/2*best.scale),
					 	Point(best.x + best.width/2*best.scale,best.y + best.height/2*best.scale),
						Scalar(0,0,255) );

			//show all particles
			for(int p=0; p < partn && showParticles; p++) {
				best = pArr[p];
				rectangle(img,
						Point(best.x-best.width/2*best.scale,best.y-best.height/2*best.scale),
					 	Point(best.x + best.width/2*best.scale,best.y + best.height/2*best.scale),
						Scalar(0,255,0),2,15,0);
			} 	
		} else {
			cvWaitKey(300);
		}

		for(size_t i = 0; i < contours.size(); i++ )
	{
	approxPolyDP( Mat(contours[i]), contours_poly[i], 10, true );
	boundRect[i] = boundingRect( Mat(contours_poly[i]) );
	int x =boundRect[i].x;
	int y =boundRect[i].y;
	int w =boundRect[i].width;
	int h =boundRect[i].height;
	if (w>objMinWidth && h>objMinHeight && w<objMaxWidth && h<objMaxHeight)	{
	
		int midx=x+(w/2);
		int midy=y+(h/2);
		const cv::Point *pts = (const cv::Point*) Mat(Region).data;
	int npts = Mat(Region).rows;
	if(isPoint(Region,Point(midx,midy)))
	 rectangle( img, boundRect[i].tl(), boundRect[i].br(), Scalar(255,0,0), 2, 8, 0 );
	else
	 rectangle( img, boundRect[i].tl(), boundRect[i].br(), Scalar(123,255,0), 2, 8, 0 );	
	count=count+1;}
	}
		// display moving object detection		
		imshow("pftracker",img);
		
	}

	return 0;
}

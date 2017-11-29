#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<iostream>
#include<conio.h> 
#include "Blob.h"

using namespace std;
using namespace cv;

//scalar constants for color
const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs); //
void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex); //add blobs to existing blobs
void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs); //adds new blob
double distanceBetweenPoints(cv::Point point1, cv::Point point2); //finds distance between points
void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName); // draw and show contours 
void drawAndShowContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName); //draw and show blobs
bool checkIfBlobsCrossedTheLine(std::vector<Blob> &blobs, int &intHorizontalLinePosition, int &carCount);//checks if blobs crossed the horizontal line
void drawBlobInfoOnImage(std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy);//draws blob info on image
void drawCarCountOnImage(int &carCount, cv::Mat &imgFrame2Copy); // draw car count on image


int main(int argc, char** argv) {

	int carCount = 0;
	VideoCapture cap;
	vector<Blob> blobs; //vector of blob type objects

	Mat firstFrame;
	Mat secondFrame;

	Point horizontalCrossLine[2]; //contains the start and end point of horizontal line to be placed in frame

	cap.open(argv[1]);
	//read video
	if (!cap.isOpened())
	{
		cout << "error reading video file";
		return 0;
	}

	//check if there are atleast two frames in a video
	if (cap.get(CV_CAP_PROP_FRAME_COUNT) < 2) {
		cout << "error .File must have atleast two frames";
	}

	cap.read(firstFrame); // read first frame	
	cap.read(secondFrame); //read second frame

	//defines the position where we want to draw the horizontal line. We can change the position of line up or down .Multiplying it with 10% will put it near the top of frame 
	int horizontalLinePosition = round(firstFrame.rows*0.40);


	// horizontal cross line is defined with two end points being (0,no.of rows*0.40) and (columns-1, rows*0.40)
	horizontalCrossLine[0].x = 0;
	horizontalCrossLine[0].y = horizontalLinePosition;

	horizontalCrossLine[1].x = firstFrame.cols - 1;
	horizontalCrossLine[1].y = horizontalLinePosition;

	char keyboardKey = 0; // records the keystroke
	bool blnFirstFrame = true; // boolean flag to see if the frame is first frame

	int frameCount = 2; 
	while (cap.isOpened() && keyboardKey != 27) {

		vector<Blob> currentFrameBlobs;  // vector of type blob (we have defined the blob class which is a custom class.
		Mat firstFrameCopy = firstFrame.clone(); //copy of first frame
		Mat secondFrameCopy = secondFrame.clone(); // copy of second frame

		Mat imgDifference; 
		Mat imgThreshold;

		//converts to gray
		cvtColor(firstFrameCopy, firstFrameCopy, CV_BGR2GRAY);
		cvtColor(secondFrameCopy, secondFrameCopy, CV_BGR2GRAY);

		//smoothens the image using 5*5 filter
		GaussianBlur(firstFrameCopy, firstFrameCopy, Size(5, 5), 0);
		GaussianBlur(secondFrameCopy, secondFrameCopy, Size(5, 5), 0);

		//finds the absolute difference between the frames
		absdiff(firstFrameCopy, secondFrameCopy, imgDifference);

		//thresholds the result of difference of image  
		
		threshold(imgDifference, imgThreshold, 30, 255, CV_THRESH_BINARY);
		//imshow("Threshold image", imgThreshold);

		//imshow("firstframe", firstFrame);
		//imshow("secondframe", secondFrame);
		//imshow("diff", imgDifference);

		//defines a 7*7 structuring element to be used for erosion and dilation
		Mat structuringElement7x7 = getStructuringElement(MORPH_RECT, Size(7, 7));


		//dilates the thresholded image(which is a thresholded difference of frames) 8 times and erodes 4 times using above defined 7*7 structuring element
		for (int idx = 0; idx < 4; idx++) {
			dilate(imgThreshold, imgThreshold, structuringElement7x7);
			dilate(imgThreshold, imgThreshold, structuringElement7x7);
			erode(imgThreshold, imgThreshold, structuringElement7x7);

		}

		Mat imgThresholdCopy = imgThreshold.clone(); //clones the thresolded image
		imshow("thresh", imgThresholdCopy);

		vector<vector<Point>> contours;
		
		// find the contours in binary thresholded image
		findContours(imgThresholdCopy, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

		//calls draw and show function on contours define below
		drawAndShowContours(imgThreshold.size(), contours, "imgContours");

		// find the convexHull object of each contour
		vector<vector<Point>> convexHulls(contours.size());

		for (int idx = 0; idx < contours.size(); idx++) {
			convexHull(contours[idx], convexHulls[idx]);
		}

		//draw and show convex hulls
		drawAndShowContours(imgThreshold.size(), contours, "imgConvexHulls");


		// iterate over all the convexHulls and creating a blob object of each convex hull and identifying the contour of car so that we can count
		for (auto &convexHull : convexHulls) {
			Blob possibleBlob(convexHull);
			if (possibleBlob.currentBoundingRect.area() > 400) 
			{
				currentFrameBlobs.push_back(possibleBlob); //if area of contour is >400 which is a vehicle then push that blob onto vector
			}
		}

		//draw and show blobs
		drawAndShowContours(imgThreshold.size(), currentFrameBlobs, "currentFrameBlobs");

		//if the frame is first frame push the blobs onto the blob vector of blobs else match the blobs of current frame to existing blobs of vector
		if (blnFirstFrame == true) {
			for (auto &currentFrameBlob : currentFrameBlobs) {
				blobs.push_back(currentFrameBlob);
			}
		}

		else {
			matchCurrentFrameBlobsToExistingBlobs(blobs, currentFrameBlobs);
		}


		drawAndShowContours(imgThreshold.size(), blobs, "blobs");
		secondFrameCopy = secondFrame.clone();

		//function to draw the blob info on images
		drawBlobInfoOnImage(blobs, secondFrameCopy);

		// check if blobs crossed the line using the check function defined below
		bool blnAtLeastOneBlobCrossedTheLine = checkIfBlobsCrossedTheLine(blobs, horizontalLinePosition, carCount);

		//turns the horizontal line from green to red (bascially creates a new green line) when the car centroid crosses that line and keeps it red other times
		if (blnAtLeastOneBlobCrossedTheLine == true) {
			cv::line(secondFrameCopy, horizontalCrossLine[0], horizontalCrossLine[1], SCALAR_GREEN, 2);
		}
		else {
			cv::line(secondFrameCopy, horizontalCrossLine[0], horizontalCrossLine[1], SCALAR_RED, 2);
		}

		//draw car count on image
		drawCarCountOnImage(carCount, secondFrameCopy);

		cv::imshow("imgFrame2Copy", secondFrameCopy);

		//cv::waitKey(0);                 // uncomment this line to go frame by frame for debugging

		// now we prepare for the next iteration

		currentFrameBlobs.clear();  //clear blobs from current frame

		firstFrame = secondFrame.clone();           // move frame 1 up to where frame 2 is

		//read next frame if it exists
		if ((cap.get(CV_CAP_PROP_POS_FRAMES) + 1) < cap.get(CV_CAP_PROP_FRAME_COUNT)) {
			cap.read(secondFrame);
		}
		else {
			std::cout << "end of video\n";
			break;
		}

		blnFirstFrame = false;
		frameCount++;
		keyboardKey = cv::waitKey(1);
	}

	if (keyboardKey != 27) {               // if the user did not press esc (i.e. we reached the end of the video)
		cv::waitKey(0);                         // hold the windows open to allow the "end of video" message to show
	}
	// note that if the user did press esc, we don't need to hold the windows open, we can simply let the program end which will close the windows

	return(0);

	}


	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// function to match blob of current frame to existing blobs in frames

void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs) {

	//prdeicts next position for all the blobs from current frames
	for (auto &existingBlob : existingBlobs) {

		existingBlob.blnCurrentMatchFoundOrNewBlob = false;

		existingBlob.predictNextPosition(); // predict next position of blob
	}

	//iterate over all blobs of current frame
	for (auto &currentFrameBlob : currentFrameBlobs) {

		int intIndexOfLeastDistance = 0;
		double dblLeastDistance = 100000.0;

		for (unsigned int i = 0; i < existingBlobs.size(); i++) {

			// if ith blob is still being tracked calulate the distance between centroid of current frame blob and predicted position of blob
			if (existingBlobs[i].blnStillBeingTracked == true) {

				double dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(), existingBlobs[i].predictedNextPosition);

				//if distance < least distance assign the index and the distance
				if (dblDistance < dblLeastDistance) {
					dblLeastDistance = dblDistance;
					intIndexOfLeastDistance = i;
				}
			}
		}

		//add blobs to existing blob if least distance is less than half of diagonal size of blob of current frame
		if (dblLeastDistance < currentFrameBlob.dblCurrentDiagonalSize * 0.5) {
			addBlobToExistingBlobs(currentFrameBlob, existingBlobs, intIndexOfLeastDistance);
		}
		//else add new blob
		else {
			addNewBlob(currentFrameBlob, existingBlobs);
		}

	}

	// checks if we still need to track the car
	for (auto &existingBlob : existingBlobs) {

		
		if (existingBlob.blnCurrentMatchFoundOrNewBlob == false) {
			existingBlob.intNumOfConsecutiveFramesWithoutAMatch++;
		}
		//if car doesnt show up in five consecutive frames its not being tracked
		if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= 5) {
			existingBlob.blnStillBeingTracked = false;
		}

	}

}

///////////////////////////////////////////////////////////////////////////////////////////////////

void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex) {

	existingBlobs[intIndex].currentContour = currentFrameBlob.currentContour;
	existingBlobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect;

	existingBlobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());

	existingBlobs[intIndex].blnStillBeingTracked = true;
	existingBlobs[intIndex].blnCurrentMatchFoundOrNewBlob = true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//add new blob
void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs) {

	currentFrameBlob.blnCurrentMatchFoundOrNewBlob = true;

	existingBlobs.push_back(currentFrameBlob);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// find distance between points 
double distanceBetweenPoints(cv::Point point1, cv::Point point2) {

	int intX = abs(point1.x - point2.x);
	int intY = abs(point1.y - point2.y);

	return(sqrt(pow(intX, 2) + pow(intY, 2)));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//drawa and show contours using standard opencv functions
void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName) {
	cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);

	cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);

	cv::imshow(strImageName, image);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// draw and show contours if the blob is still being tracked and else dont show
void drawAndShowContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName) {

	cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);

	std::vector<std::vector<cv::Point> > contours;

	for (auto &blob : blobs) {
		if (blob.blnStillBeingTracked == true) {
			contours.push_back(blob.currentContour);
		}
	}

	cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);

	cv::imshow(strImageName, image);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
bool checkIfBlobsCrossedTheLine(std::vector<Blob> &blobs, int &intHorizontalLinePosition, int &carCount) {
	bool blnAtLeastOneBlobCrossedTheLine = false;

	for (auto blob : blobs) {

		if (blob.blnStillBeingTracked == true && blob.centerPositions.size() >= 2) {
			int prevFrameIndex = (int)blob.centerPositions.size() - 2;
			int currFrameIndex = (int)blob.centerPositions.size() - 1;

			//this is the case when in video vehicles are moving away from you and not approaching you . You can reverse the condition in other case
			// increase the car count if center position of contour of previous frame is greater than  horizontal line position and that of current frame is less than horizontal line position.
			
			if (blob.centerPositions[prevFrameIndex].y > intHorizontalLinePosition && blob.centerPositions[currFrameIndex].y <= intHorizontalLinePosition) {
				carCount++;
				blnAtLeastOneBlobCrossedTheLine = true;
			}
		}

	}

	return blnAtLeastOneBlobCrossedTheLine;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// draws bounding rectangle on cars
void drawBlobInfoOnImage(std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy) {

	for (unsigned int i = 0; i < blobs.size(); i++) {

		if (blobs[i].blnStillBeingTracked == true) {
			cv::rectangle(imgFrame2Copy, blobs[i].currentBoundingRect, SCALAR_RED, 2);

			//cv::putText(imgFrame2Copy, std::to_string(i), blobs[i].centerPositions.back(), intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness);
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// draws car count on frames
void drawCarCountOnImage(int &carCount, cv::Mat &imgFrame2Copy) {

	int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
	double dblFontScale = (imgFrame2Copy.rows * imgFrame2Copy.cols) / 300000.0;
	int intFontThickness = (int)std::round(dblFontScale * 1.5);

	cv::Size textSize = cv::getTextSize(std::to_string(carCount), intFontFace, dblFontScale, intFontThickness, 0);

	cv::Point ptTextBottomLeftPosition;

	ptTextBottomLeftPosition.x = imgFrame2Copy.cols - 1 - (int)((double)textSize.width * 1.25);
	ptTextBottomLeftPosition.y = (int)((double)textSize.height * 1.25);

	cv::putText(imgFrame2Copy, std::to_string(carCount), ptTextBottomLeftPosition, intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness);

}
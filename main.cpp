#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <vector>
#include <string>
#include <iostream>
#include "common.h"

using namespace cv;

std::string cascade_file = "lbpcascade_frontalface.xml";
std::string image_dir = "img";
std::string result_window_name = "Result";

int main()
{
	CascadeClassifier classifier;
	classifier.load(cascade_file);
	Collect image_files;
	CrawlFolder(image_dir, 1, 1, &image_files);
	std::cout << classifier.isOldFormatCascade();
	std::vector<Rect> objects;
	Mat img, img_gray;
	for(auto& img_file: image_files.file_names)
	{
		img = imread(img_file);
		if(img.channels() == 3)
			cvtColor(img, img_gray, CV_BGR2GRAY);
		else
			img_gray = img.clone();
		classifier.detectMultiScale(img_gray, objects, 1.1, 2, CASCADE_SCALE_IMAGE, Size(30, 30));
		for(auto obj: objects)
		{
			rectangle(img, obj, Scalar(0, 255, 0), 2);
		}
		imshow(result_window_name, img);
		waitKey(0);
	}
	return 0;
}
/*
 * bow_matching_visualizer.h
 *
 *  Created on: November 5, 2013
 *      Author: Siriwat Kasamwattanarote
 */
#include <unistd.h>     // sysconf
#include <sys/stat.h>   // file-directory existing
#include <sys/types.h>  // file-directory
#include <dirent.h>     // file-directory
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <bitset>
#include <cmath>

#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

void readme(const string& app_name)
{ std::cout << " Usage: " << app_name << " <img1> <img2>" << std::endl; }

int main( int argc, char** argv )
{
    /*if( argc != 3 )
    { readme(string(argv[0])); return -1; }*/
	
	cout << "ORB image feature tester" << endl;
	
	Mat img1 = imread("/home/stylix/Pictures/cat1.jpg");
	Mat img2 = imread("/home/stylix/Pictures/cat2.jpg");
	if (img1.empty() || img2.empty())
	{
		printf("Can't read one of the images\n");
		return -1;
	}

	// detecting keypoints
	vector<KeyPoint> keypoints1, keypoints2;

    // Default parameters of ORB
    int nfeatures=500;
    float scaleFactor=1.2f;
    int nlevels=8;
    int edgeThreshold=15; // Changed default (31);
    int firstLevel=0;
    int WTA_K=2;
    int scoreType=ORB::HARRIS_SCORE;
    int patchSize=31;
    //int fastThreshold=20;

    ORB* detector = new ORB(
    nfeatures,
    scaleFactor,
    nlevels,
    edgeThreshold,
    firstLevel,
    WTA_K,
    scoreType,
    patchSize );

    detector->detect(img1, keypoints1);
	detector->detect(img2, keypoints2);
    std::cout << "img1 Found " << keypoints1.size() << " Keypoints " << std::endl;
	std::cout << "img2 Found " << keypoints2.size() << " Keypoints " << std::endl;

    Mat out1, out2;
    drawKeypoints(img1, keypoints1, out1, Scalar::all(255));
	drawKeypoints(img2, keypoints2, out2, Scalar::all(255));

    //imshow("Kpts", out);
	cout << "Write to: " << "/home/stylix/Pictures/cat1_pt.png" << endl;
	imwrite("/home/stylix/Pictures/cat1_pt.png", out1);
	cout << "Write to: " << "/home/stylix/Pictures/cat2_pt.png" << endl;
    imwrite("/home/stylix/Pictures/cat2_pt.png", out2);

	// computing descriptors
	OrbDescriptorExtractor extractor;
	Mat descriptors1, descriptors2;
	extractor.compute(img1, keypoints1, descriptors1);
	extractor.compute(img2, keypoints2, descriptors2);

	// matching descriptors
	
	BFMatcher* matcher = new BFMatcher(NORM_HAMMING);
	vector<DMatch> matches;
	matcher->match(descriptors1, descriptors2, matches);

	// drawing the results
	Mat img_matches;
	drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
	
	cout << "Write to: " << "/home/stylix/Pictures/cat_match.png" << endl;
    imwrite("/home/stylix/Pictures/cat_match.png", img_matches);
	
	// Release memory
	delete detector;
	delete matcher;
	
	return 0;
}

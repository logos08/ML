// razmechatel.cpp: определяет точку входа для консольного приложения.
//
#define _CRT_SECURE_NO_WARNINGS
#define MAX_OBJECTS 50
#include "stdafx.h"
#include <cxcore.h>
#include <cv.h>
#include <highgui.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "dirent.h"


using namespace std;
using namespace cv;

string IntToString(int a)
{	
	std::ostringstream temp;
	temp << a;
	return temp.str();
}

struct obj_mark
{
	int status;
	string class_name;
	Point p1;
	Point p2;
};

Mat img;

bool show_img(Mat img_sample, void* userdata)
{
	Mat new_img = img_sample.clone();
	obj_mark * marks = (obj_mark *)userdata;
	bool need_to_draw = false;

	for (int i = 0; i < MAX_OBJECTS; i++)
	{
		if (marks[i].status == 1)
		{
			need_to_draw = true;
		}
	}
	if(!need_to_draw)return 0;
	
	for (int i = 0; i<MAX_OBJECTS; i++)
	{
		if (marks[i].status>0)
		{
			rectangle(new_img, Rect(marks[i].p1, marks[i].p2), Scalar(0,0,255));
			imshow("My Window", new_img);
			//waitKey(0);
				
		}
	}
	//cout << 1;
	return 0;
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	obj_mark * marks = (obj_mark *)userdata;
	int obj_num=-1;
	for (int i=0;i<MAX_OBJECTS;i++)
	{
		if (marks[i].status<2)
		{
			obj_num = i;			
			break;
		}
	}

	show_img(img, marks);

	if (obj_num != -1)
	{
		if (event == EVENT_LBUTTONDOWN)
		{
			cout << endl << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" ;
			if (marks[obj_num].status == 0)
			{
				marks[obj_num].status = 1;
				marks[obj_num].p1.x = x;
				marks[obj_num].p1.y = y;
			}
		}
		else if (event == EVENT_RBUTTONDOWN)
		{
			cout << endl << "Right button of the mouse is clicked - Clear all!";
			for (int i = 0; i < MAX_OBJECTS; i++)
			{
				marks[i].status = 0;
			}
			imshow("My Window", img);
		}
		else if (event == EVENT_LBUTTONUP)
		{
			cout << endl << "Left button of the mouse released - position (" << x << ", " << y << ")" ;
			if (marks[obj_num].status == 1)
			{
				marks[obj_num].status = 2;
			}
		}
		else if (event == EVENT_MOUSEMOVE)
		{
			//cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;
			marks[obj_num].p2.x = x;
			marks[obj_num].p2.y = y;
		}
	}
}



int main(int argc, char *argv[])
{
	// Read image from file 
	//string d_name="F:/Work_Severstal/casks/111/helmet_blue_img/";
	string str1;
	obj_mark marks[MAX_OBJECTS];
	char path_to_imgs[MAX_PATH];
	char classname[MAX_PATH];
	
	sprintf(path_to_imgs, "%s", argv[1]);
	sprintf(classname, "%s", argv[2]);
	string spath_to_imgs = path_to_imgs;

	for (int i = 0; i < MAX_OBJECTS; i++)
	{
		marks[i].class_name = classname;
		marks[i].status = 0;
		marks[i].p1.x = 0;
		marks[i].p1.y = 0;
		marks[i].p2.x = 0;
		marks[i].p2.y = 0;
	}

	//set the callback function for any mouse event
	namedWindow("My Window", 1);
	setMouseCallback("My Window", CallBackFunc, &marks[0]);

	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir(spath_to_imgs.c_str())) != NULL)
	{
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) 
		{
			str1 = spath_to_imgs + ent->d_name;

			cout << endl << str1;

			img = imread(str1);
			if (!img.empty())
			{
				imshow("My Window", img);
				
								
				waitKey(0);
								
				FILE* textfile = fopen((str1.substr(0, str1.size() - 4)+".xml").c_str(), "w");
				str1 = ent->d_name;

				fputs("<annotation>\n", textfile);
				fputs(" <folder>Data</folder>\n", textfile);	
				fputs((" <filename>"+ str1 +"</filename>\n").c_str(), textfile);
				fputs(" <source>\n", textfile);
				fputs("  <database>Data</database>\n", textfile);
				fputs("  <annotation>helmet</annotation>\n", textfile);
				fputs("  <image>flickr</image>\n", textfile);
				fputs(" </source>\n", textfile);
				fputs(" <size>\n", textfile);
				fputs("  <width>300</width>\n", textfile);
				fputs("  <height>300</height>\n", textfile);
				fputs("  <depth>3</depth>\n", textfile);
				fputs(" </size>\n", textfile);
				fputs(" <segmented>0</segmented>\n", textfile);

				for (int i = 0; i < MAX_OBJECTS; i++)
				{
					if (marks[i].status==0)break;
					fputs(" <object>\n", textfile);
					fputs(("  <name>"+ marks[i].class_name +"</name>\n").c_str(), textfile);
					
					fputs("  <pose>Unspecified</pose>\n", textfile);
					fputs("  <truncated>0</truncated>\n", textfile);
					fputs("  <difficult>0</difficult>\n", textfile);
					fputs("  <bndbox>\n", textfile);
					fputs(("   <xmin>" + IntToString(min(marks[i].p1.x, marks[i].p2.x)) + "</xmin>\n").c_str(), textfile);
					fputs(("   <ymin>" + IntToString(min(marks[i].p1.y, marks[i].p2.y)) + "</ymin>\n").c_str(), textfile);
					fputs(("   <xmax>" + IntToString(max(marks[i].p1.x, marks[i].p2.x)) + "</xmax>\n").c_str(), textfile);
					fputs(("   <ymax>" + IntToString(max(marks[i].p1.y, marks[i].p2.y)) + "</ymax>\n").c_str(), textfile);
					fputs("  </bndbox>\n", textfile);
					fputs(" </object>\n", textfile);
				}

				fputs("</annotation>", textfile);

				fclose(textfile);

				for (int i = 0; i < MAX_OBJECTS; i++)
				{
					marks[i].status = 0;
				}
			}
		}
		closedir(dir);
	}
	else 
	{
		/* could not open directory */
		perror("");
		return EXIT_FAILURE;
	}
	
	return 0;

}


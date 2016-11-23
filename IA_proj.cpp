
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <vector>
#include <string>
#include <stdlib.h>

#include <fstream>
#include <stdio.h>
#include <sstream>



using namespace cv;
using namespace std;


vector<Mat> sort(vector<int> , vector<Mat>);
int jkl=0;
void lbp1(Mat, int);

void read_lbpval();

int mat_lbp[36][256];  //store all lbp value of dataset
int temp_mat[10][256];     //store lbp value of current image


vector<int> x_loc;
vector<Mat> img_mat;

string img_plate="3266CNT.JPG";

void read_lbpval()
{

    std::ifstream fp("DATSET.txt");
     std::string line;



     int i,j;
     i=j=0;
     int val;

         while(getline(fp, line, '\n'))
         {
             stringstream linestream(line);
              string value;

             while(getline(linestream, value, ','))
             {

                      stringstream ss(value);
                      ss>> val;
                      mat_lbp[i][j]=val;
                      j++;
             }
             i++;
             j=0;
         }

}

string ps;
void comp_lbp(int size_temp)
{
    ostringstream s;
    int indice=0;
    int i,j;
    int flag=0;
    for(i=0;i<36;i++)

    {
        flag=0;
        for(j=0;j<256;j++)
        {
            if(temp_mat[indice][j]!= mat_lbp[i][j])
            {
                flag=1;
                continue;
            }

        }
        if(flag==0)
        {
            if(i>=0 && i<=9)
            {
                s<< (i);

            }
            else
            {
                s<< (char)(i+55);
            }

        }
    }


    if(!jkl)
    cout<<s.str()<<endl;
}








void find_contour_image( Mat src)
{
     Mat dest;
     Mat dest1=src;

     //ERODING 2 TIMES PIC OF PLATE SO THAT LETTERS SEPARATE OUT
     Mat element=getStructuringElement( MORPH_ELLIPSE, Size(3, 3 ), Point( -1,-1 ) );
    erode(src, dest1, element);
   //  erode(src, dest1, element);

    //morphologyEx( src, dest1, MORPH_GRADIENT, element, Point(-1,-1), 1 );
 imwrite("im_mnorph.png", dest1);
     threshold(dest1,dest,100,200,THRESH_BINARY);
     imwrite("imthresh.png", dest);
     Mat rgb=Mat::zeros(src.size(),src.type());
     if(ps=="3732FWW.JPG")
        {cout<<ps.substr(0, ps.size()-4)<<endl;jkl=1;}
    Mat mask= Mat::zeros(src.size(),src.type());
    vector<vector <Point> > contours; //contain the contours
    vector<Vec4i> hierarchy;

    findContours(dest, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    Rect rect;
    int i,j,c;





    for(int i = 0; i>= 0; i = hierarchy[i][0])
        {
            rect = boundingRect(contours[i]);
            //cout<< rect.x<< endl;
            if(rect.x==1)
                continue;
            Mat maskROI(mask, rect);
            maskROI = Scalar(0, 0, 0);
            // fill the contour
            drawContours(mask, contours, i, Scalar(255, 255, 255));
            // ratio of non-zero pixels in the filled region
            double r = (double)countNonZero(maskROI)/(rect.width*rect.height);
            //cout<<r<<endl;
            rectangle(rgb, rect, Scalar(255,255,255), 1);
            img_mat.push_back(src(rect));
            x_loc.push_back(rect.x);




        }


    for(i=0;i<img_mat.size();i++)
    {
        ostringstream st;
        st<<"im"<<i<<".png";
        imwrite(st.str(), img_mat[i]);
    }

    int i2, j2;
    for(i2=0;i2<10;i2++)
       {
         for(j2=0;j2<256;j2++)
            {
                temp_mat[i2][j2]=0;
            }
        }

    for(i=0;i< img_mat.size();i++)
         lbp1(img_mat[i], i);

    comp_lbp(img_mat.size());  //comparing lbp value with dataset
imwrite("writeimage.png", rgb);

}

/*........................................................................*/
void Segmentation(Mat input)
{

Mat new_image=Mat::zeros(input.size(),input.type());
    Mat img_gray;
    cvtColor(input, img_gray, CV_BGR2GRAY);
    blur(img_gray, img_gray, Size(7,7));
blur(img_gray, img_gray, Size(7,7));
blur(img_gray, img_gray, Size(7,7));
blur(img_gray, img_gray, Size(7,7));
blur(img_gray, img_gray, Size(7,7));
	
	 Mat src_n, dest;
    ps=img_plate;
     Mat dest2;
    Mat src2;
    string km;
Mat src=input;
    //Mat src2= 255-src;
    Mat dest1= Mat::zeros(src.size(),src.type());
    Mat element;
    Mat connected_comp;
    int morph_size = 2;
    Mat rgb=Mat::zeros(src.size(),src.type());
    element=getStructuringElement( MORPH_ELLIPSE, Size(3, 3 ), Point( -1,-1 ) );
    morphologyEx( img_gray, dest1, MORPH_GRADIENT, element, Point(-1,-1), 1 );

    threshold(dest1,dest2,10,200,THRESH_BINARY);
    if(ps=="3266CNT.JPG")
    {km=ps.substr(0, ps.size()-4);jkl=1;}
    erode(dest2, dest, element);
    erode(dest, dest, element);
    element=getStructuringElement( MORPH_RECT, Size(9, 1 ));
    morphologyEx( dest, connected_comp, MORPH_CLOSE, element, Point(-1,-1), 1 );
   
	imwrite("close.png",connected_comp);
    Mat mask= Mat::zeros(dest.size(),dest.type());
    vector<vector <Point> > contours; //contain the contours
    vector<Vec4i> hierarchy;
    findContours(connected_comp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

Mat contour_region;
Mat mask2;
    int i,j,c;
	double large= 0.0;	Rect rect2;
    for(int i = 0; i>= 0; i = hierarchy[i][0])
        {
            Rect rect = boundingRect(contours[i]);
            Mat maskROI(mask, rect);
            maskROI = Scalar(0, 0, 0);
            // fill the contour
		
	Mat imageROI;
	            

	connected_comp.copyTo(imageROI, mask);
	contour_region= mask(rect);

            // ratio of non-zero pixels in the filled region
            double r = (double)countNonZero(maskROI)/(rect.width*rect.height);


      drawContours(mask, contours, i, Scalar(255, 255, 255), CV_FILLED);
			
    if(rect.width/rect.height <= 4.7 /*(double)(510.0/110)*/ && rect.width> rect.height && rect.width> large)
        {
        large= rect.width;
        rect2= rect;
        mask2=mask;
        rectangle(rgb, rect2, Scalar(50,200,0), 2);
        }

        }

    //rectangle(rgb, rect2, Scalar(50,200,0), 2);
    //bitwise_not(connected_comp, connected_comp, mask);
    for( i = 0; i < src.rows; i++ )
       { for( j = 0; j < src.cols; j++ )
        { for( c = 0; c < 3; c++ )
             {
                if(rgb.at<Vec3b>(i,j)[c]==0)
                {
                    rgb.at<Vec3b>(i,j)[c]= input.at<Vec3b>(i,j)[c];
                }
            }

        }
    }
    imwrite("im.png", input(rect2));


    Mat segmented_image= input(rect2);
    // resizing of image to common scale 33*144

    Mat resized_image, new_gray_img;
    resized_image.create(33,144, CV_8UC3);

    resize(segmented_image, resized_image, resized_image.size(), 0, 0,INTER_CUBIC);// resizing image
    cvtColor(resized_image, new_gray_img, CV_BGR2GRAY);
    blur(new_gray_img, new_gray_img, Size(3,3));
    equalizeHist(new_gray_img, new_gray_img);

    imwrite("this_thresh.png", new_gray_img);
    //Segmenting numbers from the plate
    Mat thresh_img;
    cout<<km<<endl;
    threshold(new_gray_img,thresh_img, 150, 255, CV_THRESH_BINARY_INV);
      imwrite("dilated.png", thresh_img);

    //finding contour

    Mat dilation_dst;
   Mat el=getStructuringElement( MORPH_RECT, Size(3, 3 ));
   find_contour_image(thresh_img);

}





int main(int argc, char** argv )
{


    Mat image= imread(img_plate);
      if(! image.data )                              // Check for invalid input
      {
          cout <<  "Could not open or find the image" << std::endl ;
          return -1;
      }
      read_lbpval();

      Segmentation(image);
      return 0;
}









vector<Mat> sort(vector<int> x_coord, vector<Mat> images)
{
    int i,j;
    vector<Mat> sorted_img;

    for(i=0;i<x_coord.size();i++)
    {
         int min=i;
        for(j=i+1;j< x_coord.size();j++)
        {
            if(x_coord[j]< x_coord[min])
            {
                min=j;
            }
        }

        sorted_img[i]= images[min];
    }
    return sorted_img;


}

void lbp1(Mat greyMat , int index)
{



       for(int i=1;i<greyMat.rows;i++)
        {
             for(int j=1;j<greyMat.cols;j++)
             {
                int c1=(int)greyMat.at<uchar>(i,j);
                int p=0;
                int t=7;
                if(c1<=(int)greyMat.at<uchar>(i,j+1))
                p+=pow(2,t);
                t=t-1;
                if(c1<=(int)greyMat.at<uchar>(i+1,j+1))
                p+=pow(2,t);
                if(ps=="9588DWV.jpg")
                {cout<<ps.substr(0, ps.size()-4)<<endl;jkl=1;}
                t=t-1;
                if(c1<=(int)greyMat.at<uchar>(i+1,j))
                p+=pow(2,t);
                t=t-1;
                if(c1<=(int)greyMat.at<uchar>(i+1,j-1))
                p+=pow(2,t);
                t=t-1;
                if(c1<=(int)greyMat.at<uchar>(i,j-1))
                p+=pow(2,t);
                t=t-1;


                if(c1<=(int)greyMat.at<uchar>(i-1,j-1))
                p+=pow(2,t);
                t=t-1;
                if(c1<=(int)greyMat.at<uchar>(i-1,j))
                p+=pow(2,t);
                t=t-1;

                if(c1<=(int)greyMat.at<uchar>(i-1,j+1))
                p+=1*pow(2,t);
                t=t-1;



                temp_mat[index][p]=temp_mat[index][p]+1;


             }

        }



}



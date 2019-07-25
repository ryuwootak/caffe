#include <stdio.h>
///rrrrrr
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <time.h>
using namespace cv;
using namespace std;
void padding_next(int row, int column, int padding, int **res, int **p);
void max_pooling(int width,int column,int filtersize,int stride,int** input,int** max);
int	***padding_handling(int channel, int row, int column, int padding, int	***p);
void Convolution(int width, int column, int filtersize, int stride, int** res, int** filter, int** conv);
int	***Edge(int channel, int width, int column, int stride, int ***input, int **filter);
int ***Blur(int channel, int width, int column, int stride, int ***input, int **filter);
int	***Maxpooling(int channel, int width, int column, int filtersize, int padding, int stride, int ***input);
//7월 27일 수정 예정

//2Â÷¿ø ¹è¿­Àº **, 3Â÷¿ø ¹è¿­Àº ***Æ÷ÀÎÅÍ ÀÌ¿ë
//1¹ø padding ÀÌ¹ÌÁö°ª³ÖÀœ
int	***padding_handling(int channel, int row, int column, int padding, int	***p)
{
	int ***ans; 
	
	ans = (int***)malloc(channel*sizeof(int**));
	
	for (int i = 0;i<channel;i++){
		*(ans+i) = (int**)malloc((row+(2*padding))*sizeof(int*));
		for (int j=0;j<(row+(2*padding));j++){
			*(*(ans+i)+j)=(int*)malloc((column +(2*padding))*sizeof(int));
		}
	} //µ¿ÀûÇÒŽç

	for (int i = 0;i< channel;i++){
		padding_next(row,column,padding,p[i],ans[i]);
	} //padding/   

	Mat image_padding(row+2*padding,column+2*padding,CV_8UC3);

	for (int z = 0;z<channel;z++){
		for (int y = 0;y<image_padding.rows;y++){
			for (int x = 0; x<image_padding.cols;x++){
				image_padding.at<cv::Vec3b>(y, x)[z]=ans[z][y][x];
			}
		}
	}//µ¿ÀûÇÒŽçµÈ ans¿¡ ÀÌ¹ÌÁö°ª ŽëÀÔ

    return ans;
}
//padding 
void padding_next(int row, int column, int padding, int **res, int **p)
{
	int i,j;
	
	for(i=0;i<padding;i++){
		for(j=0;j<(column+(2*padding));j++){
			p[i][j] = 0;
		}
	}

    for(i=padding;i<(row+padding);i++){
		for(j=0 ; j<padding; j++){
			p[i][j]=0;
		}

        for(j=padding;j<column+padding;j++){
			p[i][j] = res[i-padding][j-padding];
		}
		
		for(j=column+padding;j<column+2*padding ; j++){
			p[i][j] = 0;
		}
	}

    for(i=row+padding;i<row+2*padding;i++){
		for(j=row+padding;j<(column+(2*padding));j++){
			p[i][j] = 0;
		}
	}
} //³× ±ž¿ªÀž·Î ³ªŽ²Œ­ padding°ª 0Àž·Î ³ÖÀœ

//convolution-1
void Convolution(int width, int column, int filtersize, int stride, int** res, int** filter, int** conv)
{
    int i,j,k,l,m,sum;
	
	for(l=0;l<width;l++){
		for(m=0;m<column;m++){
			sum = 0;
			for(i=0; i<filtersize ; i++){
				for(j=0 ; j<filtersize; j++){
					sum = sum + (res[l*stride+i][m*stride+j] * filter[i][j]);
				}
			}
			conv[l][m] = sum;
		}
	}
}

//edge-2
int	***Edge(int channel, int width, int column, int stride, int ***input, int **filter)
{
	int ***conv;
	int i, j;
	int x,y,z,temp;

	conv=(int***)malloc(channel*sizeof(int**));
	for(i=0;i<channel;i++){
		*(conv+i)=(int**)malloc(width*sizeof(int*));
		for(j= 0;j< width;j++){
			*(*(conv+i)+j)=(int*)malloc(column*sizeof(int));
		}
	}

	for (i=0;i<channel;i++){
		Convolution(width,column,3,stride,input[i],filter,conv[i]);
	}

    Mat image_edge(width,column,CV_8UC3);

	for(z=0;z<channel;z++){
		for(y=0;y<image_edge.rows;y++){
			for(x=0;x<image_edge.cols;x++){
				temp = conv[z][y][x];
				if(temp>=256){
					temp=255;
				}
				else if(temp<0){
					temp=0;
				}
				image_edge.at<cv::Vec3b>(y, x)[z] = temp;
			}
		}
	}
	
    namedWindow("Edge Detect",WINDOW_AUTOSIZE);
	imshow("Edge Detect",image_edge);
	
	return conv;
}

//blur-3
int ***Blur(int channel, int width, int column, int stride, int ***input, int **filter)
{
    int ***conv;
	int i, j;
	int x,y,z, temp;
	
	conv=(int***)malloc(channel*sizeof(int**));

	for(i=0;i<channel;i++){
		*(conv+i)=(int**)malloc(width*sizeof(int*));
		for(j=0;j<width;j++){
			*(*(conv+i)+j)=(int*)malloc(column*sizeof(int));
		}
	}

	for(i=0;i<channel;i++){
		Convolution(width,column,5,stride,input[i],filter,conv[i]);
	}

    Mat image_blur(width,column,CV_8UC3);

	for(z=0;z<channel;z++){
		for(y=0;y<image_blur.rows;y++){
			for(x=0;x<image_blur.cols;x++){
				temp=conv[z][y][x]/256;
				if(temp>255){
					temp=255;
				}
				else if(temp<0){
					temp=0;
				}
				image_blur.at<cv::Vec3b>(y, x)[z] = temp;
			}
		}
	}

    namedWindow("blur",WINDOW_AUTOSIZE);
	imshow("blur",image_blur);

	return conv;
}

//maxpooling-4
int	***Maxpooling(int channel, int width, int column, int filtersize, int padding, int stride, int ***input)
{
	int ***max;
	int i, j;
	int x,y,z;

	max=(int***)malloc(channel*sizeof(int**));

	for(i=0;i<channel;i++){
		*(max+i)=(int**)malloc(width*sizeof(int*));
		for(j= 0;j<width;j++){
			*(*(max+i)+j) = (int*)malloc(column*sizeof(int));
		}
	}

	for(i= 0;i<channel;i++){
		max_pooling(width,column,filtersize,stride,input[i],max[i]);
	}

    Mat image_max(width,column,CV_8UC3);

	for(z=0;z<channel;z++){
		for(y= 0;y<image_max.rows;y++){
			for(x=0;x<image_max.cols;x++){
				image_max.at<cv::Vec3b>(y,x)[z]=max[z][y][x];
			}
		}
	}

    namedWindow("Maxpooling",WINDOW_AUTOSIZE);
	imshow("Maxpooling",image_max);
	
	return max;
}

void max_pooling(int width,int column,int filtersize,int stride,int** input,int** max)
{
    int i,j,k,m,n,limit;
	
	for(m=0;m<width;m++){
		for(n=0;n<column;n++){
			max=0;
			for(i=0;i<filtersize;i++){
				for(j=0;j<filtersize;j++){
					if(limit<input[m*stride+i][n*stride+j]){
						limit=input[m*stride+i][n*stride+j];
					}
				}
			}
			max[m][n]=limit;
		}
	}
}

int main()
{
	Mat image;

	int ***p, ***padding_init,***conv,**filter, ***act;;
	int row,col,channel,padding,stride,filtersize,width,column,select=0;

	int edge[3][3] = {-1,-1,-1,-1,8,-1,-1,-1,-1};
	int blur[5][5] = {1,4,6,4,1,4,16,24,16,4,6,24,36,24,6,4,16,24,16,4,1,4,6,4,1};    

	cout<<"Padding value ?" ;
	scanf(" %d", &padding);

	cout <<"Stride value ? ";
	scanf(" %d", &stride);

	image = imread("lalala.jpg",IMREAD_COLOR);

	if(image.empty()){

		cout << "No image" << endl;
		return -1;
	}

	row = image.rows;
	col = image.cols;
	channel = image.channels();

	p = (int***)malloc(channel*sizeof(int**));

	for (int i=0;i<channel;i++){
		*(p+i) = (int**)malloc(row*sizeof(int*));
			
		for (int j=0;j<row;j++){
			*(*(p+i)+j)=(int*)malloc(col*sizeof(int));
		}
	}
	
	for (int z=0;z<channel;z++){
		for (int y=0; y<row; y++){
			for (int x=0; x<col; x++){
				p[z][y][x] = image.at<cv::Vec3b>(y, x)[z]; 
			}
		}
	}

	padding_init = padding_handling(channel, row, col, padding, p);

	cout<<" 1.Edge 2.Blur 3.max_pooling  : ";
	scanf(" %d",&select);   

	switch(select){
		case 1 :
			filtersize = 3;
			filter=(int**)malloc(channel*sizeof(int*));

			width=((row-filtersize+2*padding)/stride)+1;
			column=((col-filtersize+2*padding)/stride)+1;

			for (int i=0; i < filtersize; i++){
				*(filter + i) = (int*)malloc(filtersize*sizeof(int));}

			for (int i = 0; i < filtersize;i++){
				for (int j = 0; j < filtersize;j++){
					filter[i][j] = edge[i][j];}
												}

			conv = Edge(channel,width,column,stride,padding_init,filter);
			break;  
        
		case 2:
			filtersize = 5;
			filter = (int**)malloc(filtersize*sizeof(int*));

			width = ((row-filtersize+2*padding)/stride)+1;
			column = ((col-filtersize+2*padding)/stride)+1;
			
			for (int i=0;i<filtersize;i++){
				*(filter + i) = (int*)malloc(filtersize*sizeof(int));
			}
			
			for (int i=0;i <filtersize; i++){
				for (int j = 0; j <filtersize; j++){
					filter[i][j] = blur[i][j];
				}
			}    
			
			conv = Blur(channel,width,column,stride,padding_init,filter);
			break;        

		case 3:
			
			cout <<"filtersizežŠ ÀÔ·ÂÇØÁÖŒŒ¿ä F*F¿¡Œ­ F°ª : "<<endl;
			scanf("%d", &filtersize);
			
			width = ((row-filtersize+2*padding)/stride)+1;
			column = ((col-filtersize+2*padding)/stride)+1;
			conv = Maxpooling(channel,width,column,filtersize,padding,stride,padding_init);
			break;
			
		default:
			cout << "ÀßžøÀÔ·ÂÇÏ¿ŽœÀŽÏŽÙ."<<endl; break;
	}
	   	
	waitKey(0);

	//µ¿ÀûÇÒŽç ÇØÁŠ
		for (int i = 0; i < channel; i++){
			for (int j = 0; j < (width + (2*padding)); j++){
				free(*(*(padding_init+ i) + j));
			}
			free(*(padding_init+ i));
		}
		free(padding_init);
		
	for (int i = 0;i< filtersize;i++){
			free(*(filter+i));}
		free(filter);   

	    for (int i = 0;i < channel; i++){
			for (int j = 0; j < width; j++){
				free(*(*(conv+i)+j));
			}
			free(*(conv + i));
		}
		free(conv);
		
	for (int i = 0; i < channel; i++){
		for (int j = 0; j < width; j++){
			free(*(*(p+i)+j));
		}
		free(*(p+i));
	}
	free(p);    
	
	return 0;
}


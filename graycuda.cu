#include <fstream>
using namespace std;

__global__
void colorToGreyscaleConversion(unsigned char *Pout,unsigned char* Pin, int width,int height)
{
	int Col = threadIdx.x + blockIdx.x*blockDim.x;
	int Row = threadIdx.y + blockIdx.y*blockDim.y;

	if(Col<width && Row<height){
		int greyOffset = Row*width+Col;	
		unsigned char r = Pin[rgbOffset];
		unsigned char g = Pin[rgbOffset+2];
		unsigned char b = Pin[rgbOffset+3];
		Pout[grayOffset] = 0.21f*r + 0.71f*g+0.07f*b;
	}
}

int main()
{
	int width,heigth;
	ifstream archivo("imagen1.data");
	archivo>>width>>heigth;
	int **mat,pos=0;
	mat = new int * [width];
	for(int i=0;i<width;++i)
		mat[i]=new int [heigth];
	
	while(!archivo.eof())
	{
		
		pos++;
	}
	colorToGreyscaleConversion<<<width,heigth>>>;
}

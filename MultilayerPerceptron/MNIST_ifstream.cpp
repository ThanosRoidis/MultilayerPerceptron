#include "MNIST_ifstream.h"

MNIST_ifstream::MNIST_ifstream(char *filename)
{
	isBigEndian = is_big_endian();
	
	in = new ifstream(filename, ios::binary);
}

MNIST_ifstream::~MNIST_ifstream()
{
	in->close();
	delete in;
	//dtor
}

bool MNIST_ifstream::is_big_endian()
{
	union
	{
		uint32_t i;
		char c[4];
	} bint = { 0x01020304 };

	return bint.c[0] == 1 ? true : false;
}



int MNIST_ifstream::readInt()
{
	if (in->is_open())
	{
		int num;
		in->read((char *)&num, sizeof(int));
		if (isBigEndian)
		{
			return num;
		}
		else
		{
			return ((num >> 24) & 0xff) | // move byte 3 to byte 0
				((num << 8) & 0xff0000) | // move byte 1 to byte 2
				((num >> 8) & 0xff00) | // move byte 2 to byte 1
				((num << 24) & 0xff000000); // byte 0 to byte 3
		}
	}
	return 0;
}

double MNIST_ifstream::readDouble(){
	if (in->is_open())
	{
		double num;
		in->read((char *)&num, sizeof(double));
		if (isBigEndian)
		{
			return num;
		}
		else
		{
			
			double swapped;
			unsigned char *dst = (unsigned char *)&swapped;
			unsigned char *src = (unsigned char *)&num;

			dst[0] = src[7];
			dst[1] = src[6];
			dst[2] = src[5];
			dst[3] = src[4];
			dst[4] = src[3];
			dst[5] = src[2];
			dst[6] = src[1];
			dst[7] = src[0];

			return swapped;
		}
	}
	return 0;
}

unsigned char MNIST_ifstream::readByte()
{
	if (in->is_open())
	{
		unsigned char c;
		in->read((char *)&c, sizeof(char));
		return c;
	}
	else
	{
		return 0;
	}
}

void MNIST_ifstream::close()
{
	if (in->is_open())
	{
		in->close();
	}
}

bool MNIST_ifstream::is_open(){
	return in->is_open();
}


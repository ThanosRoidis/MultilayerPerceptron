#ifndef MNIST_IFSTREAM_H
#define MNIST_IFSTREAM_H

#include <iostream>
#include <fstream>
#include <stdint.h>
#include <vector>
#include <ctime>

using namespace std;

class MNIST_ifstream
{
private:
	bool isBigEndian;
	ifstream *in;
	bool is_big_endian();

public:

	MNIST_ifstream(char *filename);

	virtual ~MNIST_ifstream();

	int readInt();

	unsigned char readByte();

	double readDouble();

	void close();

	bool is_open();

	static void readData_MNIST_PCA(vector<vector<double>> &trainImages, vector<unsigned char> &trainLabels, vector<vector<double>> &testImages, vector<unsigned char> &testLabels, int reducedDimensions)
	{

		clock_t startTime, endTime;
		startTime = clock();

		char trainImagesFilename[] = "trainImagesPCA.bin";
		char testImagesFilename[] = "testImagesPCA.bin";
		char trainLabelsFilename[] = "train-labels.idx1-ubyte";
		char testLabelsFilename[] = "t10k-labels.idx1-ubyte";


		MNIST_ifstream trainImagesInput(trainImagesFilename);
		MNIST_ifstream trainLabelsInput(trainLabelsFilename);
		MNIST_ifstream testImagesInput(testImagesFilename);
		MNIST_ifstream testLabelsInput(testLabelsFilename);


		int numberOfTrainingSamples, numberOfTestingSamples;
		//
		std::cout << "TRAIN IMAGES(" << trainImagesFilename << ")" << endl;
		std::cout << " Magic Number:" << trainImagesInput.readInt() << endl;
		numberOfTrainingSamples = trainImagesInput.readInt();
		int dims = trainImagesInput.readInt();
		std::cout << " Number Of Samples:" << numberOfTrainingSamples << endl;
		std::cout << " Dimensions: " << dims << endl;

		std::cout << "TRAIN LABELS(" << trainLabelsFilename << ")" << endl;
		std::cout << " Magic Number:" << trainLabelsInput.readInt() << endl;
		std::cout << " Number Of Samples:" << trainLabelsInput.readInt() << endl << endl;


		trainImages.resize(numberOfTrainingSamples);
		trainLabels.resize(numberOfTrainingSamples);
		int imagePixels = dims;
		for (int i = 0; i < numberOfTrainingSamples; i++)
		{
			trainImages[i].resize(imagePixels);
			for (int j = 0; j < imagePixels; j++){
				trainImages[i][j] = trainImagesInput.readDouble();
			}

			trainLabels[i] = trainLabelsInput.readByte();
		}


		//Read test input

		std::cout << "TEST IMAGES(" << testImagesFilename << ")" << endl;
		std::cout << " Magic Number:" << testImagesInput.readInt() << endl;
		numberOfTestingSamples = testImagesInput.readInt();
		std::cout << " Number Of Samples:" << numberOfTestingSamples << endl;
		std::cout << " Dims:" << testImagesInput.readInt() << endl;


		std::cout << "TEST LABELS(" << testLabelsFilename << ")" << endl;
		std::cout << " Magic Number:" << testLabelsInput.readInt() << endl;
		std::cout << " Number Of Samples:" << testLabelsInput.readInt() << endl << endl;


		testImages.resize(numberOfTestingSamples);
		testLabels.resize(numberOfTestingSamples);
		for (int i = 0; i < numberOfTestingSamples; i++)
		{
			testImages[i].resize(imagePixels);
			for (int j = 0; j < imagePixels; j++){
				testImages[i][j] = testImagesInput.readDouble();
			}
			testLabels[i] = testLabelsInput.readByte();

		}

		for (int i = 0; i < trainImages.size(); i++){
			trainImages[i].resize(reducedDimensions);
		}

		for (int i = 0; i < testImages.size(); i++){
			testImages[i].resize(reducedDimensions);
		}

		endTime = clock();
		std::cout << "Reading Time: " << double(endTime - startTime) / CLOCKS_PER_SEC << "s " << endl << endl;

	}





	static void readData_MNIST(vector<vector<double>> &trainImages, vector<unsigned char> &trainLabels, vector<vector<double>> &testImages, vector<unsigned char> &testLabels)
	{
		clock_t startTime, endTime;
		startTime = clock();


		char trainImagesFilename[] = "train-images.idx3-ubyte";
		char testImagesFilename[] = "t10k-images.idx3-ubyte";
		char trainLabelsFilename[] = "train-labels.idx1-ubyte";
		char testLabelsFilename[] = "t10k-labels.idx1-ubyte";


		MNIST_ifstream trainImagesInput(trainImagesFilename);
		MNIST_ifstream trainLabelsInput(trainLabelsFilename);
		MNIST_ifstream testImagesInput(testImagesFilename);
		MNIST_ifstream testLabelsInput(testLabelsFilename);


		int numberOfTrainingSamples, numberOfTestingSamples;
		//
		std::cout << "TRAIN IMAGES(" << trainImagesFilename << ")" << endl;
		std::cout << " Magic Number:" << trainImagesInput.readInt() << endl;
		numberOfTrainingSamples = trainImagesInput.readInt();
		int width = trainImagesInput.readInt();
		int height = trainImagesInput.readInt();
		std::cout << " Number Of Samples:" << numberOfTrainingSamples << endl;
		std::cout << " Width:" << width << endl;
		std::cout << " Height:" << height << endl << endl;

		std::cout << "TRAIN LABELS(" << trainLabelsFilename << ")" << endl;
		std::cout << " Magic Number:" << trainLabelsInput.readInt() << endl;
		std::cout << " Number Of Samples:" << trainLabelsInput.readInt() << endl << endl;


		trainImages.resize(numberOfTrainingSamples);
		trainLabels.resize(numberOfTrainingSamples);
		int imagePixels = width * height;
		for (int i = 0; i < numberOfTrainingSamples; i++)
		{
			trainImages[i].resize(imagePixels);
			for (int j = 0; j < imagePixels; j++){
				trainImages[i][j] = (double)trainImagesInput.readByte() / 255;
			}

			trainLabels[i] = trainLabelsInput.readByte();
		}

		//Read test input

		std::cout << "TEST IMAGES(" << testImagesFilename << ")" << endl;
		std::cout << " Magic Number:" << testImagesInput.readInt() << endl;
		numberOfTestingSamples = testImagesInput.readInt();
		std::cout << " Number Of Samples:" << numberOfTestingSamples << endl;
		std::cout << " Width:" << testImagesInput.readInt() << endl;
		std::cout << " Height:" << testImagesInput.readInt() << endl << endl;

		std::cout << "TEST LABELS(" << testLabelsFilename << ")" << endl;
		std::cout << " Magic Number:" << testLabelsInput.readInt() << endl;
		std::cout << " Number Of Samples:" << testLabelsInput.readInt() << endl << endl;


		testImages.resize(numberOfTestingSamples);
		testLabels.resize(numberOfTestingSamples);
		for (int i = 0; i < numberOfTestingSamples; i++)
		{
			testImages[i].resize(imagePixels);
			for (int j = 0; j < imagePixels; j++){
				testImages[i][j] = (double)testImagesInput.readByte() / 255;
			}
			testLabels[i] = testLabelsInput.readByte();

		}
		endTime = clock();
		std::cout << "Reading Time: " << double(endTime - startTime) / CLOCKS_PER_SEC << "s " << endl << endl;
	}






};



#endif // MNIST_IFSTREAM_H

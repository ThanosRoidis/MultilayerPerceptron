#include <iostream>
#include <fstream>
#include <cstdlib>
#include <list>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <mutex>
#include <GL/glut.h>
#include <string>

#include <MNIST_ifstream.h>
#include <MultilayerPerceptronNetwork.h>

using namespace std;


#define WINDOW_WIDTH 500
#define WINDOW_HEIGHT 250


vector<vector<double>> originalImages;

vector<vector<double>> reconstructedImages;



int samplesTested = 0;

int percent = 0;

std::mutex mtx;

void increaseProgress(){
	mtx.lock();
	samplesTested++;
	if (samplesTested % 100 == 0){
		percent++;
		std::cout << "\b\b\b";
		std::cout << setw(2) << setfill('0') << percent << "%";
	}
	mtx.unlock();
}






void resize(int width, int height) {
	// we ignore the params and do:
	glutReshapeWindow(WINDOW_WIDTH, WINDOW_HEIGHT);
}




void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT);
	//glutSwapBuffers();
	//return;
	glBegin(GL_POINTS);

	//Draw from left to right
	for (int digit = 0; digit < 10; digit++){
		//Draw original on top
		int top;
		if (digit < 5){
			top = 28 * 4 - 1;
		}
		else{
			top = 28 * 2 - 1 - 1;
		}
		top = top - 0.5f;

		int left = (digit % 5) * 29 + 0.5f;


		//Draw original on top
		for (int i = 0; i < 28; i++){
			for (int j = 0; j < 28; j++){
				glColor3d(originalImages[digit][i * 28 + j], originalImages[digit][i * 28 + j], originalImages[digit][i * 28 + j]);
				glVertex2d(left + j, top - i);
			}
		}
		//Draw reconstructed on bottom
		for (int i = 0; i < 28; i++){
			for (int j = 0; j < 28; j++){
				glColor3d(reconstructedImages[digit][i * 28 + j], reconstructedImages[digit][i * 28 + j], reconstructedImages[digit][i * 28 + j]);
				glVertex2d(left + j, top - 28 - i);
			}
		}
	}

	glEnd();
	glutSwapBuffers();
}




void showDigits(int argc, char** argv){


	/* Standard GLUT initialization */
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);




	int windowWidth = 500 + 4 * 4;// 100 * originalImages.size();
	int windowHeight = 400 + 4;//
	glutInitWindowSize(windowWidth, windowHeight);
	glutInitWindowPosition((glutGet(GLUT_SCREEN_WIDTH) - windowWidth) / 2, (glutGet(GLUT_SCREEN_HEIGHT) - windowHeight) / 2);


	//glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	//glutInitWindowPosition((glutGet(GLUT_SCREEN_WIDTH) - WINDOW_WIDTH) / 2, (glutGet(GLUT_SCREEN_HEIGHT) - WINDOW_HEIGHT) / 2);

	glutCreateWindow("MNIST Digit");
	glutDisplayFunc(display);
	//glutReshapeFunc(resize);

	glEnable(GL_BLEND);
	glClearColor(1.0, 1.0, 1.0, 0.0); /* white background */

	glPointSize(5);




	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//4 horizontal gaps and 1 vertical
	gluOrtho2D(0, 28 * 5 - 1 + 4, -3, 4 * 28 - 1);
	//gluOrtho2D(0, 55, 0, 27);
	glMatrixMode(GL_MODELVIEW);

	glutMainLoop();
}





void printResultsAutoencoder(vector<int> &neuronsPerLayer, int trainingSamplesPerEpoch, int numberOfEpochs, double b, double trainingTime){
	std::cout << "MULTILAYER PERCEPTRON (Autoencoder)" << endl;
	std::cout << " Dimensions: " << neuronsPerLayer[0] << endl;
	std::cout << " Training samples per epoch: " << trainingSamplesPerEpoch << endl;
	std::cout << " Epochs: " << numberOfEpochs << endl;
	for (int i = 0; i < neuronsPerLayer.size(); i++){
		std::cout << "    #Neurons in Layer " << i << ": " << neuronsPerLayer[i] << endl;
	}
	std::cout << " Learning Rate: " << b << endl;
	std::cout << " Training Time: " << trainingTime << "s " << endl;
}

void printResultsClassifier(double successRate, vector<int> &neuronsPerLayer, int trainingSamplesPerEpoch, int numberOfEpochs, double b, double trainingTime, double testingTime){
	std::cout << "MULTILAYER PERCEPTRON (Classifier)" << endl;
	std::cout << " Dimensions: " << neuronsPerLayer[0] << endl;
	std::cout << " Training samples per epoch: " << trainingSamplesPerEpoch << endl;
	std::cout << " Epochs: " << numberOfEpochs << endl;
	for (int i = 0; i < neuronsPerLayer.size(); i++){
		std::cout << "    #Neurons in Layer " << i << ": " << neuronsPerLayer[i] << endl;
	}
	std::cout << " Learning Rate: " << b << endl;
	std::cout << " Success Rate: " << successRate << "%" << endl;
	std::cout << " Training Time: " << trainingTime << "s " << endl;
	std::cout << " Testing Time: " << testingTime << "s" << endl << endl;
}





void runSimpleClassifier(){
	clock_t startTime, endTime;
	srand(time(NULL));

	vector<vector<double>> trainImages, testImages;
	vector<unsigned char> trainLabels, testLabels;

	//Read data
	MNIST_ifstream::readData_MNIST_PCA(trainImages, trainLabels, testImages, testLabels,300);

	int nThreads = 8;

	vector<int> neuronsPerLayer(3);
	neuronsPerLayer[0] = trainImages[0].size();
	neuronsPerLayer[1] = 300;
	neuronsPerLayer[2] = 10;

	int trainingSamplesPerEpoch = 60000;
	int numberOfEpochs = 5;

	//learining rate
	double b = 0.1;


	//--------------------------------------------------------------------
	//---------------------------- CLASSIFIER ----------------------------
	//--------------------------------------------------------------------


	MultilayerPerceptronNetwork classifier(neuronsPerLayer, b);


	//Train Neural Network
	startTime = clock();

	classifier.trainClassification(trainImages, trainLabels, numberOfEpochs, trainingSamplesPerEpoch);

	endTime = clock();

	double trainingTime = double(endTime - startTime) / CLOCKS_PER_SEC;
	std::cout << endl << "TRAINING FINISHED!!!" << endl << endl;


	//Testing
	startTime = clock();
	int successes = classifier.getSuccesses(testImages, testLabels);
	endTime = clock();
	double testingTime = double(endTime - startTime) / CLOCKS_PER_SEC;


	//Print Results
	double successRate = 100.0 * successes / testImages.size();
	printResultsClassifier(successRate, neuronsPerLayer, trainingSamplesPerEpoch, numberOfEpochs, b, trainingTime, testingTime);



	cout << "Deallocating memory...";
	trainImages.clear();
	testImages.clear();
	trainLabels.clear();
	testLabels.clear();
	cout << "   done!" << endl;

	system("pause");
}







void runClassifier_autoencoder(string autoencoderFilename){
	clock_t startTime, endTime;
	srand(time(NULL));

	vector<vector<double>> trainImages, testImages;
	vector<unsigned char> trainLabels, testLabels;

	//Read data
	MNIST_ifstream::readData_MNIST(trainImages, trainLabels, testImages, testLabels);


	//-------------------------------------------------------------------------------
	//----------------------------Classifier (autoencoding)--------------------------
	//-------------------------------------------------------------------------------

	MultilayerPerceptronNetwork autoencoder(autoencoderFilename);


	cout << "Passing images throught the encoder...";
	//cut down autoencoder and get new reduced dimension images
	autoencoder.removeSecondHalf();

	vector<vector<double>> reducedTrainImages(trainImages.size());
	for (int i = 0; i < reducedTrainImages.size(); i++){
		autoencoder.feedForward(trainImages[i]);
		reducedTrainImages[i] = autoencoder.getOutput();
	}

	vector<vector<double>> reducedTestImages(testImages.size());
	for (int i = 0; i < reducedTestImages.size(); i++){
		autoencoder.feedForward(testImages[i]);
		reducedTestImages[i] = autoencoder.getOutput();
	}


	cout << "   done!" << endl;


	//Classifier parameters
	vector<int> classifierNeuronsPerLayer(3);

	classifierNeuronsPerLayer[0] = reducedTrainImages[0].size();
	classifierNeuronsPerLayer[1] = 200;
	classifierNeuronsPerLayer[2] = 10;

	double b_classifier = 0.1;
	int trainingSamplesPerEpoch_classifier = 60000;
	int numberOfEpochs_classifier = 30;



	MultilayerPerceptronNetwork classifier(classifierNeuronsPerLayer, b_classifier);


	//Train
	startTime = clock();
	classifier.trainClassification(reducedTrainImages, trainLabels, numberOfEpochs_classifier, trainingSamplesPerEpoch_classifier);
	endTime = clock();
	double trainingTime = double(endTime - startTime) / CLOCKS_PER_SEC;
	std::cout << endl << "TRAINING FINISHED!!!" << endl << endl;



	//Testing
	startTime = clock();
	int successes = classifier.getSuccesses(reducedTestImages, testLabels);
	endTime = clock();
	double testingTime = double(endTime - startTime) / CLOCKS_PER_SEC;


	//Print Results
	double successRate = 100.0 * successes / reducedTestImages.size();
	printResultsClassifier(successRate, classifierNeuronsPerLayer, trainingSamplesPerEpoch_classifier, numberOfEpochs_classifier, b_classifier, trainingTime, testingTime);



	cout << "Deallocating memory...";
	trainImages.clear();
	reducedTrainImages.clear();
	testImages.clear();
	reducedTestImages.clear();
	trainLabels.clear();
	testLabels.clear();
	cout << "   done!" << endl;


	system("pause");
}





void runClassifier_autoencoder_pretrain(string autoencoderFilename){
	clock_t startTime, endTime;
	srand(time(NULL));

	vector<vector<double>> trainImages, testImages;
	vector<unsigned char> trainLabels, testLabels;

	//Read data
	MNIST_ifstream::readData_MNIST(trainImages, trainLabels, testImages, testLabels);


	//-------------------------------------------------------------------------------
	//----------------------------Classifier (autoencoding)--------------------------
	//-------------------------------------------------------------------------------

	MultilayerPerceptronNetwork autoencoder(autoencoderFilename);
	cout << endl;

	//cut down autoencoder and get new reduced dimension images
	autoencoder.removeSecondHalf();

	//Classifier parameters
	vector<int> classifierNeuronsPerLayer(2);
	classifierNeuronsPerLayer[0] = autoencoder.getOutput().size();
	classifierNeuronsPerLayer[1] = 10;


	double b_classifier = 0.1;
	int trainingSamplesPerEpoch_classifier = 60000;
	int numberOfEpochs_classifier = 3;


	MultilayerPerceptronNetwork classifier(classifierNeuronsPerLayer, b_classifier);

	autoencoder.attach(classifier);

	//Train
	startTime = clock();
	cout << "(Training the whole network)" << endl << endl;
	autoencoder.trainClassification(trainImages, trainLabels, numberOfEpochs_classifier, trainingSamplesPerEpoch_classifier);
	endTime = clock();
	double trainingTime = double(endTime - startTime) / CLOCKS_PER_SEC;
	std::cout << endl << "TRAINING FINISHED!!!" << endl << endl;



	//Testing
	startTime = clock();
	int successes = autoencoder.getSuccesses(testImages, testLabels);
	endTime = clock();
	double testingTime = double(endTime - startTime) / CLOCKS_PER_SEC;


	//Print Results
	double successRate = 100.0 * successes / testImages.size();
	printResultsClassifier(successRate, autoencoder.getNeuronsPerLayer(), trainingSamplesPerEpoch_classifier, numberOfEpochs_classifier, b_classifier, trainingTime, testingTime);



	cout << "Deallocating memory...";
	trainImages.clear();
	testImages.clear();
	trainLabels.clear();
	testLabels.clear();
	cout << "   done!" << endl;


	system("pause");
}








void runAutoencoder(bool save, bool show, int argc, char** argv){
	clock_t startTime, endTime;
	srand(time(NULL));

	vector<vector<double>> trainImages, testImages;
	vector<unsigned char> trainLabels, testLabels;

	//Read data
	MNIST_ifstream::readData_MNIST(trainImages, trainLabels, testImages, testLabels);
	
	vector<int> neuronsPerLayer(5);
	neuronsPerLayer[0] = trainImages[0].size();
	neuronsPerLayer[1] = 300;
	neuronsPerLayer[2] = 100;
	neuronsPerLayer[3] = 300;;
	neuronsPerLayer[4] = trainImages[0].size();

	int trainingSamplesPerEpoch = 20000;
	int numberOfEpochs = 2;

	//learining rate
	double b = 0.1;

	//-----------------------------------------------------------------------------------------
	//----------------------------------- AutoEncoder -----------------------------------------
	//-----------------------------------------------------------------------------------------

	
	//MultilayerPerceptronNetwork autoencoder(neuronsPerLayer, b);
	MultilayerPerceptronNetwork autoencoder("autoencoder_300_100TPL_.bin");


	bool trainWithNoise = false;
	if (trainWithNoise){

		//Get noisy images
		vector<vector<double>> noiseImages;
		noiseImages.resize(trainImages.size());
		for (int i = 0; i < trainImages.size(); i++){
			noiseImages[i].resize(trainImages[i].size());
			autoencoder.feedForward(trainImages[i]);
			noiseImages[i] = autoencoder.getOutput();
		}

		//MultilayerPerceptronNetwork autoencoder("autoencoder_300_300_TPL_finetune.bin");
		autoencoder.b = b;

		startTime = clock();

		autoencoder.trainAutoencoderPerLayerWithNoise(noiseImages, trainImages, numberOfEpochs, trainingSamplesPerEpoch);

	}
	else{
		autoencoder.trainAutoencoderPerLayer(trainImages, numberOfEpochs, trainingSamplesPerEpoch);
	}



	bool finetune = false;
	if (finetune){

		int finetuneEpochs = 3;
		double finetuneLearningRate = 0.001;
		int finetuningSamplesPerEpoch = trainingSamplesPerEpoch / 2;
		cout << "Fine-tuning" << endl << endl;

		autoencoder.b = finetuneLearningRate;
		autoencoder.trainAutoencoder(trainImages, finetuneEpochs, finetuningSamplesPerEpoch);
	}


	endTime = clock();

	double trainingTime = double(endTime - startTime) / CLOCKS_PER_SEC;
	std::cout << endl << "TRAINING FINISHED!!!" << endl << endl;


	//Print results
	printResultsAutoencoder(autoencoder.getNeuronsPerLayer(), trainingSamplesPerEpoch, numberOfEpochs, b, trainingTime);
	cout << endl << endl;
	
	
	/*
	MultilayerPerceptronNetwork autoencoder("autoencoder_300_100_50_2_TPL");

	vector<int> tmp = autoencoder.getNeuronsPerLayer();

	for (int i = 0; i < tmp.size(); i++){
		cout << tmp[i] << endl;
	}*/


	//make filename and save
	if (save){

		string autoencoderFilename = "autoencoder_";
		for (int i = 1; i < neuronsPerLayer.size() / 2; i++){
			autoencoderFilename.append(std::to_string(neuronsPerLayer[i]));
			autoencoderFilename.append("_");
		}
		autoencoderFilename.append(std::to_string(neuronsPerLayer[neuronsPerLayer.size() / 2]));

		autoencoderFilename.append("_");
		autoencoderFilename.append("TPL_finetune");
		autoencoderFilename.append(".bin");

		autoencoder.saveToFile(autoencoderFilename);
	}


	system("pause");




	//Chooce the 10 images to show
	if (show){
		//Show an image for each class in the test set.
		//For each class the chosen image is the first image found in the test set that belongs to the class
		originalImages.resize(10);
		reconstructedImages.resize(10);

		for (int digit = 0; digit < 10; digit++){
			for (int j = 0; j < testImages.size(); j++){
				if (testLabels[j] == digit){
					originalImages[digit] = testImages[j];
					autoencoder.feedForward(testImages[j]);
					reconstructedImages[digit] = autoencoder.getOutput();
					break;
				}
			}
		}
	}

	//Deallocate
	cout << "Deallocating memory...";
	trainImages.clear();
	testImages.clear();
	trainLabels.clear();
	testLabels.clear();
	cout << "   done!" << endl;

	//Show images
	if (show){
		showDigits(argc, argv);
	}
}


void store2D_Representation(string autoencoderFilename){
	vector<vector<double>> trainImages, testImages;
	vector<unsigned char> trainLabels, testLabels;

	//Read data
	MNIST_ifstream::readData_MNIST(trainImages, trainLabels, testImages, testLabels);
	MultilayerPerceptronNetwork autoencoder(autoencoderFilename);

	autoencoder.removeSecondHalf();


	vector<vector<int>> digits(10);
	for (int i = 0; i < 1000; i++){
		digits[trainLabels[i]].emplace_back(i);
	}


	for (int i = 0; i < 10; i++){
		string filename = std::to_string(i);
		filename.append(".txt");

		ofstream myfile;
		myfile.open(filename, ios::out);

		for (int j = 0; j < digits[i].size(); j++){
			autoencoder.feedForward(trainImages[digits[i][j]]);
			vector<double> output = autoencoder.getOutput();
			for (int k = 0; k < output.size(); k++) {
				myfile << output[k] << " ";
			}
			myfile << endl;
		}
		myfile.close();
	}
	system("pause");
}



void main(int argc, char** argv){
	runAutoencoder(false, true, argc, argv);
	//runClassifier_autoencoder("autoencoder_300.bin");
	//runClassifier_autoencoder_pretrain("autoencoder_500_300_TPL.bin");
	//runSimpleClassifier();
	//store2D_Representation("autoencoder_300_100_50_2_TPL.bin");
}


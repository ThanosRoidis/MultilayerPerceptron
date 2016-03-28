#ifndef MULTILAYERPERCEPTRIONNETWORK_H
#define MULTILAYERPERCEPTRIONNETWORK_H

#include <vector>
#include <time.h>  
#include <algorithm>


using namespace std;

class MultilayerPerceptronNetwork
{
private:
	//fu[i][j] is the output of the j_th neuron on the i_th layer
	vector<vector<double>> fu;

	//the delta of each neuron, starting from the first hidden layer
	vector<vector<double>> delta;

	//The derivative of each layer, starting from the first hidden layer
	vector<vector<double>> derivative;

	//The synapses weights
	vector < vector < vector<double> > > w;

	//One bias neuron for each layer
	vector<vector<double>> bias;

	
	double epochError;

	int epochSuccesses;

	
public:


	double b;

	MultilayerPerceptronNetwork(vector<int> neuronsPerLayer,double learningFactor);

	MultilayerPerceptronNetwork(string filename);

	~MultilayerPerceptronNetwork();


	

	void startEpoch();

	void feedForward(vector<double> &image);

	void adjustWeightsBP(vector<double> &image, vector<double> &bitRep);


	void trainAutoencoder(vector<vector<double>> &trainImages, int numberOfEpochs, int trainingSamplesPerEpoch);

	void trainAutoencoderWithNoise(vector<vector<double>> &trainImages, vector<vector<double>> &targetImages, int numberOfEpochs, int trainingSamplesPerEpoch);


	void trainAutoencoderPerLayer(vector<vector<double>> &trainImages, int numberOfEpochs, int trainingSamplesPerEpoch);

	void trainAutoencoderPerLayerWithNoise(vector<vector<double>> &trainImages,vector<vector<double>> &targetImages, int numberOfEpochs, int trainingSamplesPerEpoch);



	void trainClassification(vector<vector<double>> &trainImages, vector<unsigned char> &trainLabels, int numberOfEpochs, int trainingSamplesPerEpoch);



	vector<double> getOutput();	

	unsigned char getClass(vector<double> &image);

	int getSuccesses(vector<vector<double>> &testImages, vector<unsigned char> &testLabels);

	double getEpochError();

	int getEpochSuccesses();

	void removeSecondHalf();

	bool attach(MultilayerPerceptronNetwork &other);


	void saveToFile(string filename);

	vector<int> getNeuronsPerLayer();


	static double f(double u){
		return 1 / (1 + exp(-u));
	}



	static vector<double> classToArray(unsigned char value){
		vector<double> ret(10, 0.1);
		ret[value] = 0.9;
		//vector<double> ret(10, 0);
		//ret[value] = 1;

		return ret;
	}


	static unsigned char arrayToClass(vector<double> &_array){
		float val;
		unsigned char minDiffPos = 0;
		double minDiffVal = 10000000;

		for (int i = 0; i < 10; i++){
			val = 0.9 - _array[i];
			if (val < 0) val = -val;
			if (val < minDiffVal)
			{
				minDiffVal = val;
				minDiffPos = i;
			}
		}

		/*
		float val;
		unsigned char minDiffPos = 0;
		double minDiffVal = -10000000;

		for (int i = 0; i < 10; i++){
			val =  _array[i];
			if (val > minDiffVal)
			{
				minDiffVal = val;
				minDiffPos = i;
			}
		}*/

		return minDiffPos;
	}




	void printStuff();

};

#endif MULTILAYERPERCEPTRIONNETWORK_H
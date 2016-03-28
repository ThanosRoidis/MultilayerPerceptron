#include "MultilayerPerceptronNetwork.h"


#include <iostream>
#include <fstream>
#include <string>
#include "ThreadPool.h"

MultilayerPerceptronNetwork::MultilayerPerceptronNetwork(vector<int> neuronsPerLayer, double learningFactor){

	b = learningFactor;
	epochError = 0;
	epochSuccesses = 0;


	//The output of each neuron
	//fu[i][j] is the output of the j_th neuron on the i_th layer
	fu.resize(neuronsPerLayer.size());
	for (int i = 0; i < fu.size(); i++){
		fu[i].resize(neuronsPerLayer[i]);
	}


	//The delta and the derivative arrays are smaller by 1 than the fu array, since the neurons on the output layer have no delta/derivative 


	//the delta of each neuron, starting from the first hidden layer
	//i.e. delta[0][0] = delta of the first neuron of the first hidden layer, 
	//delta[delta.size() - 1][0] delta of the first neuron on the output layer

	delta.resize(neuronsPerLayer.size() - 1);
	for (int i = 0; i < delta.size(); i++){
		delta[i].resize(neuronsPerLayer[i + 1]);
	}

	//The derivative of each neuron(bias included), starting from the first hidden layer
	derivative.resize(neuronsPerLayer.size() - 1);
	for (int i = 0; i < derivative.size(); i++){
		derivative[i].resize(neuronsPerLayer[i + 1]);
	}


	//The synapses weights
	//w[n][i][j] is the weight of the synapse from the i_th neuron on the n_th layer to the j_th neuron of the {n+1}_layer(next layer)

	//One less than the number of layers
	w.resize(neuronsPerLayer.size() - 1);
	//foreach layer of synapses
	for (int n = 0; n < w.size(); n++){

		w[n].resize(neuronsPerLayer[n]);
		double width = 1 / sqrt(neuronsPerLayer[n]);
		//foreach neuron in the layer, that the synapse is coming out from
		for (int i = 0; i < w[n].size(); i++){
			//do not connect to the bias of the next layer
			w[n][i].resize(neuronsPerLayer[n + 1]);
			for (int j = 0; j < w[n][i].size(); j++){
				double weight = width*((double)rand() / (RAND_MAX)) - width / 2;
				w[n][i][j] = weight;
			}
		}
	}



	//One bias neuron for each layer
	bias.resize(neuronsPerLayer.size() - 1);
	for (int n = 0; n < bias.size(); n++){
		double width = 1 / sqrt(neuronsPerLayer[n]);
		bias[n].resize(neuronsPerLayer[n + 1]);
		for (int j = 0; j < bias[n].size(); j++){
			bias[n][j] = width*((double)rand() / (RAND_MAX)) - width / 2;
		}
	}
}



MultilayerPerceptronNetwork::MultilayerPerceptronNetwork(string filename){
	std::cout << "Reading network's weights from file \"" << filename << "\"...";

	b = 0.1;
	epochError = 0;
	epochSuccesses = 0;

	//Read Weights from the file
	ifstream myfile;
	myfile.open(filename, ios::in | ios::binary);

	vector<int>::size_type vector_size;
	myfile.read((char *)&vector_size, sizeof(vector_size));
	int noLayers = vector_size;

	vector<int> neuronsPerLayer(noLayers);
	for (int i = 0; i < noLayers; i++){
		myfile.read((char *)&neuronsPerLayer[i], sizeof(neuronsPerLayer[i]));
	}




	w.resize(neuronsPerLayer.size() - 1);
	for (int n = 0; n < w.size(); n++){
		w[n].resize(neuronsPerLayer[n]);
		for (int i = 0; i < w[n].size(); i++){
			w[n][i].resize(neuronsPerLayer[n + 1]);
			for (int j = 0; j < w[n][i].size(); j++){
				//myfile >> w[n][i][j];
				myfile.read((char *)&w[n][i][j], sizeof(w[n][i][j]));
			}
		}
	}



	bias.resize(neuronsPerLayer.size() - 1);
	for (int n = 0; n < bias.size(); n++){
		bias[n].resize(neuronsPerLayer[n + 1]);
		for (int j = 0; j < bias[n].size(); j++){
			myfile.read((char *)&bias[n][j], sizeof(bias[n][j]));
		}
	}

	myfile.close();

	//Initialize the rest

	fu.resize(neuronsPerLayer.size());
	for (int i = 0; i < fu.size(); i++){
		fu[i].resize(neuronsPerLayer[i]);
	}

	delta.resize(neuronsPerLayer.size() - 1);
	for (int i = 0; i < delta.size(); i++){
		delta[i].resize(neuronsPerLayer[i + 1]);
	}

	derivative.resize(neuronsPerLayer.size() - 1);
	for (int i = 0; i < derivative.size(); i++){
		derivative[i].resize(neuronsPerLayer[i + 1]);
	}

	std::cout << "   done!" << endl;

	//printStuff();
	cout << endl;
}


MultilayerPerceptronNetwork::~MultilayerPerceptronNetwork()
{
	fu.clear();
	w.clear();
	bias.clear();
	derivative.clear();
	delta.clear();
}



void MultilayerPerceptronNetwork::saveToFile(string filename){
	std::cout << "Saving network's weights to file \"" << filename << "\"...";

	vector<int> neuronsPerLayer = getNeuronsPerLayer();

	ofstream myfile;
	myfile.open(filename, ios::out | ios::binary);


	//save layers size
	vector<int>::size_type vectorSize = neuronsPerLayer.size();
	myfile.write((char *)&vectorSize, sizeof(vectorSize));

	for (int i = 0; i < neuronsPerLayer.size(); i++){

		myfile.write((char *)&neuronsPerLayer[i], sizeof(neuronsPerLayer[i]));

	}

	//save synaptic weights
	for (int i = 0; i < w.size(); i++){
		for (int j = 0; j < w[i].size(); j++){
			for (int k = 0; k < w[i][j].size(); k++){
				myfile.write((char *)&w[i][j][k], sizeof(w[i][j][k]));
			}
		}
	}

	//save bias weights
	for (int i = 0; i < bias.size(); i++){
		for (int j = 0; j < bias[i].size(); j++){
			myfile.write((char *)&bias[i][j], sizeof(bias[i][j]));
		}
	}


	myfile.close();

	std::cout << "   done!" << endl;
}




void MultilayerPerceptronNetwork::feedForward(vector<double> &image){
	//set the input on the first layer (image as output)
	for (int i = 0; i < fu[0].size(); i++){
		fu[0][i] = image[i];
	}

	
	//for each layer, starting from the second one, calculate its neurons output and derivative 
	for (int L = 1; L < fu.size(); L++){
		std::vector< std::future<void> > results;

		//for each neuron on that layer
		for (int j = 0; j < fu[L].size(); j++){
			
			//The sum of its incoming synapses
			double u = 0;
			//for each neuron on the L-1 layer that is connected to the j_th neuron on the L layer (no bias)
			for (int i = 0; i < fu[L - 1].size(); i++){
				u += w[L - 1][i][j] * fu[L - 1][i];
			}
			//Add bias
			u += bias[L - 1][j];
			fu[L][j] = f(u);
		}

	}
}





void MultilayerPerceptronNetwork::adjustWeightsBP(vector<double> &image, vector<double> &target){

	

	//for each neuron, starting from the first hidden layer, calculate its derivative
	for (int L = 0; L < derivative.size(); L++){
		for (int j = 0; j < fu[L + 1].size(); j++){
			derivative[L][j] = fu[L + 1][j] * (1 - fu[L + 1][j]);
		}
	}


	//The delta and the derivative arrays are smaller by 1 than the fu array, since the neurons on the output layer have no delta/derivative 
	int lastDeltaLayer = delta.size() - 1;
	int lastFULayer = fu.size() - 1;

	//Calculate delta on the last layer, based on error
	double e;
	for (int i = 0; i < target.size(); i++){
		e = target[i] - fu[lastFULayer][i];
		delta[lastDeltaLayer][i] = derivative[lastDeltaLayer][i] * e;
		epochError += e*e;
	}

	//calculate deltas, from the last layer hidden layer to the first hidden layer
	for (int L = delta.size() - 2; L >= 0; L--){
		//for each neuron on that layer 
		int weightLayer = L + 1;
		for (int i = 0; i < delta[L].size(); i++){
			double sum = 0;
			for (int j = 0; j < w[weightLayer][i].size(); j++){
				//with the delta on the next Layer, L + 1, so I use the weightLayer variable
				sum += w[weightLayer][i][j] * delta[weightLayer][j];
			}
			delta[L][i] = sum * derivative[L][i];
		}
	}


	//Recalculate the synaptic weights

	//for each layer
	for (int L = 0; L < w.size(); L++){
		//for each neuron of that layer
		for (int i = 0; i < w[L].size(); i++){
			//for each outcoming synapses
			for (int j = 0; j < w[L][i].size(); j++){
				//Here the L on the delta array is actually the next layer
				w[L][i][j] += b * delta[L][j] * fu[L][i];
			}
		}
		
		//recalculate the weights from the bias
		for (int j = 0; j < bias[L].size(); j++){
			bias[L][j] += b * delta[L][j];
		}
	}
}




void MultilayerPerceptronNetwork::trainAutoencoder(vector<vector<double>> &trainImages, int numberOfEpochs, int trainingSamplesPerEpoch){
	vector<int> indexes(trainingSamplesPerEpoch);
	for (int i = 0; i < indexes.size(); i++){
		indexes[i] = i;
	}

	//For each epoch
	for (int k = 0; k < numberOfEpochs; k++){
		int epochStartTime = clock();

		startEpoch();
		//For each image
		for (int i = 0; i < trainingSamplesPerEpoch; i++){
			feedForward(trainImages[indexes[i]]);
			adjustWeightsBP(trainImages[indexes[i]], trainImages[indexes[i]]);
		}

		// Shuffle
		std::random_shuffle(indexes.begin(), indexes.end());
		int epochDuration = clock() - epochStartTime;

		printf("Epoch: %d      Error: %f       time: %fs \n", k + 1, getEpochError() / trainingSamplesPerEpoch, (double)epochDuration / CLOCKS_PER_SEC);
	}
}




void MultilayerPerceptronNetwork::trainAutoencoderWithNoise(vector<vector<double>> &trainImages, vector<vector<double>> &targetImages,int numberOfEpochs, int trainingSamplesPerEpoch){
	vector<int> indexes(trainingSamplesPerEpoch);
	for (int i = 0; i < indexes.size(); i++){
		indexes[i] = i;
	}

	//For each epoch
	for (int k = 0; k < numberOfEpochs; k++){
		int epochStartTime = clock();

		startEpoch();
		//For each image
		for (int i = 0; i < trainingSamplesPerEpoch; i++){
			feedForward(trainImages[indexes[i]]);
			adjustWeightsBP(trainImages[indexes[i]], targetImages[indexes[i]]);
		}

		// Shuffle
		std::random_shuffle(indexes.begin(), indexes.end());
		int epochDuration = clock() - epochStartTime;

		printf("Epoch: %d      Error: %f       time: %fs \n", k + 1, getEpochError() / trainingSamplesPerEpoch, (double)epochDuration / CLOCKS_PER_SEC);
	}
}






void MultilayerPerceptronNetwork::trainAutoencoderPerLayer(vector<vector<double>> &trainImages, int numberOfEpochs, int trainingSamplesPerEpoch){
	vector<int> neuronsPerLayer = getNeuronsPerLayer();

	//The images that will be used to train each subnetwork
	vector<vector<double>> trainImagesUsed(trainImages.size());
	for (int i = 0; i < trainImages.size(); i++){
		trainImagesUsed[i] = trainImages[i];
	}

	for (int i = 0; i < neuronsPerLayer.size() / 2;i++){
		vector<int> smallNetworkNeuronsPerLayer(3);
		smallNetworkNeuronsPerLayer[0] = neuronsPerLayer[i];
		smallNetworkNeuronsPerLayer[1] = neuronsPerLayer[i + 1];
		smallNetworkNeuronsPerLayer[2] = neuronsPerLayer[i];

		MultilayerPerceptronNetwork smallNetwork(smallNetworkNeuronsPerLayer, b);
		smallNetwork.w[0] = w[i];
		smallNetwork.w[1] = w[neuronsPerLayer.size() - 2 - i];
		smallNetwork.bias[0] = bias[i];
		smallNetwork.bias[1] = bias[neuronsPerLayer.size() - 2 - i];



		//smallNetwork.printStuff();


		cout << "Training " << smallNetworkNeuronsPerLayer[0] << "->" << smallNetworkNeuronsPerLayer[1] << "->" << smallNetworkNeuronsPerLayer[2]<<endl<<endl;
		smallNetwork.trainAutoencoder(trainImagesUsed, numberOfEpochs, trainingSamplesPerEpoch);
		cout << endl;
		//
		w[i] = smallNetwork.w[0];
		w[neuronsPerLayer.size() - 2 - i] = smallNetwork.w[1];
		
		bias[i] = smallNetwork.bias[0];
		bias[neuronsPerLayer.size() -2 - i] = smallNetwork.bias[1];
		

		//Recalculate the input for the next network;
		if (i != neuronsPerLayer.size() / 2 - 1){
			smallNetwork.removeSecondHalf();
			for (int j = 0; j < trainImagesUsed.size(); j++){
				smallNetwork.feedForward(trainImagesUsed[j]);
				trainImagesUsed[j] = smallNetwork.getOutput();
			}
		}

	}
}




void MultilayerPerceptronNetwork::trainAutoencoderPerLayerWithNoise(vector<vector<double>> &trainImages, vector<vector<double>> &targetImages,int numberOfEpochs, int trainingSamplesPerEpoch){
	vector<int> neuronsPerLayer = getNeuronsPerLayer();

	//The images that will be used to train each subnetwork
	vector<vector<double>> trainImagesUsed(trainImages.size());
	for (int i = 0; i < trainImages.size(); i++){
		trainImagesUsed[i] = trainImages[i];
	}

	for (int i = 0; i < neuronsPerLayer.size() / 2; i++){
		vector<int> smallNetworkNeuronsPerLayer(3);
		smallNetworkNeuronsPerLayer[0] = neuronsPerLayer[i];
		smallNetworkNeuronsPerLayer[1] = neuronsPerLayer[i + 1];
		smallNetworkNeuronsPerLayer[2] = neuronsPerLayer[i];

		MultilayerPerceptronNetwork smallNetwork(smallNetworkNeuronsPerLayer, b);
		smallNetwork.w[0] = w[i];
		smallNetwork.w[1] = w[neuronsPerLayer.size() - 2 - i];
		smallNetwork.bias[0] = bias[i];
		smallNetwork.bias[1] = bias[neuronsPerLayer.size() - 2 - i];



		//smallNetwork.printStuff();


		cout << "Training " << smallNetworkNeuronsPerLayer[0] << "->" << smallNetworkNeuronsPerLayer[1] << "->" << smallNetworkNeuronsPerLayer[2] << endl << endl;
		if (i == 0){
			smallNetwork.trainAutoencoderWithNoise(trainImagesUsed,targetImages, numberOfEpochs, trainingSamplesPerEpoch);
		}
		else{
			smallNetwork.trainAutoencoder(trainImagesUsed, numberOfEpochs, trainingSamplesPerEpoch);
		}
		cout << endl;
		//
		w[i] = smallNetwork.w[0];
		w[neuronsPerLayer.size() - 2 - i] = smallNetwork.w[1];

		bias[i] = smallNetwork.bias[0];
		bias[neuronsPerLayer.size() - 2 - i] = smallNetwork.bias[1];


		//Recalculate the input for the next network;
		if (i != neuronsPerLayer.size() / 2 - 1){
			smallNetwork.removeSecondHalf();
			for (int j = 0; j < trainImagesUsed.size(); j++){
				smallNetwork.feedForward(trainImagesUsed[j]);
				trainImagesUsed[j] = smallNetwork.getOutput();
			}
		}

	}
}






void MultilayerPerceptronNetwork::trainClassification(vector<vector<double>> &trainImages, vector<unsigned char> &trainLabels, int numberOfEpochs, int trainingSamplesPerEpoch){

	vector<int> indexes(trainingSamplesPerEpoch);
	for (int i = 0; i < indexes.size(); i++){
		indexes[i] = i;
	}

	//For each epoch
	for (int k = 0; k < numberOfEpochs; k++){
		int epochStartTime = clock();
		startEpoch();
		//For each image
		for (int i = 0; i < trainingSamplesPerEpoch; i++){
			feedForward(trainImages[indexes[i]]);
			if (arrayToClass(fu[fu.size() - 1]) == trainLabels[indexes[i]]){
				epochSuccesses++;
			}
			adjustWeightsBP(trainImages[indexes[i]], classToArray(trainLabels[indexes[i]]));
		}
		// Shuffle
		std::random_shuffle(indexes.begin(), indexes.end());
		int epochDuration = clock() - epochStartTime;

		printf("Epoch: %d      Error: %f       success: %.2f%%      time: %fs \n", k + 1, getEpochError() / trainingSamplesPerEpoch, (100.0 * getEpochSuccesses()) / trainingSamplesPerEpoch, (double)epochDuration / CLOCKS_PER_SEC);
	}
}




bool MultilayerPerceptronNetwork::attach(MultilayerPerceptronNetwork &other){
	vector<int>  oldNeuronsPerLayer = getNeuronsPerLayer();

	vector<int> otherNeuronsPerLayer = other.getNeuronsPerLayer();

	//If the left neural network does not have the same amount of neurons on its last as the first layer of the right network, return
	if (oldNeuronsPerLayer[oldNeuronsPerLayer.size() - 1] != otherNeuronsPerLayer[0]){
		return false;
	}

	//calculate the new number of neurons on each layer
	int newNoLayers = oldNeuronsPerLayer.size() + otherNeuronsPerLayer.size() - 1;

	vector<int> newNeuronsPerLayer(newNoLayers);
	for (int i = 0; i < oldNeuronsPerLayer.size(); i++){
		newNeuronsPerLayer[i] = oldNeuronsPerLayer[i];
	}

	for (int i = 1; i < otherNeuronsPerLayer.size(); i++){
		newNeuronsPerLayer[oldNeuronsPerLayer.size() - 1 + i] = otherNeuronsPerLayer[i];
	}


	//adjust weights
	w.resize(newNoLayers - 1);
	for (int i = oldNeuronsPerLayer.size() - 1; i < newNoLayers - 1; i++){
		w[i] = other.w[i - (oldNeuronsPerLayer.size() - 1)];
	}

	//adjust bias
	bias.resize(newNoLayers - 1);
	for (int i = oldNeuronsPerLayer.size() - 1; i < newNoLayers - 1; i++){
		bias[i] = other.bias[i - (oldNeuronsPerLayer.size() - 1)];
	}


	//reset all the other variables

	epochError = 0;
	epochSuccesses = 0;

	fu.resize(newNeuronsPerLayer.size());
	for (int i = 0; i < fu.size(); i++){
		fu[i].resize(newNeuronsPerLayer[i]);
	}

	delta.resize(newNeuronsPerLayer.size() - 1);
	for (int i = 0; i < delta.size(); i++){
		delta[i].resize(newNeuronsPerLayer[i + 1]);
	}

	derivative.resize(newNeuronsPerLayer.size() - 1);
	for (int i = 0; i < derivative.size(); i++){
		derivative[i].resize(newNeuronsPerLayer[i + 1]);
	}

	return true;
}






void MultilayerPerceptronNetwork::removeSecondHalf(){

	int noLayers = fu.size() / 2 + 1;
	fu.resize(noLayers);

	bias.resize(noLayers - 1);
	w.resize(noLayers - 1);
	delta.resize(noLayers - 1);
	derivative.resize(noLayers - 1);

}




int MultilayerPerceptronNetwork::getSuccesses(vector<vector<double>> &testImages, vector<unsigned char> &testLabels){
	int successes = 0;
	for (int i = 0; i < testImages.size(); i++){
		if (getClass(testImages[i]) == testLabels[i]){
			successes++;
		}
	}
	return successes;
}



vector<double> MultilayerPerceptronNetwork::getOutput(){
	return vector<double>(fu[fu.size() - 1]);
}



unsigned char MultilayerPerceptronNetwork::getClass(vector<double> &image)
{
	feedForward(image);

	float val;
	unsigned char minDiffPos = 0;
	double minDiffVal = 10000000;

	for (int i = 0; i < 10; i++){
		val = 0.9 - fu[fu.size() - 1][i];
		if (val < 0) val = -val;
		if (val < minDiffVal)
		{
			minDiffVal = val;
			minDiffPos = i;
		}
	}

	return minDiffPos;
}







void MultilayerPerceptronNetwork::startEpoch(){
	epochError = 0;
	epochSuccesses = 0;
}


double  MultilayerPerceptronNetwork::getEpochError(){
	return epochError;
}

int  MultilayerPerceptronNetwork::getEpochSuccesses(){
	return epochSuccesses;
}



vector<int> MultilayerPerceptronNetwork::getNeuronsPerLayer(){
	vector<int> neuronsPerLayer(fu.size());
	for (int i = 0; i < fu.size(); i++){
		neuronsPerLayer[i] = fu[i].size();
	}
	return neuronsPerLayer;
}




void MultilayerPerceptronNetwork::printStuff(){

	vector<int> neuronsPerLayer = getNeuronsPerLayer();
	for (int i = 0; i < neuronsPerLayer.size(); i++){
		std::cout << "    #Neurons in Layer " << i << ": " << neuronsPerLayer[i] << endl;
	}

	for (int i = 0; i < fu.size(); i++){
		std::cout << "    fu: " << i << ": " << fu[i].size() << endl;
	}
	for (int i = 0; i < delta.size(); i++){
		std::cout << "    delta " << i << ": " << delta[i].size() << endl;
	}
	for (int i = 0; i < derivative.size(); i++){
		std::cout << "    derivative " << i << ": " << derivative[i].size() << endl;
	}

	for (int i = 0; i < w.size(); i++){
		std::cout << "    w " << i << ": " << w[i].size() << endl;
	}

	for (int i = 0; i < bias.size(); i++){
		std::cout << "    bias " << i << ": " << w[i].size() << endl;
	}


}
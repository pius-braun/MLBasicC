// Neural Network in C++
// See the readme documentation for additional information
//
// Copyright (C) 2018  Pius Braun
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.


#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <Eigen/Dense>

#include "nwhelpers.h"
#include "param.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;


typedef struct MonitoringData {
	double *pTrainCost;
	int *pTrainAcc;
	double *pValiCost;
	int *pValiAcc;
	double *pTestCost;
	int *pTestAcc;
} MonitoringData;


typedef struct RawData {
	char *traindata;
	char *trainlabels;
	char *validata;
	char *valilabels;
	char *testdata;
	char *testlabels;
	int nTrainD, nTrainL, nValiD, nValiL, nTestD, nTestL;
} RawData;


// this is for test
void Snapshot(FILE* f, MatrixXd m) {
	int fi, fj;
	for (fi = 0; fi < m.rows(); fi++) {
		for (fj = 0; fj < m.cols(); fj++)
			fprintf(f, "%+.3lf ", m(fi, fj));
		fprintf(f, "\n");
	}
	fprintf(f, "\n");
}


void getBatch(char* pDat, char* pLab, MatrixXd &mBatch, MatrixXd &mLabels, 
				int* pSet, int batchsize) {
	int i,j, num;
	for (i = 0; i < batchsize; i++) {
		for (j = 0; j < mBatch.cols(); j++) {
			mBatch(i, j) = 
				((double)((unsigned char)*(pDat + ((*(pSet + i)) * mBatch.cols()) + j))) / 255.0;
		}
	}
	
	mLabels.setZero();
	
	for (i = 0; i < batchsize; i++) {
		num = *(pLab + (*(pSet+i)));
		mLabels(i, num) = 1.0;
	}
}


class NeuronBase {

	public:
	
		NeuronBase() 
		{ }

	public:
		virtual const char* WhoAmI() {
			return NEURON_NONE;
		}
	
		virtual MatrixXd fn(MatrixXd z) {
			return MatrixXd::Zero(z.rows(), z.cols());
		}
		
		virtual MatrixXd prime(MatrixXd z) {
			return MatrixXd::Zero(z.rows(), z.cols());
		}
};

class NeuronLinear : public NeuronBase {

		using NeuronBase::NeuronBase;
		
		public:
			virtual const char* WhoAmI() {
				return NEURON_LINEAR;
			}

			MatrixXd fn(MatrixXd z) {
				return z;
			}

			MatrixXd prime(MatrixXd z) {
				return MatrixXd::Ones(z.rows(), z.cols());
			}
};


class NeuronSigmoid : public NeuronBase {

		using NeuronBase::NeuronBase;
		
		public:
			virtual const char* WhoAmI() {
				return NEURON_SIGMOID;
			}

			MatrixXd fn(MatrixXd z) {
				return (1.0 / (1.0 + exp(-z.array())));
			}

			MatrixXd prime(MatrixXd z) {
				MatrixXd m1 = fn(z);
				MatrixXd m2 = 1.0 - m1.array();
				return m1.cwiseProduct(m2);
			}
};


class NeuronSoftmax : public NeuronBase {

		using NeuronBase::NeuronBase;
		
		public:
			virtual const char* WhoAmI() {
				return NEURON_SOFTMAX;
			}

			MatrixXd fn(MatrixXd z) {
				// numerically stable softmax
				if (z.cols() == 1) {
					double d = z.maxCoeff();
					z = z.array() - d;
					MatrixXd m1 = exp(z.array());
					d = m1.sum();
					m1 = m1 / d;
					return m1;
				}
				z = z.colwise() - z.rowwise().maxCoeff();
				MatrixXd m1 = exp(z.array());
				MatrixXd m2 = m1.rowwise().sum();
				MatrixXd m3 = (m1.cwiseQuotient(m2.replicate(1, z.cols())));
				return m3;
			}

			MatrixXd prime(MatrixXd z) {
				// not used here
				return NeuronBase::prime(z);
			}
};



class NetworkBase {

	public:
	
		typedef struct Layer {
				struct Layer *pNext;
				struct Layer *pPrev;
				int nSize;
				int nSizePrev;		// DIM: Rows * Cols
				MatrixXd mWeights;  //  nSizePrev * nSize
				VectorXd vBiases;	//  nSize
				MatrixXd mZ;		//  BATCH_SIZE * nSize
				MatrixXd mRes;		//  BATCH_SIZE * nSize
				VectorXd vZTest;	//  nSize
				VectorXd vResTest;	//  nSize
				MatrixXd mNablaW;	//  nSizePrev * nSize
				VectorXd vNablaB;	//  nSize
				MatrixXd mDelta;	//  BATCH_SIZE * nSize
				NeuronBase *neuron;	// Neuron Functions
		} Layer;


		// the Neural Net
		Layer *pNet;
		
		NetworkBase(Args a) {
			int i, nSizePrev;
			Layer *lInput;
			Layer *pRun;
			Layer *lTmp;
			
			nSizePrev = 0;
			for (i = 0; i <= a.nLayers; i++) {
				lTmp = new Layer();
				lTmp->pNext = NULL;
				lTmp->pPrev = NULL;
				lTmp->nSizePrev = nSizePrev;
				if (i == 0) {
					lTmp->nSize = a.nSizeIn;
					lTmp->pPrev = NULL;
					lInput = lTmp;
				}
				else {
					lTmp->nSize = *(a.pnNeurons + i - 1);
					pRun->pNext = lTmp;
					lTmp->pPrev = pRun;
				}
				nSizePrev = lTmp->nSize;
				pRun = lTmp;
			}
			
			// Memory Allocation and Random initialization for weights and biases
			i = 0;
			pRun = lInput->pNext;
			while (pRun != NULL) {
				pRun->mWeights = MatrixXd(pRun->nSizePrev, pRun->nSize);
				pRun->vBiases = VectorXd(pRun->nSize);
				pRun->mZ = MatrixXd(a.nBatchSize, pRun->nSize);
				pRun->mRes = MatrixXd(a.nBatchSize, pRun->nSize);
				pRun->mNablaW = MatrixXd(pRun->nSizePrev, pRun->nSize);
				pRun->vNablaB = VectorXd(pRun->nSize);
				pRun->mDelta = MatrixXd(a.nBatchSize, pRun->nSize);
				pRun->vZTest = VectorXd(pRun->nSize);
				pRun->vResTest = VectorXd(pRun->nSize);
				switch (*(a.pcNeurons+i)) {
					case 'A': 
						pRun->neuron = new NeuronLinear();
						break;
					case 'B': 
						pRun->neuron = new NeuronSigmoid(); 
						break;
					case 'C': 
						pRun->neuron = new NeuronSoftmax(); 
						break;
					default:
						printf("Error creating Neuron %c\n", *(a.pcNeurons+i));
						pRun->neuron = NULL;
						break;
				}
				i++;
				rndInitW(pRun->mWeights);
				rndInitB(pRun->vBiases);
				pRun = pRun->pNext;
			}
			lInput->mRes = MatrixXd(a.nBatchSize, lInput->nSize);
			
			pNet = lInput;
		}
		
		~NetworkBase() {
			lFree(pNet);
		}
		
		void SGD(RawData rd, MonitoringData md, Args a) {
			
			int i, j, nSlices, nHits;
			MatrixXd mLabels(a.nBatchSize, a.nSizeOut);
			int* pShuffle = new int[rd.nTrainD];
			
			for (j = 0; j < rd.nTrainD; j++)
				*(pShuffle + j) = j;
			for (j = 0; j < a.nEpochs; j++) {
				shuffle(pShuffle, rd.nTrainD);
				nSlices = rd.nTrainD / a.nBatchSize;
				for (i = 0; i < nSlices; i++) {
					getBatch(rd.traindata, rd.trainlabels, pNet->mRes, mLabels, 
								(pShuffle+i*a.nBatchSize), a.nBatchSize);

					update_mini_batch(mLabels, a.nBatchSize, a.dLearningRate, a.dWeightDecay, rd.nTrainD);
				}
				printf("Ep-%d:", j+1);

				monitor(a.nMonitor[0], a.nMonitor[1], rd.traindata, rd.trainlabels, rd.nTrainD, 
						md.pTrainCost+j, md.pTrainAcc+j, a.nSizeIn, a.nSizeOut, a.dWeightDecay);
				if (a.nMonitor[0])
					printf("\tCosts on training data:\t\t%.3lf\n", *(md.pTrainCost+j));
				if (a.nMonitor[1])
					printf("\tAccurracy on training data:\t%.2lf %%\n", 
							(double)*(md.pTrainAcc+j) / (double)rd.nTrainD * 100.0);
					
				monitor(a.nMonitor[2], a.nMonitor[3], rd.validata, rd.valilabels, rd.nValiD, 
						md.pValiCost+j, md.pValiAcc+j, a.nSizeIn, a.nSizeOut, a.dWeightDecay);
				if (a.nMonitor[2])
					printf("\tCosts on validation data:\t%.3lf\n", *(md.pValiCost+j));
				if (a.nMonitor[3])
					printf("\tAccurracy on validation data:\t%.2lf %%\n", 
							(double)*(md.pValiAcc+j) / (double)rd.nValiD * 100.0);

				monitor(a.nMonitor[4], a.nMonitor[5], rd.testdata, rd.testlabels, rd.nTestD, 
						md.pTestCost+j, md.pTestAcc+j, a.nSizeIn, a.nSizeOut, a.dWeightDecay);
				if (a.nMonitor[4])
					printf("\tCosts on test data:\t\t%.3lf\n", *(md.pTestCost+j));
				if (a.nMonitor[5])
					printf("\tAccurracy on test data:\t\t%.2lf %%\n", 
							(double)*(md.pTestAcc+j) / (double)rd.nTestD * 100.0);
			}
			delete pShuffle;
		}

		virtual const char* WhoAmI() {
			return COST_NULL;
		}
		
		int save(FILE* f) {
			int fi, fj, nLayer;
			Layer *pRun = pNet->pNext;
			nLayer = 1;
			while (pRun != NULL) {
				fprintf(f, "Weights\tLayer %d\n", nLayer);
				for (fi = 0; fi < pRun->mWeights.rows(); fi++) {
					for (fj = 0; fj < pRun->mWeights.cols(); fj++)
						fprintf(f, "%+19.17lf\t", pRun->mWeights(fi, fj));
					fprintf(f, "\n");
				}
				fprintf(f, "Biases\tLayer %d\n", nLayer);
				for (fi = 0; fi < pRun->vBiases.size(); fi++)
					fprintf(f, "%+19.17lf\t", pRun->vBiases(fi));
				fprintf(f, "\n\n");
				nLayer++;
				pRun = pRun->pNext;
			}
		}

	private:

		// IN: 3 times batchsize * nSizeOut
		// OUT: batchsize * nSizeOut
		virtual MatrixXd cost_derivative(MatrixXd mZ, MatrixXd mOut, MatrixXd mY) {
			return MatrixXd::Zero(mZ.rows(), mZ.cols());
		}

		// IN: 2 times batchsize * nSizeOut
		// OUT: double
		virtual double cost_function(MatrixXd mOut, MatrixXd mY) {
			return 0.0;
		}

		
		void lFree(Layer *pStart) {
			Layer *pRun = pStart;
			Layer *pTmp;
			while (pRun != NULL) {
				delete pRun->neuron;
				pTmp = pRun->pNext;
				delete pRun;
				pRun = pTmp;
			}
		}

		void rndInitW(MatrixXd &m) {
			int i, j;
			time_t tm;
			srand((unsigned int)time(&tm));
			for (i = 0; i < m.rows(); i++)
				for (j = 0; j < m.cols(); j++)
					m(i, j) = randn(0.0, 0.1) / sqrt((double)m.rows()); 
		}

		void rndInitB(VectorXd &v) {
			int i;
			time_t tm;
			srand((unsigned int)time(&tm));
			for (i = 0; i < v.size(); i++)
				v(i) = randn(0.0, 0.1) / sqrt((double)v.rows()); // cos((double)i) * 0.3; 
		}


		void feedforward_n(int batchsize) {
			int nBatch;
			Layer *pRun;
			
			// feedforward batch input
			pRun = pNet->pNext;
			while (pRun != NULL) {
				
				// Z(l)[batchsize*nSize] = Res(l-1)[batchsize*nSizePrev] MATMULT Weights(l)[nSizePrev*nSize] PLUS Biases(l)[nSize]
				pRun->mZ = pRun->pPrev->mRes * pRun->mWeights;
				for (nBatch = 0; nBatch < batchsize; nBatch++)
					pRun->mZ.row(nBatch) += pRun->vBiases;

				// Res(l)[batchsize*nSize] = neuron_fn(Z(l)[batchsize*nSize])
				pRun->mRes = pRun->neuron->fn(pRun->mZ);
				pRun = pRun->pNext;
			}
		}

		VectorXd feedforward_t(VectorXd vIn) {
			Layer *pRun;
			VectorXd vResPrev = vIn;
			
			// feedforward single input vector
			pRun = pNet->pNext;
			while (pRun != NULL) {

				// Z(l)[nSize] = Res(l-1)[nSizePrev] MATMULT Weights(l)[nSizePrev*nSize] PLUS Biases(l)[nSize]
				pRun->vZTest = pRun->mWeights.transpose() * vResPrev + pRun->vBiases;

				// Res(l)[nSize] = neuron_fn(Z(l)[nSize])
				pRun->vResTest = pRun->neuron->fn(pRun->vZTest);

				vResPrev = pRun->vResTest;
				pRun = pRun->pNext;
			}
			return vResPrev;
		}

		void backprop(MatrixXd mLabels, int batchsize) {
			Layer *pRun;
			feedforward_n(batchsize);
			
			pRun = pNet->pNext;
			while (pRun != NULL) {
				pRun->mNablaW.setZero();
				pRun->vNablaB.setZero();
				pRun = pRun->pNext;
			}
			
			// Backpropagation
			// goto output layer
			pRun = pNet->pNext;
			while (pRun->pNext != NULL)
				pRun = pRun->pNext;
			
			// Delta(lOut)[batchsize*nSize] = cost_derivative(Z(lOut)[batchsize*nSize], Res(lOut)[batchsize*nSize], labels(lOut)[batchsize*nSize])
			pRun->mDelta = cost_derivative(pRun->mZ, pRun->mRes, mLabels);
			
			// NablaB(lOut)[nSize] = ColSum(Delta(lOut)[batchsize*nSize])
			pRun->vNablaB = pRun->mDelta.colwise().sum();
			
			// NablaW(lOut)[nSizePrev*nSize] = transpose(Delta(lOut)[batchsize*nSize]) MATMULT Res(lOut-1)[batchsize*nSizePrev]
			pRun->mNablaW = pRun->pPrev->mRes.transpose() * pRun->mDelta;
			
			pRun = pRun->pPrev;
			while (pRun->pPrev != NULL) {

				// Delta(l)[batchsize*nSize] =  Delta(l+1)[batchsize*nSize(l+1)] MATMULT transpose(Weights(l+1)[nSize * nSize(l+1)])
				pRun->mDelta = pRun->pNext->mDelta * pRun->pNext->mWeights.transpose();

				// Delta(l)[batchsize*nSize] = Delta(l)[batchsize*nSize] DOTMULT neuron_prime(z(l)[batchsize*nSize)
				pRun->mDelta = pRun->mDelta.cwiseProduct(pRun->neuron->prime(pRun->mZ));
				
				// NablaB(l)[nSize] = ColSum(Delta(l)[batchsize*nSize])
				pRun->vNablaB = pRun->mDelta.colwise().sum();
				
				// NablaW(l)[nSizePrev*nSize] = transpose(Delta(l)[batchsize*nSize]) MATMULT Res(l-1)[batchsize*nSizePrev]
				pRun->mNablaW = pRun->pPrev->mRes.transpose() * pRun->mDelta;

				pRun = pRun->pPrev;
			}
		}


		void update_mini_batch(MatrixXd mLabels, int batchsize, double eta, double lambda, int nSizeTrain) {
			Layer *pRun;
			
			backprop(mLabels, batchsize);
			pRun = pNet->pNext;

			while (pRun != NULL) {
				pRun->mWeights = pRun->mWeights * (1.0 - eta*(lambda/(double)nSizeTrain))
									- (pRun->mNablaW * (eta/((double)(batchsize))));
				pRun->vBiases = pRun->vBiases - ((eta/((double)(batchsize))) * pRun->vNablaB);
				pRun = pRun->pNext;
			}
		}

		void monitor(int bCost, int bAccuracy, char* pMonD, char* pMonL, int nSize, double *pCost, int *pAccuracy,
						int nSizeIn, int nSizeOut, double dWeightDecay) {
			int i, j, jMax;
			double dTmp, dWeightSum;
			MatrixXd mTest(1, nSizeIn);
			MatrixXd mResult(1, nSizeOut);
			MatrixXd mLabel(1, nSizeOut);
			Layer *pRun;

			*pCost = 0.0;
			*pAccuracy = 0;
			if ((!bCost) && (!bAccuracy))
				return;

			for (i = 0; i < nSize; i++) {
				getBatch(pMonD, pMonL, mTest, mLabel, &i, 1);
				mResult.row(0) = feedforward_t(mTest.row(0));
				if (bAccuracy) {
					
					dTmp = 0.0;
					jMax = 0;
					for (j = 0; j < nSizeOut; j++) {
						if (mResult(0, j) > dTmp) {
							dTmp = mResult(0, j);
							jMax = j;
						}
					}
					if (jMax == *(pMonL+i))
						*pAccuracy = (*pAccuracy) + 1;
				}
				if (bCost) {
					dTmp = cost_function(mResult, mLabel);
					
					dWeightSum = 0;
					pRun = pNet->pNext;
					while (pRun != NULL) {
						dWeightSum = dWeightSum + pRun->mWeights.squaredNorm();
						pRun = pRun->pNext;
					}
					*pCost = dTmp + dWeightSum * (dWeightDecay / (2.0 * (double)nSize));
					
				}
			}
		}
		
};


class NetworkQ : public NetworkBase {

	// constructor inheritance
	using NetworkBase::NetworkBase;
	
	private:
		
		const char* WhoAmI() {
			return COST_Q;
		}
		
		// IN: 2 times batchsize * nSizeOut
		// OUT: double
		double cost_function(MatrixXd mOut, MatrixXd mY) {
			double n = (double)mOut.rows();
			if (mOut.cols() == 1)
				n = 1.0;
			return ((mOut-mY).cwiseProduct(mOut-mY)).sum() / (2.0 * n);
		}

		// IN: 3 times batchsize * nSizeOut
		// OUT: batchsize * nSizeOut
		MatrixXd cost_derivative(MatrixXd mZ, MatrixXd mOut, MatrixXd mY) {
			Layer *pRun = pNet->pNext;
			while (pRun->pNext != NULL)
				pRun = pRun->pNext;

			return (mOut - mY).cwiseProduct(pRun->neuron->prime(mZ));
		}
};


class NetworkCEB : public NetworkBase {

	// constructor inheritance
	using NetworkBase::NetworkBase;

	private:
		const char* WhoAmI() {
			return COST_CEB;
		}
	
		
		// return a*log(a) + (1-y)*log(1-a)
		// watch for a = 0 or a = 1
		double crossEntropy(double a, double y) {
			double term1, term2;
			if (a == 0.0)
				term1 = 0.0;
			else
				term1 = y * log(a);
			if (a == 1.0)
				term2 = 0.0;
			else term2 = (1.0 - y) * log(1.0 - a);
			return (term1 + term2);
		}

		// IN: 2 times batchsize * nSizeOut
		// OUT: double
		double cost_function(MatrixXd mOut, MatrixXd mY) {
			int i;
			double res = 0.0;
			for (i = 0; i < mOut.rows() * mOut.cols(); i++)
				res = res + crossEntropy(mOut(i), mY(i));
			if (mOut.cols() != 1)
				res = res / (double)mOut.rows();
			return (-res);
		}

		// IN: 3 times batchsize * nSizeOut
		// OUT: batchsize * nSizeOut
		MatrixXd cost_derivative(MatrixXd mZ, MatrixXd mOut, MatrixXd mY) {
			return (mOut - mY);
		}		
};


class NetworkCEM : public NetworkBase {

	// constructor inheritance
	using NetworkBase::NetworkBase;

	private:
		const char* WhoAmI() {
			return COST_CEM;
		}

		// IN: 2 times batchsize * nSizeOut
		// OUT: double
		double cost_function(MatrixXd mOut, MatrixXd mY) {
			double res = 0.0;
			int i;
			for (i = 0; i < mOut.rows()*mOut.cols(); i++)
 				if (mY(i) != 0.0)
					res = res + log(mOut(i));
 			if (mOut.rows() != 1)
				res = res / (double)mOut.rows();
			return (-res);
		}

		// IN: 3 times batchsize * nSizeOut
		// OUT: batchsize * nSizeOut
		MatrixXd cost_derivative(MatrixXd mZ, MatrixXd mOut, MatrixXd mY) {
			int i, j, k;
			MatrixXd mRes(mZ.rows(), mZ.cols());
			for (i = 0; i < mZ.rows(); i++) {
				for (j = 0; j < mZ.cols(); j++) {
					if (mY(i,j) != 0.0) {
						for (k = 0; k < mZ.cols(); k++) {
							if (k == j)
								mRes(i, k) = - mOut(i,k) * (1.0 - mOut(i,k));
							else
								mRes(i, k) = mOut(i,k) * mOut(i,j);
						}
					}
				}
			}
			return (mRes);
		}		
};



// -------------- Network ------------------


int main(int argc, char *argv[])
{
	clock_t t0 = clock();	

	RawData rd;
	MonitoringData md;
	Args args;
	int dummy;
	int nCols;
	
	// parse command line
	if (getopt(argc, argv, &args) != 0)
		return -1;

	
	// Read training set, test set
	rd.traindata = readidxfile(args.sTrainData, &rd.nTrainD, &nCols);
	rd.testdata = readidxfile(args.sTestData, &rd.nTestD, &dummy);
	rd.trainlabels = readidxfile(args.sTrainLabels, &rd.nTrainL, &dummy);
	rd.testlabels = readidxfile(args.sTestLabels, &rd.nTestL, &dummy);
	if ((rd.traindata == NULL) || (rd.testdata == NULL) || 
		(rd.trainlabels == NULL) || (rd.testlabels == NULL)) {
		printf("Error reading files: not enough RAM or file corrupt\n");
		return -1;
	}
	args.nSizeIn = nCols;
	args.nSizeOut = *(args.pnNeurons+args.nLayers-1);

	argsSummary(stdout, '\t', args);

	
	// TODO: implement validation set
/*	#define VALI_SIZE 0
	rd.validata = (rd.traindata + VALI_SIZE * *(pNeurons));
	rd.valilabels = (rd.trainlabels + VALI_SIZE);
	rd.nValiD = VALI_SIZE;
	rd.nValiL = VALI_SIZE;
	rd.nTrainD = rd.nTrainD - VALI_SIZE;
	rd.nTrainL = rd.nTrainL - VALI_SIZE;
*/

	NetworkBase* pNet;
	
	switch (args.cCostFunction) {
		case 'A':
			pNet = new NetworkQ(args);
			break;
		case 'B':
			pNet = new NetworkCEB(args);
			break;
		case 'C':
			pNet = new NetworkCEM(args);
			break;
		default:
			printf("Error: invalid cost function\n");
			break;
	}

	md.pTrainCost = new double[args.nEpochs];
	md.pTrainAcc = new int[args.nEpochs];
	md.pValiCost = new double[args.nEpochs];
	md.pValiAcc = new int[args.nEpochs];
	md.pTestCost = new double[args.nEpochs];
	md.pTestAcc = new int[args.nEpochs];
	
	pNet->SGD(rd, md, args);

	if (args.sSaveFilename != NULL) {
		FILE *f = fopen(args.sSaveFilename,"wb");
		if (f == NULL) {
			printf("error saving results to %s\n", args.sSaveFilename);
		} else {
			time_t tm;
			time(&tm);
			fprintf(f, "Created,%s\n", ctime(&tm));
			argsSummary(f, ',', args);
			pNet->save(f);
			fclose(f);
		}
	}
	delete pNet;
	delete md.pTrainCost;
	delete md.pTrainAcc;
	delete md.pValiCost;
	delete md.pValiAcc;
	delete md.pTestCost;
	delete md.pTestAcc;
	
	delete rd.traindata;
	delete rd.testdata;
	delete rd.trainlabels;
	delete rd.testlabels;

	printf("Duration: %d ticks, %.3f seconds\n", clock()-t0, (double)(clock()-t0) / (double)(CLOCKS_PER_SEC));
}

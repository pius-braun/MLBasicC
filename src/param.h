#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <sys/stat.h>
#include <cerrno>

// DEFAULTS
// Default cost function: Quadratic
#define DEF_COSTFUNCTION 'A'
// Batch Size
#define DEF_BATCHSIZE 20
// Epochs
#define DEF_EPOCHS 20
// Learning Rate
#define DEF_LEARNINGRATE 0.1
// Weight Decay
#define DEF_WEIGHTDECAY 3.0
// Monitor: only accuracy on Test Data
#define DEF_MONITOR "000001"

// The cost functions
#define COST_NULL "NONE"
#define COST_Q "A = Quadratic"
#define COST_CEB "B = Binary CrossEntropy"
#define COST_CEM "C = Multiclass Cross Entropy"

// The types of neurons
#define NEURON_NONE "NONE"
#define NEURON_LINEAR "A = Linear"
#define NEURON_SIGMOID "B = Sigmoid"
#define NEURON_SOFTMAX "C = Softmax"

typedef struct Args {
	int nLayers;
	int nSizeIn;
	int nSizeOut;
	int* pnNeurons;
	char* pcNeurons;
	char* sTrainData;
	char* sTrainLabels;
	char* sValiData;
	char* sValiLabels;
	char* sTestData;
	char* sTestLabels;
	char cCostFunction;
	int nBatchSize;
	int nEpochs;
	double dLearningRate;
	double dWeightDecay;
	int nMonitor[6];
	char* sSaveFilename;
} Args;

void usage() {
	printf("\n");
	printf("Usage: network -L <layers> -N <neurons> -T <training files> [options]\n");
	printf("  -L <layers>          List (integers) of the number of neurons if each layer\n");
	printf("  -N <neurons>         List (char, A,B,C) for the type of neuron in each layer\n");
	printf("  -T <training files>  Two file names (data,labels) with the training data\n");
	printf("[options]\n");
	printf("  -t <test files>      Two file names (data,labels) with the test data\n");
	printf("  -c <cost function>   The cost function, one of: (A,B,C)\n");
	printf("  -b <batch size>      Integer (default 20), batch size\n");
	printf("  -e <epochs>          Integer (default 20), number of training interations\n");
	printf("  -l <learning rate>   Floating point, (default 0.1), learning rate\n");
	printf("  -w <weight decay>    Floating point, (default 3.0), weight decay\n");
	printf("  -m <monitor output>  Flags for cost/accuracy output, default '000001'\n");
	printf("  -s <filename>        Save training results to file <filename>\n");
	printf("For more details call:\n");
	printf("    network -?\n");
}

void usage_detailed() {
	printf("\n");
	printf("Detailed usage: network -L <layers> -N <neurons> -T <training files> [options]\n");
	printf("\n");
	printf("  -L <layers>          List (integer numbers) of the number of neurons if each\n");
	printf("                       layer 1..n (except input layer 0), separated by comma.\n");
	printf("  -N <neurons>         Type of neuron in every layer 1..n, separated by comma.\n");
	printf("                       Each one of:\n");
	printf("                         A: Linear  a = z\n");
	printf("                         B: Sigmoid a = 1/(1-exp(-z))\n");
	printf("                         C: Softmax a = exp(z_i)/sum(exp(z_i)).\n");		
	printf("  -T <training files>  Two file names (data,labels) with the training data in\n");
	printf("                       IDX format, separated by comma.\n");
	printf("\n");
	printf("  [options]\n");
	printf("  -?                   This page.\n");
	printf("  -t <test files>      Two file names (data,labels) with the test data in IDX\n");
	printf("                       format, separated by comma.\n");
	printf("  -c <cost function>   The cost function, one of:\n");
	printf("                         A: (default) / Quadratic:\n");
	printf("                             cost = 1/n * sum(a-y)^2 (default)\n");
	printf("                         B: Binary Cross Entropy\n");
	printf("                             cost = 1/n * sum(y*ln(a) + (1-y)*ln(1-a))\n");
	printf("                         C: Multiclass Cross Entropy\n");
	printf("                             cost = -ln(a[y]) (requires Softmax output layer).\n");		
	printf("  -b <batch size>      Integer number (default 20), number of records in each\n");
	printf("                       training batch.\n");
	printf("  -e <epochs>          Integer (default 20), number of interations of the main\n");
	printf("                       training loop.\n");
	printf("  -l <learning rate>   Floating point number, (default 0.1), the learning rate\n");
	printf("                        in stochastic gradient descent.\n");
	printf("  -w <weight decay>    Floating point number, (default 3.0), the weight decay in\n");
	printf("                       the cost function.\n");
	printf("  -m <monitor output>  Flag for the learning quality measures to be displayed\n");
	printf("                       List of 6 times '0' (= no) or '1' (= yes).\n");
	printf("                         Pos 1:  monitor costs on training data\n");
	printf("                         Pos 2:  monitor accuracy on training data\n");
	printf("                         Pos 3:  monitor costs on validation data (unused)\n");
	printf("                         Pos 4:  monitor accuracy on validation data(unused)\n");
	printf("                         Pos 5:  monitor costs on test data (requires -t option)\n");
	printf("                         Pos 6:  monitor accuracy on test data (req. -t option)\n");
	printf("                         (Default is '000001').\n");
	printf("  -s <filename>        Save training results to file <filename>.\n");
}

#define FSTAT_NOFILE 0
#define FSTAT_EXISTS 1
#define FSTAT_ERROR 2

int fileStat(const char* fname) {
	struct stat buf;
	int status;
	status = stat(fname, &buf);
	if (status == 0) return FSTAT_EXISTS;
	if (errno == ENOENT) return FSTAT_NOFILE;
	return FSTAT_ERROR;
}

int getopt (int argc, char **argv, Args* a) {
	int c;
	int i,j;
	int nLayersL, nLayersN;
	char* pCh;
	
	a->nLayers = 0;
	a->pnNeurons = NULL;
	a->pcNeurons = NULL;
	a->sTrainData = NULL;
	a->sTrainLabels = NULL;
	a->sValiData = NULL;
	a->sValiLabels = NULL;
	a->sTestData = NULL;
	a->sTestLabels = NULL;
	a->cCostFunction = DEF_COSTFUNCTION;
	a->nBatchSize = DEF_BATCHSIZE;
	a->nEpochs = DEF_EPOCHS;
	a->dLearningRate = DEF_LEARNINGRATE;
	a->dWeightDecay = DEF_WEIGHTDECAY;
    for (i = 0; i < 6; i++)
	  a->nMonitor[i] = DEF_MONITOR[i] - '0';
	a->sSaveFilename = NULL;

  opterr = 0;
	if (argc == 1) {
		usage();
		return -1;
	}
	while ((c = getopt(argc, argv, "t:c:b:e:l:w:m:s:L:N:T:?")) != -1) {
		switch (c) {
		case 'L':
			nLayersL = 1;
			pCh = optarg;
			while (*pCh)
				if (*pCh++ == ',') nLayersL++;
			a->pnNeurons = new int[nLayersL];
			i = 0;
			do {
				pCh = strchr(optarg, ',');
				if (pCh != NULL) *pCh = 0;
				if (sscanf(optarg, "%d", (a->pnNeurons+i)) != 1) {
					printf("Error: -L option requires integer number / integer list\n");
					delete a->pnNeurons;
					return -1;
				}
				i++;
				optarg = pCh+1;
			}
			while (i < (nLayersL));
			break;
		case 'N':
			nLayersN = 1;
			pCh = optarg;
			while (*pCh)
				if (*pCh++ == ',') nLayersN++;
			a->pcNeurons = new char[nLayersN];
			i = 0;
			do {
				pCh = strchr(optarg, ',');
				*(a->pcNeurons+i) = *optarg;
				i++;
				optarg = pCh+1;
			}
			while (i < (nLayersN));
			break;
		case 'T':
			if ((pCh = strchr(optarg, ',')) == NULL) {
				printf("Error: option -T requires 2 file names: <data>,<labels>\n");
				return -1;
			}
			*pCh = 0;
			a->sTrainData = optarg;
			a->sTrainLabels = pCh+1;
			break;
		case 'c':
			a->cCostFunction = *optarg;
			break;
		case 't':
			if ((pCh = strchr(optarg, ',')) == NULL) {
				printf("Error: option -t requires 2 file names: <data>,<labels>\n");
				return -1;
			}
			*pCh = 0;
			a->sTestData = optarg;
			a->sTestLabels = pCh+1;			
			break;
		case 'b':
			if (sscanf(optarg, "%d", &a->nBatchSize) != 1) {
				printf("Error: -b option requires integer number\n");
				return -1;
			}
			break;
		case 'e':
			if (sscanf(optarg, "%d", &a->nEpochs) != 1) {
				printf("Error: -e option requires integer number\n");
				return -1;
			}
			break;
		case 'l':
			if (sscanf(optarg, "%lf", &a->dLearningRate) != 1) {
				printf("Error: -l option requires floating point number\n");
				return -1;
			}
			break;
		case 'w':
			if (sscanf(optarg, "%lf", &a->dWeightDecay) != 1) {
				printf("Error: -w option requires floating point number\n");
				return -1;
			}
			break;
		case 'm':
			if (strlen(optarg) != 6) {
				printf("Error: -m option requires string of 6 times 0 or 1\n");
				return -1;
			}
			for (i = 0; i < 6; i++) {
				a->nMonitor[i] = (*(optarg+i)) - '0';
				if ((a->nMonitor[i] > 1) || (a->nMonitor[i] < 0)) {
					printf("Error: -m option: invalid value in field %d: %d\n", i, a->nMonitor[i]);
					return -1;
				}
			}
			break;
		case 's':
			a->sSaveFilename = optarg;
			break;
		case '?':
			if (optopt == '?') {
				usage_detailed();
				return -1;
			}
			if (strchr("tcbelwmsLNT", optopt) != NULL)
				printf("Error: option -%c requires an argument.\n", optopt);
			else if (isprint (optopt))
				printf("Error: unknown option `-%c'.\n", optopt);
			else
				printf("Error: unknown option character `\\x%x'.\n", optopt);
			usage();
			return -1;
			break;
		default:
			return -1;
		}
	}
 	
	
	// Checks
	// Required options -L -N -T
	if ((a->pnNeurons == NULL) || (a->pcNeurons == NULL) || (a->sTrainData == NULL) 
		|| (a->sTrainLabels == NULL)) {
		printf("Error: -L, -N, -T options are required\n");
		usage();
		return -1;
	}
	
	// Same number of -L and -N parameters
	if (nLayersN != nLayersL) {
		printf("Error: -L and -N arguments must have the same number of elements\n");
		return -1;
	}
	else
		a->nLayers = nLayersN;

	// nonsense option?
	for (i = optind; i < argc; i++)
		printf ("Error: non-option argument %s\n", argv[i]);
	if (optind != argc) {
		usage();
		return -1;
	}

	// -N is A, B or C
	for (i = 0; i < a->nLayers; i++) {
		if ((*(a->pcNeurons+i) < 'A') || (*(a->pcNeurons+i) > 'C')) {
			printf("Error: -N option invalid parameter \"%c\"\n", *(a->pcNeurons+i));
			return -1;
		} 
	}

	// -c is A, B or C
	if (((a->cCostFunction) < 'A') || ((a->cCostFunction) > 'C')) {
		printf("Error: -c option invalid parameter \"%c\"\n", a->cCostFunction);
		return -1;
	} 
	
	// Cost function A only with Neurons A,B, Cost function B only with Neuron B
	// cost function C only with neouron C in output layer.
	if (a->cCostFunction == 'A') {
		if 	(strchr("AB", *(a->pcNeurons + a->nLayers - 1)) == NULL) {
			printf("Error: output layer neuron for Quadratic cost function 'A' must be 'A' or 'B', not '%c'\n", 
					*(a->pcNeurons + a->nLayers - 1));
			return -1;
		}
	} else if (a->cCostFunction == 'B') {
		if ((*(a->pcNeurons + a->nLayers - 1)) != 'B') {
			printf("Error: output layer neuron for Binary CE cost function 'B' must be 'B', not '%c'\n",
					*(a->pcNeurons + a->nLayers - 1));
			return -1;
		}
	} else if (a->cCostFunction == 'C') {
		if ((*(a->pcNeurons + a->nLayers - 1)) != 'C') {
			printf("Error: output layer neuron for Multiclass CE cost function 'C' must be 'C', not '%c'\n",
					*(a->pcNeurons + a->nLayers - 1));
			return -1;
		}
	}

	// Check training and test files
	i = fileStat(a->sTrainData);
	if (i != FSTAT_EXISTS) {
		printf("Error: -T option training data file not found: %s\n", a->sTrainData);
		return -1;
	}
	
	i = fileStat(a->sTrainLabels);
	if (i != FSTAT_EXISTS) {
		printf("Error: -T option training labels file not found: %s\n", a->sTrainLabels);
		return -1;
	}
	
	if (a->sTestData != NULL) {
		i = fileStat(a->sTestData);
		if (i != FSTAT_EXISTS) {
			printf("Error: -t option test data file not found: %s\n", a->sTestData);
			return -1;
		}
	}

	if (a->sTestLabels != NULL) {
		i = fileStat(a->sTestLabels);
		if (i != FSTAT_EXISTS) {
			printf("Error: -t option test labels file not found: %s\n", a->sTestLabels);
			return -1;
		}
	}
	
	// monitoring flags 5 & 6 need -t option
	if ((a->nMonitor[4] != 0) || (a->nMonitor[5] != 0)) {
		if (a->sTestData == NULL) {
			printf("Error: -m option flags 5 or 6 are set. -t option required\n");
			return -1;
		}
	}

	// check svefile / overwrite
	if (a->sSaveFilename != NULL) {
		i = fileStat(a->sSaveFilename);
		if (i == FSTAT_EXISTS) {
			printf("Save File already exists: %s\n", a->sSaveFilename);
			printf("Overwrite (Y/N) > ");
			if (toupper(getc(stdin)) != 'Y') {
				printf(" ... cancelled\n");
				return -1;
			}
			printf("\n");
		} else {
			if (i == FSTAT_ERROR) {
				printf("Error: -s option invalid filename / path: %s\n", a->sSaveFilename);
				return -1;
			}
		}
	}
	return 0;
}


void argsSummary(FILE* f, char sep, Args a) {
	int i;
	fprintf(f, "Training parameters\n");
	fprintf(f, "%cNumber of layers:  %c%d\n", sep, sep, a.nLayers+1);
	fprintf(f, "%c%cInput  layer (%3d):%c%d neurons%ctype NONE\n", sep, sep, 0, sep, a.nSizeIn, sep);
	for (i = 0; i < a.nLayers; i++) {
		switch (*(a.pcNeurons+i)) {
			case 'A':
				fprintf(f, "%c%c%s layer (%3d):%c%d neurons%ctype %s\n", sep, sep, ((i+1)==a.nLayers)?"Output":"Hidden", 
				(i+1), sep, *(a.pnNeurons+i), sep, NEURON_LINEAR);
				break;
			case 'B':
				fprintf(f, "%c%c%s layer (%3d):%c%d neurons%ctype %s\n", sep, sep, ((i+1)==a.nLayers)?"Output":"Hidden",
				(i+1), sep, *(a.pnNeurons+i), sep, NEURON_SIGMOID);
				break;
			case 'C':
				fprintf(f, "%c%c%s layer (%3d):%c%d neurons%ctype %s\n", sep, sep, ((i+1)==a.nLayers)?"Output":"Hidden",
				(i+1), sep, *(a.pnNeurons+i), sep, NEURON_SOFTMAX);
				break;
			default:
				break;
		}
	}
	switch (a.cCostFunction) {
		case 'A':
			fprintf(f, "%cCost function:                    %c%s\n", sep, sep, COST_Q);
			break;
		case 'B':
			fprintf(f, "%cCost function:                    %c%s\n", sep, sep, COST_CEB);
			break;
		case 'C':
			fprintf(f, "%cCost function:                    %c%s\n", sep, sep, COST_CEM);
			break;
		default:
			break;
	}
	fprintf(f, "%cTraining data:                    %c%s\n", sep, sep, a.sTrainData);
	fprintf(f, "%cTraining labels:                  %c%s\n", sep, sep, a.sTrainLabels);
	if (a.sTestData == NULL)
		fprintf(f, "%cTest data:                        %c%s\n", sep, sep, "NONE");
	else 
		fprintf(f, "%cTest data:                        %c%s\n", sep, sep, a.sTestData);
	if (a.sTestLabels == NULL)
		fprintf(f, "%cTest labels:                      %c%s\n", sep, sep, "NONE");
	else 
		fprintf(f, "%cTest labels:                      %c%s\n", sep, sep, a.sTestLabels);	
	fprintf(f, "%cBatchsize:                        %c%d\n", sep, sep, a.nBatchSize);
	fprintf(f, "%cEpochs:                           %c%d\n", sep, sep, a.nEpochs);
	fprintf(f, "%cLearning rate:                    %c%.3lf\n", sep, sep, a.dLearningRate);
	fprintf(f, "%cWeigth decay:                     %c%.3lf\n", sep, sep, a.dWeightDecay);
	fprintf(f, "%cMonitor costs on training data:   %c%s\n", sep, sep, ((a.nMonitor[0]==1)?"YES":"no"));
	fprintf(f, "%cMonitor accuracy on training data:%c%s\n", sep, sep, ((a.nMonitor[1]==1)?"YES":"no"));
	fprintf(f, "%cMonitor costs on test data:       %c%s\n", sep, sep, ((a.nMonitor[4]==1)?"YES":"no"));
	fprintf(f, "%cMonitor accuracy on test data:    %c%s\n", sep, sep, ((a.nMonitor[5]==1)?"YES":"no"));
	if (a.sSaveFilename == NULL)
		fprintf(f, "%cSave result to:                     %c%s\n", sep, sep, "No file");
	else 
		fprintf(f, "%cSave result to:                     %c%s\n", sep, sep, a.sSaveFilename);
}

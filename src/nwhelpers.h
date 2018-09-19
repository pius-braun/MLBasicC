#include <cstdlib>
#include <cstring>
#include <cstdlib>
#include <ctime>

// -------------- Helpers for deep learning network ------------------

#define MAGIC_DATA 2051
#define MAGIC_LABL 2049

// this function is from "phoxis.org", Licensing is Creative Commons and GNU GPL
double randn (double mu, double sigma)
{
  double U1, U2, W, mult;
  static double X1, X2;
  static int call = 0;
 
  if (call == 1)
    {
      call = !call;
      return (mu + sigma * (double) X2);
    }
 
  do
    {
      U1 = -1 + ((double) rand () / RAND_MAX) * 2;
      U2 = -1 + ((double) rand () / RAND_MAX) * 2;
      W = pow (U1, 2) + pow (U2, 2);
    }
  while (W >= 1 || W == 0);
 
  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;
 
  call = !call;
 
  return (mu + sigma * (double) X1);
}


char* readidxfile(const char* fname, int* nCount, int* nCols) {
	FILE* f;
	char buf[8];
	char* cp1;
	char* resbuf;
	int nbytes;
	unsigned __int32 magic, nitems, nrows, ncols;

	*nCount = 0;
	f = fopen(fname, "rb");
	if (f == NULL) 
	  return NULL;

  if (fread(&buf, 1, 8, f) != 8) {
		fclose(f);
		return NULL;
	}
	
	cp1 = (char*)(&magic);
	*(cp1) = buf[3]; *(cp1+1) = buf[2]; *(cp1+2) = buf[1]; *(cp1+3) = buf[0];
	cp1 = (char*)(&nitems);
	*(cp1) = buf[7]; *(cp1+1) = buf[6]; *(cp1+2) = buf[5]; *(cp1+3) = buf[4];
	
	if (!((magic == MAGIC_DATA) || (magic == MAGIC_LABL))) {
		fclose(f);
		return NULL;
	}

	if (magic == MAGIC_DATA) { // it is a data file
		if (fread(&buf, 1, 8, f) != 8) {
			fclose(f);
			return NULL;
		}
		cp1 = (char*)(&nrows);
		*(cp1) = buf[3]; *(cp1+1) = buf[2]; *(cp1+2) = buf[1]; *(cp1+3) = buf[0];
		cp1 = (char*)(&ncols);
		*(cp1) = buf[7]; *(cp1+1) = buf[6]; *(cp1+2) = buf[5]; *(cp1+3) = buf[4];
		nbytes = nitems * nrows * ncols;
	}
	else
		nbytes = nitems;

	resbuf = new char[nbytes];
	if (resbuf == NULL) {
		fclose(f);
		return NULL;
	}
	
	if ((fread(resbuf, 1, nbytes, f) != nbytes) && (!feof(f))) {
		delete resbuf;
		fclose(f);
		return NULL;
	}
	
	fclose(f);
	*nCount = nitems;
	*nCols = nrows * ncols;
	return resbuf;
}


void shuffle(int* pShuffle, int nSize) {
	time_t tm;
	srand((unsigned int)time(&tm));
    if (nSize > 1) 
    {
        int i, j, tmp;
        for (i = 0; i < nSize - 1; i++) {
          j = i + rand() / (RAND_MAX / (nSize - i) + 1);
          tmp = *(pShuffle+j);
          *(pShuffle+j) = *(pShuffle+i);
          *(pShuffle+i) = tmp;
        }
    }
}


uint64_t rdtsc(){
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}


uint64_t g_nPerfMon[10] = {0};
uint64_t g_nPerfMonDist[10] = {0};

void PMstart(int ntick) {
	g_nPerfMon[ntick] = rdtsc(); 
}

void PMstop(int ntick) {
	g_nPerfMonDist[ntick] += (rdtsc() - g_nPerfMon[ntick]); 
}

void PMlist() {
	uint64_t sum = 0;
	int i;
	for (i=0; i < 10; i++)
		sum = sum + g_nPerfMonDist[i];
	for (i=0; i < 10; i++) {
		printf("PM%d: %.1f%%\n", i, ((double)(g_nPerfMonDist[i])/(double)(sum))*100.0);
	}
}


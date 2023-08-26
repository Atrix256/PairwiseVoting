#include <stdio.h>
#include <vector>
#include <random>
#include <unordered_set>
#include <stdint.h>

#define DETERMINISTIC() true

typedef uint32_t uint32;

std::mt19937 GetRNG()
{
	#if DETERMINISTIC()
		std::mt19937 rng;
		return rng;
	#else
		std::random_device rd;
		std::mt19937 rng(rd());
		return rng;
	#endif
}

// Gauss in the Haus
// https://nrich.maths.org/2478
int SumIntegersOneToN(int N)
{
	return (N * (N + 1)) / 2;
}

// F^(-1)(x)
int Inverse_SumIntegersOneToN(int N)
{
	return (int)std::floor(0.5f + std::sqrt(2.0f * float(N) + 0.25f));
}

uint32 NumBits(uint32 x)
{
	uint32 ret = 0;
	while (x)
	{
		x /= 2;
		ret++;
	}
	return ret;
}

void DoGraphTest(uint32 numNodes, uint32 numIterations)
{
	const uint32 c_numNodesBits = NumBits(numNodes);
	const uint32 c_numNodesMask = (1 << c_numNodesBits) - 1;

	std::mt19937 rng = GetRNG();

	std::uniform_int_distribution<uint32> nodeDist1(0, numNodes);
	std::uniform_int_distribution<uint32> nodeDist2(0, numNodes - 1);

	// TODO: run for a number of iterations (voting rounds?) or until diamater reaches a limit, or until a number of total votes is reached?
	std::unordered_set<uint32> connectionsMade;
	std::vector<uint32> connectionsList;
	std::vector<size_t> connectionsListCounts; // a count at each iteration
	std::vector<double> adjacencyMatrix(numNodes * numNodes, 0.0);

	uint32 iteration = 0;
	while (iteration < numIterations)
	{
		// TODO: calculate eigenvalues of adjacency matrix. 1st eigenvalue is d. do d/2 pair selections
		// TODO: what is d for the first round?
		int d = numNodes / 2;

		for (int newLinkIndex = 0; newLinkIndex < d; ++newLinkIndex)
		{
			// Generate a new pair of nodes to link to
			// A node can't link to itself
			// A link can be made only once
			uint32 nextLink, nextLinkA, nextLinkB;
			do
			{
				// Generate 2 random, unique node indices
				nextLinkA = nodeDist1(rng);
				nextLinkB = nodeDist2(rng);
				if (nextLinkB >= nextLinkA)
					nextLinkB++;

				// Properly order the links and make the total link record
				if (nextLinkA > nextLinkB)
					std::swap(nextLinkA, nextLinkB);
				nextLink = (nextLinkA << c_numNodesBits) | nextLinkB;
			}
			while(nextLinkA == nextLinkB || connectionsMade.count(nextLink) > 0);

			// Record this new link
			connectionsMade.insert(nextLink);
			connectionsList.push_back(nextLink);
			adjacencyMatrix[nextLinkA * numNodes + nextLinkB] = 1.0f;
			adjacencyMatrix[nextLinkB * numNodes + nextLinkA] = 1.0f;
		}

		// remember the size of the list after this iteration, so we can separate the iteration lists later
		connectionsListCounts.push_back(connectionsList.size());

		// TODO: score the graph. is it by diameter? or that other metric??

		iteration++;
	}

	// TODO: write ordered list of links to disk
}

int main(int argc, char** argv)
{
	DoGraphTest(10, 1);
	return 0;
}

// Implementing this video, and inspired to add more
// https://www.youtube.com/watch?v=XSDBbCaO-kc

/*
TODO:
* need eigen library to calculate eigenvectors of connection matrix!
* also need FPE code from other blog post, to do that stuff.
*/

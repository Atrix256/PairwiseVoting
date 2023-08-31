#define _CRT_SECURE_NO_WARNINGS // for stb

#include <stdio.h>
#include <vector>
#include <random>
#include <unordered_set>
#include <stdint.h>
#include <queue>
#include <direct.h>
#include <stdarg.h>

#include "csv.h"
#include "FPE.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static const size_t	c_pageRankMaxIterations = 100;
static const float	c_pageRankConvergenceEpsilon = 0.001f;
static const float	c_pageRankDamping = 0.85f;

static const uint32 c_FPENumRounds = 4;
static const uint32 c_FPEKey = 0x1337beef;

static const int c_FPEImageSize = 256;

#define DETERMINISTIC() true

typedef uint32_t uint32;
typedef uint64_t uint64;

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

inline uint64 NodesToConnectionIndex(uint32 nodeA, uint32 nodeB)
{
	if (nodeA > nodeB)
		std::swap(nodeA, nodeB);

	return uint64(nodeA) << 32 | uint64(nodeB);
}

inline void ConnectionIndexToNodes(uint64 connectionIndex, uint32& nodeA, uint32& nodeB)
{
	nodeA = connectionIndex >> 32;
	nodeB = (uint32)connectionIndex;
}

float DotProduct(const float* A, const float* B, uint32 count)
{
	float ret = 0.0f;
	for (uint32 index = 0; index < count; ++index)
		ret += A[index] * B[index];
	return ret;
}

float Distance(const std::vector<float>& A, const std::vector<float>& B)
{
	float ret = 0.0f;
	for (size_t i = 0; i < A.size(); ++i)
	{
		float diff = B[i] - A[i];
		ret += diff * diff;
	}
	return std::sqrt(ret);
}

uint32 CalculateScoringDistance(uint32 numNodes, const std::unordered_set<uint64>& connections)
{
	// We need a node score to determine which node would win in a match or a vote.
	// We will just use the node index cause it doesn't really matter.
	auto NodeScore = [](uint32 node) { return (float)node; };

	// make a matrix that describes how node values flow from each node to others.
	// Row r describes how values flow into node r from other nodes.
	std::vector<float> M(numNodes * numNodes, 0.0f);
	for (uint64 connection : connections)
	{
		uint32 nodeA, nodeB;
		ConnectionIndexToNodes(connection, nodeA, nodeB);

		// This is where the vote or tournament play happens.
		// If A wins, B flows into A, else A flows into B.
		if (NodeScore(nodeA) > NodeScore(nodeB))
			M[nodeA * numNodes + nodeB] = 1.0f;
		else
			M[nodeB * numNodes + nodeA] = 1.0f;
	}

	// now make sure each column sums up to 1.0
	for (uint32 column = 0; column < numNodes; ++column)
	{
		float sum = 0.0f;
		for (uint32 row = 0; row < numNodes; ++row)
			sum += M[row * numNodes + column];
		if (sum != 0.0f)
		{
			for (uint32 row = 0; row < numNodes; ++row)
				M[row * numNodes + column] /= sum;
		}
	}

	// Apply a 1 - c_pageRankDamping percent chance for anyone to win, despite the vote.
	for (float& f : M)
		f = f * c_pageRankDamping + (1.0f - c_pageRankDamping) / float(numNodes);

	// put equal value in each node to start out.
	std::vector<float> V(numNodes, 1.0f / float(numNodes));

	// Now repeatedly do: V = MV.
	// Each operation is equivelent to letting the values flow from each node to the
	// nodes that beat them, equally.  The winners will get more score and give away less score.
	// We'll get a steady state in the end and be able to score the nodes.
	std::vector<float> NewV(numNodes);
	for (size_t index = 0; index < c_pageRankMaxIterations; ++index)
	{
		// NewV = M * V
		for (uint32 row = 0; row < numNodes; ++row)
			NewV[row] = DotProduct(V.data(), &M[row*numNodes], numNodes);

		// Normalize NewV to sum to 1.0 again
		float sum = 0.0f;
		for (float f : NewV)
			sum += f;
		for (float& f : NewV)
			f /= sum;

		// Exit out if we've converged enough
		float diff = Distance(V, NewV);
		if (diff <= c_pageRankConvergenceEpsilon)
			break;

		// NewV is the V for next iteration
		std::swap(V, NewV);
	}

	// Make a sorted list of the nodes, from highest score to lowest
	struct NodeInfo
	{
		uint32 nodeIndex;
		float score;
	};
	std::vector<NodeInfo> scoreBoard(numNodes);
	for (uint32 i = 0; i < numNodes; ++i)
	{
		scoreBoard[i].nodeIndex = i;
		scoreBoard[i].score = NewV[i];
	}
	std::sort(scoreBoard.begin(), scoreBoard.end(), [](const NodeInfo& A, const NodeInfo& B) {return A.score > B.score; });

	// Calculate the accuracy of the score.
	// The accuracy is the sum of the abs difference between the ranking in the scoreboard,
	// and what rank the node should be at.
	// I believe this is equivelent to optimal transport.
	// A node should be at (numNodes - nodeIndex - 1) in the scoreboard, since a node's score
	// is the same as it's node index. 0 will always lose to 1, will always lose to 2, etc.
	uint32 scoringDistance = 0;
	for (uint32 rank = 0; rank < numNodes; ++rank)
	{
		uint32 actualRank = (numNodes - scoreBoard[rank].nodeIndex - 1);
		if (rank <= actualRank)
			scoringDistance += actualRank - rank;
		else
			scoringDistance += rank - actualRank;
	}
	return scoringDistance;
}

bool AcceptCycle(std::unordered_set<uint64>& existingConnections, const std::vector<uint32>& nodeVisitOrder)
{
	// If any of the connections already exist, don't accept this cycle
	for (size_t index = 0; index < nodeVisitOrder.size(); ++index)
	{
		uint32 nodeA = nodeVisitOrder[index];
		uint32 nodeB = nodeVisitOrder[(index + 1) % nodeVisitOrder.size()];

		uint64 connection = NodesToConnectionIndex(nodeA, nodeB);

		if (existingConnections.count(connection) > 0)
			return false;
	}

	// Else, accept the cycle
	for (size_t index = 0; index < nodeVisitOrder.size(); ++index)
	{
		uint32 nodeA = nodeVisitOrder[index];
		uint32 nodeB = nodeVisitOrder[(index + 1) % nodeVisitOrder.size()];

		uint64 connection = NodesToConnectionIndex(nodeA, nodeB);

		existingConnections.insert(connection);
	}

	return true;
}

// Search through the graph to find the shortest path from nodeA to nodeB, using the connections stored in connectionsMade.
// It uses a set to make sure no paths visit nodes already accounted for by other paths.
uint32 CalculateDistance(uint32 nodeA, uint32 nodeB, uint32 numNodes, const std::unordered_set<uint64>& connectionsMade)
{
	struct PathEntry
	{
		uint32 lastNode;
		uint32 distance;
	};

	auto cmp = [](const PathEntry& A, const PathEntry& B) { return A.distance > B.distance; };
	std::priority_queue<PathEntry, std::vector<PathEntry>, decltype(cmp)> paths(cmp);

	std::unordered_set<uint32> nodesAlreadyVisited;

	paths.push({ nodeA, 0 });

	while (paths.size() > 0)
	{
		// Get the shortest path and remove it from the list
		PathEntry bestPath = paths.top();
		paths.pop();

		// remember that we've already visited this lastNode so we don't do it again and back track
		nodesAlreadyVisited.insert(bestPath.lastNode);

		// If this node connects to nodeB, we are done!
		uint64 connection = NodesToConnectionIndex(bestPath.lastNode, nodeB);
		if (connectionsMade.count(connection) > 0)
			return bestPath.distance + 1;

		// Otherwise, add all the connections from this node to check
		for (uint32 i = 0; i < numNodes; ++i)
		{
			// only consider going to a node if we haven't already visited it
			if (nodesAlreadyVisited.count(i) > 0)
				continue;

			uint64 connection = NodesToConnectionIndex(bestPath.lastNode, i);
			if (connectionsMade.count(connection) > 0)
				paths.push({ i, bestPath.distance + 1 });
		}
	}

	// This shouldn't ever happen.
	// This means there isn't a connection from nodeA to nodeB, but after
	// the first iteration, the graph should be connected.
	printf("ERROR: CalculateDistance() couldn't find a path!\n");
	return ~uint32(0);
}

uint32 CalculateRadius(uint32 numNodes, const std::unordered_set<uint64>& connectionsMade)
{
	// Note: this loop could be parallelized across threads
	uint32 radius = 0;
	for (uint32 nodeA = 0; nodeA < numNodes; ++nodeA)
	{
		for (uint32 nodeB = nodeA + 1; nodeB < numNodes; ++nodeB)
		{
			uint32 distance = CalculateDistance(nodeA, nodeB, numNodes, connectionsMade);
			radius = std::max(radius, distance);
		}
	}
	return radius;
}

bool GenerateConnections_RandomCycle(uint32 numNodes, uint32 iteration, std::unordered_set<uint64>& connectionsMade, std::mt19937& rng, std::vector<uint64>& newConnections)
{
	// Our list of nodes
	static std::vector<uint32> nodeVisitOrder;
	nodeVisitOrder.resize(numNodes);
	for (uint32 index = 0; index < numNodes; ++index)
		nodeVisitOrder[index] = index;

	// Generate a random cycle which doesn't use any connections that already exist
	uint32 attempts = 0;
	do
	{
		std::shuffle(nodeVisitOrder.begin(), nodeVisitOrder.end(), rng);
		attempts++;
	}
	while (!AcceptCycle(connectionsMade, nodeVisitOrder) && attempts < 100000);
	if (attempts == 100000)
	{
		printf("ERROR: Couldn't find a valid random cycle at iteration %u. Stopping early.\n", iteration);
		return false;
	}

	newConnections.resize(numNodes);
	for (uint32 index = 0; index < numNodes; ++index)
	{
		uint32 nodeA = nodeVisitOrder[index];
		uint32 nodeB = nodeVisitOrder[(index + 1) % numNodes];
		newConnections[index] = NodesToConnectionIndex(nodeA, nodeB);
	}

	return true;
}

template <typename TGenerateConnectionsFN>
void DoGraphTest(uint32 numNodes, uint32 numIterations, uint32 numTests, const char* fileNameBase, const TGenerateConnectionsFN& GenerateConnectionsFN, CSV& csv)
{
	printf("%s\n", fileNameBase);

	const int c_colLabel = (int)csv.columns.size();
	const int c_colVotes = c_colLabel + 1;
	const int c_colVotesStddev = c_colLabel + 2;
	const int c_colRadius = c_colLabel + 3;
	const int c_colRadiusStddev = c_colLabel + 4;
	const int c_colScoringDistance = c_colLabel + 5;
	const int c_colScoringDistanceStddev = c_colLabel + 6;

	csv.SetColumnLabel(c_colLabel, fileNameBase);
	csv.SetColumnLabel(c_colVotes, "votes");
	csv.SetColumnLabel(c_colVotesStddev, "votes stddev");
	csv.SetColumnLabel(c_colRadius, "radius");
	csv.SetColumnLabel(c_colRadiusStddev, "radius stddev");
	csv.SetColumnLabel(c_colScoringDistance, "normed scoring dist");
	csv.SetColumnLabel(c_colScoringDistanceStddev, "normed scoring dist stddev");

	std::mt19937 rng = GetRNG();

	int lastPercent = -1;
	for (uint32 testIndex = 0; testIndex < numTests; ++testIndex)
	{
		int percent = (numTests > 1) ? int(100.0f * (float(testIndex) / float(numTests - 1))) : 100;
		if (lastPercent != percent)
		{
			printf("\r%i%%", percent);
			lastPercent = percent;
		}

		// The list of connections already made.
		// Used to check if a cycle is valid, only using connections not yet made
		std::unordered_set<uint64> connectionsMade;

		// Iterate!
		std::vector<std::vector<uint64>> connectionsList(numIterations);
		std::vector<uint32> radiusList(numIterations, ~uint32(0));
		std::vector<float> scoringDistanceList(numIterations, 0.0f);
		for (uint32 iteration = 0; iteration < numIterations; ++iteration)
		{
			std::vector<uint64> newConnections;
			if (!GenerateConnectionsFN(numNodes, iteration, connectionsMade, rng, newConnections))
				break;

			// In the first test, save off each round of connections, to save out to a text file
			if (testIndex == 0)
			{
				connectionsList[iteration].resize(numNodes);
				for (size_t index = 0; index < numNodes; ++index)
					connectionsList[iteration][index] = newConnections[index];
				std::sort(connectionsList[iteration].begin(), connectionsList[iteration].end());
			}

			// The shortest path between two nodes is a distance between the nodes.
			// Considering all node pairs, the longest distance is the radius
			uint32 radius = CalculateRadius(numNodes, connectionsMade);

			// The scoring distance is the sum for all nodes of:
			// the distance from the rank it is, to the rank it should be.
			// We also normalize it by dividing by numNodes to make it comparable between graphs of different sizes
			float scoringDistance = (float)CalculateScoringDistance(numNodes, connectionsMade);
			scoringDistance /= (float)numNodes;

			if (testIndex == 0)
			{
				radiusList[iteration] = radius;
				scoringDistanceList[iteration] = scoringDistance;
			}

			// Store the data in the csv
			float votes = (float)((iteration + 1) * numNodes);
			csv.SetDataRunningAverage(c_colVotes, iteration, votes, testIndex);
			csv.SetDataRunningAverage(c_colVotesStddev, iteration, votes*votes, testIndex);

			csv.SetDataRunningAverage(c_colRadius, iteration, (float)radius, testIndex);
			csv.SetDataRunningAverage(c_colRadiusStddev, iteration, (float)radius * radius, testIndex);

			csv.SetDataRunningAverage(c_colScoringDistance, iteration, scoringDistance, testIndex);
			csv.SetDataRunningAverage(c_colScoringDistanceStddev, iteration, scoringDistance * scoringDistance, testIndex);
		}

		// Write out the details for the first test
		if (testIndex == 0)
		{
			char fileName[1024];
			sprintf_s(fileName, "%s.example.txt", fileNameBase);
			FILE* file;
			fopen_s(&file, fileName, "wb");
			if (file)
			{
				for (uint32 iteration = 0; iteration < numIterations; ++iteration)
				{
					fprintf(file, "Iteration %u, radius %u, scoring dist %f\n", iteration, radiusList[iteration], scoringDistanceList[iteration]);
					for (uint64 connection : connectionsList[iteration])
					{
						uint32 nodeA, nodeB;
						ConnectionIndexToNodes(connection, nodeA, nodeB);
						fprintf(file, "%i,%i\n", nodeA, nodeB);
					}
				}
				fclose(file);
			}
		}
	}

	// Calculate stddev of votes
	for (size_t index = 0; index < csv.columns[c_colVotesStddev].data.size(); ++index)
	{
		float avg = csv.columns[c_colVotes].data[index];
		float avgSquared = csv.columns[c_colVotesStddev].data[index];
		float variance = std::max(avgSquared - avg * avg, 0.0f);
		float stdDev = std::sqrt(variance);
		csv.columns[c_colVotesStddev].data[index] = stdDev;
	}

	// Calculate stddev of radius
	for (size_t index = 0; index < csv.columns[c_colRadiusStddev].data.size(); ++index)
	{
		float avg = csv.columns[c_colRadius].data[index];
		float avgSquared = csv.columns[c_colRadiusStddev].data[index];
		float variance = std::max(avgSquared - avg * avg, 0.0f);
		float stdDev = std::sqrt(variance);
		csv.columns[c_colRadiusStddev].data[index] = stdDev;
	}

	// Calculate stddev of scoring distance
	for (size_t index = 0; index < csv.columns[c_colScoringDistanceStddev].data.size(); ++index)
	{
		float avg = csv.columns[c_colScoringDistance].data[index];
		float avgSquared = csv.columns[c_colScoringDistanceStddev].data[index];
		float variance = std::max(avgSquared - avg * avg, 0.0f);
		float stdDev = std::sqrt(variance);
		csv.columns[c_colScoringDistanceStddev].data[index] = stdDev;
	}

	printf("\r100%%\n");
}

void DoGraphTests(uint32 numNodes, uint32 numIterations, uint32 numTests, const char* fileNameBase)
{
	CSV csv;

	char buffer[256];
	sprintf_s(buffer, "%s_RandomCycle", fileNameBase);
	DoGraphTest(numNodes, numIterations, numTests, buffer, GenerateConnections_RandomCycle, csv);

	// save the data
	csv.Save("%s.csv", fileNameBase);
}

int main(int argc, char** argv)
{
	_mkdir("out");

	// Make a white noise texture by shuffling another texture that has a flat histogram
	{
		std::vector<unsigned char> pixels(c_FPEImageSize * c_FPEImageSize);
		for (uint32 i = 0; i < c_FPEImageSize * c_FPEImageSize; ++i)
			pixels[i] = (unsigned char)(i % 256);
		stbi_write_png("out/_fpe_before.png", c_FPEImageSize, c_FPEImageSize, 1, pixels.data(), 0);

		std::vector<unsigned char> pixels2(c_FPEImageSize * c_FPEImageSize);
		for (uint32 i = 0; i < c_FPEImageSize * c_FPEImageSize; ++i)
		{
			uint32 index = FPE_Encrypt(i, c_FPEKey, c_FPEImageSize * c_FPEImageSize, c_FPENumRounds);
			pixels2[index] = pixels[i];
		}
		stbi_write_png("out/_fpe_after.png", c_FPEImageSize, c_FPEImageSize, 1, pixels2.data(), 0);
	}

	// Show FPE round trip
	{
		static const uint32 c_numItems = 8;
		for (uint32 index = 0; index < c_numItems; ++index)
		{
			uint32 encrypted = FPE_Encrypt(index, c_FPEKey, c_numItems, c_FPENumRounds);
			uint32 decrypted = FPE_Decrypt(encrypted, c_FPEKey, c_numItems, c_FPENumRounds);

			printf("%u -> %u -> %u\n", index, encrypted, decrypted);
		}
	}

	// Do the graph tests, with various sized graphs

	DoGraphTests(10, 3, 100, "out/10");
	DoGraphTests(60, 5, 100, "out/60");

	return 0;
}

// Implementing this video, and inspired to add more
// https://www.youtube.com/watch?v=XSDBbCaO-kc

/*
TODO:
! the deterministic FPE based implementation(s)
* when connections can't be generated, don't add the scores into the results.
- why do we need to normalize again in power iteration? it seems like it should converge to nonzero without.
*/

/*
Blog:
* do it at a handful of node counts and show graphs
* show the FPE images, dfts, and histograms
* show the round trip of FPE for 8 items
*/

/*
Notes:
* the video is great, and explains the motivation for the algorithm.
* the algorithm itself is real simple and you don't need eigenvalues or adjacency graphs at runtime.
* just visit every node in a random order, making sure no path has been taken before.
* each of these cycles (hamiltonian?) decreases the radius.
* could use FPE instead of actually shuffling the list.
* link to the actual implementation, there are other practical things considered, like people who don't use their vote.
* talk about page ranke
* mention the other video that just has over a million votes, so you need every pair voted on multiple times.
* node score is the node index for these examples
* imagine if you had a low discrepancy sequence that specified which connections to make, to make a well connected graph with small radius.

pagerank:
https://en.wikipedia.org/wiki/PageRank
https://www.ccs.neu.edu/home/vip/teach/IRcourse/4_webgraph/notes/Pagerank%20Explained%20Correctly%20with%20Examples.html
https://towardsdatascience.com/pagerank-algorithm-fully-explained-dc794184b4af
https://en.wikipedia.org/wiki/Power_iteration
* it's hard to justify damping in this context. it is like a lack of confidence in the votes. I guess there is unknown?
 * but it also keeps V from going to zero from having all the energy leave the nodes that have only outgoing connections
*/
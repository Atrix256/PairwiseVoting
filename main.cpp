#include <stdio.h>
#include <vector>
#include <random>
#include <unordered_set>
#include <stdint.h>
#include <queue>
#include <direct.h>
#include <stdarg.h>

#include "csv.h"

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

void DoGraphTest(uint32 numNodes, uint32 numIterations, uint32 numTests, const char* fileNameBase)
{
	printf("%s\n", fileNameBase);

	static const int c_colVotes = 0;
	static const int c_colVotesStddev = 1;
	static const int c_colRadius = 2;
	static const int c_colRadiusStddev = 3;

	CSV csv;
	csv.SetColumnLabel(c_colVotes, "votes");
	csv.SetColumnLabel(c_colVotesStddev, "votes stddev");
	csv.SetColumnLabel(c_colRadius, "radius");
	csv.SetColumnLabel(c_colRadiusStddev, "radius stddev");

	std::mt19937 rng = GetRNG();

	int lastPercent = -1;
	for (uint32 testIndex = 0; testIndex < numTests; ++testIndex)
	{
		int percent = int(100.0f * (float(testIndex) / float(numTests - 1)));
		if (lastPercent != percent)
		{
			printf("\r%i%%", percent);
			lastPercent = percent;
		}

		// The list of nodes we'll shuffle each iteration to get a new cycle.
		std::vector<uint32> nodeVisitOrder(numNodes);
		for (uint32 index = 0; index < numNodes; ++index)
			nodeVisitOrder[index] = index;

		// The list of connections already made.
		// Used to check if a cycle is valid, only using connections not yet made
		std::unordered_set<uint64> connectionsMade;

		// Iterate!
		std::vector<std::vector<uint64>> connectionsList(numIterations);
		std::vector<uint32> radiusList(numIterations, ~uint32(0));
		for (uint32 iteration = 0; iteration < numIterations; ++iteration)
		{
			// Generate a random cycle which doesn't use any connections that already exist
			uint32 attempts = 0;
			do
			{
				std::shuffle(nodeVisitOrder.begin(), nodeVisitOrder.end(), rng);
				attempts++;
			} while (!AcceptCycle(connectionsMade, nodeVisitOrder) && attempts < 100000);
			if (attempts == 100000)
			{
				printf("ERROR: Couldn't find a valid random cycle at iteration %u. Stopping early.\n", iteration);
				break;
			}

			// In the first test, save off each round of connections, to save out to a text file
			if (testIndex == 0)
			{
				connectionsList[iteration].resize(numNodes);
				for (size_t index = 0; index < numNodes; ++index)
				{
					uint32 nodeA = nodeVisitOrder[index];
					uint32 nodeB = nodeVisitOrder[(index + 1) % nodeVisitOrder.size()];

					uint64 connection = NodesToConnectionIndex(nodeA, nodeB);

					connectionsList[iteration][index] = connection;
				}
				std::sort(connectionsList[iteration].begin(), connectionsList[iteration].end());
			}

			// The shortest path between two nodes is a distance between the nodes.
			// Considering all node pairs, the longest distance is the radius
			uint32 radius = CalculateRadius(numNodes, connectionsMade);

			if (testIndex == 0)
				radiusList[iteration] = radius;

			// Store the data in the csv
			float votes = (float)((iteration + 1) * numNodes);
			csv.SetDataRunningAverage(c_colVotes, iteration, votes, testIndex);
			csv.SetDataRunningAverage(c_colVotesStddev, iteration, votes*votes, testIndex);

			csv.SetDataRunningAverage(c_colRadius, iteration, (float)radius, testIndex);
			csv.SetDataRunningAverage(c_colRadiusStddev, iteration, (float)radius * radius, testIndex);
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
					fprintf(file, "Iteration %u, radius %u\n", iteration, radiusList[iteration]);
					for (uint64 connection : connectionsList[iteration])
					{
						uint32 nodeA, nodeB;
						ConnectionIndexToNodes(connection, nodeA, nodeB);
						fprintf(file, "%i,%i\n", nodeA, nodeB);
					}
				}

				// TODO: also should show accuracy stats
				// TODO: should be tracking accuracy stats for each as well!

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

	// save the data
	csv.Save("%s.csv", fileNameBase);
	printf("\r100%%\n");
}

int main(int argc, char** argv)
{
	_mkdir("out");
	DoGraphTest(10, 3, 100, "out/10");
	DoGraphTest(60, 5, 100, "out/60");
	return 0;
}

// Implementing this video, and inspired to add more
// https://www.youtube.com/watch?v=XSDBbCaO-kc

/*
TODO:
- also need to implement page rank, and can check accuracy. could give a random score to each node that determines the voting, and is the ground truth in page rank
 * the score could also just be the node index. maybe make a node "actual score" function and return node index, but comment that it could be a random score or anything else.
 * also, comparing the page rank winner list vs the actual winner list. maybe sum of abs distance of everyone from their true location? kind of like an optimal transport
 
! the deterministic FPE based ones
*/

/*
Notes:
* the video is great, and explains the motivation for the algorithm.
* the algorithm itself is real simple and you don't need eigenvalues or adjacency graphs at runtime.
* just visit every node in a random order, making sure no path has been taken before.
* each of these cycles (hamiltonian?) decreases the radius.
* could use FPE instead of actually shuffling the list.
* link to the actual implementation, there are other practical things considered, like people who don't use their vote.
*/
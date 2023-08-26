#include <stdio.h>
#include <vector>
#include <random>
#include <unordered_set>
#include <stdint.h>
#include <queue>

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

// Not the most efficient way to find the shortest distance from nodeA to nodeB
// It's brute force without any heuristic but also doesn't keep track of what paths it has already tried!
uint32 CalculateDistance(uint32 nodeA, uint32 nodeB, uint32 numNodes, const std::unordered_set<uint64>& connectionsMade)
{
	struct PathEntry
	{
		uint32 lastNode;
		uint32 distance;
	};

	auto cmp = [](const PathEntry& A, const PathEntry& B) { return A.distance > B.distance; };
	std::priority_queue<PathEntry, std::vector<PathEntry>, decltype(cmp)> paths(cmp);

	paths.push({ nodeA, 0 });

	while (paths.size() > 0)
	{
		// Get the shortest path and remove it from the list
		PathEntry bestPath = paths.top();
		paths.pop();

		// If this node connects to nodeB, we are done!
		uint64 connection = NodesToConnectionIndex(bestPath.lastNode, nodeB);
		if (connectionsMade.count(connection) > 0)
			return bestPath.distance + 1;

		// Otherwise, add all the connections from this node to check
		for (uint32 i = 0; i < numNodes; ++i)
		{
			uint64 connection = NodesToConnectionIndex(bestPath.lastNode, i);
			if (connectionsMade.count(connection) > 0)
				paths.push({ i, bestPath.distance + 1 });
		}
	}

	// This shouldn't ever happen.
	// This means there isn't a connection from nodeA to nodeB, but after
	// the first iteration, the graph should be connected.
	return ~uint32(0);
}

uint32 CalculateRadius(uint32 numNodes, const std::unordered_set<uint64>& connectionsMade)
{
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

void DoGraphTest(uint32 numNodes, uint32 numIterations)
{
	printf("%u nodes\n", numNodes);

	std::mt19937 rng = GetRNG();

	// The list of nodes we'll shuffle each iteration to get a new cycle.
	std::vector<uint32> nodeVisitOrder(numNodes);
	for (uint32 index = 0; index < numNodes; ++index)
		nodeVisitOrder[index] = index;

	// The list of connections already made.
	// Used to check if a cycle is valid, only using connections not yet made
	std::unordered_set<uint64> connectionsMade;

	// Iterate!
	for (uint32 iteration = 0; iteration < numIterations; ++iteration)
	{
		// TODO: give up after a number of loops for random generation below?
		// can we calculate how long until it's impossible?

		// Generate a random cycle which doesn't use any connections that already exist
		do
		{
			std::shuffle(nodeVisitOrder.begin(), nodeVisitOrder.end(), rng);
		}
		while(!AcceptCycle(connectionsMade, nodeVisitOrder));

		// The shortest path between two nodes is a distance between the nodes.
		// Considering all node pairs, the longest distance is the radius
		uint32 radius = CalculateRadius(numNodes, connectionsMade);

		// TODO: store in a table, calculate mean and std dev.
		printf("    [%u] radius %u\n", iteration, radius);
	}
}

int main(int argc, char** argv)
{
	DoGraphTest(40, 5);
	return 0;
}

// Implementing this video, and inspired to add more
// https://www.youtube.com/watch?v=XSDBbCaO-kc

/*
TODO:
* profile. the first seems to take the longest. how come?
* For N nodes:
 - generate a random cycle, calculate radius
 - do this M times
 - do the whole test O times
 - show avg and stddev at each number of cycles.
 - do this for several values of N and make a graph.
 - make sure a cycle doesn't use a path taken previously.
 - also need to implement page rank, and can check accuracy. could give a random score to each node that determines the voting, and is the ground truth in page rank
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
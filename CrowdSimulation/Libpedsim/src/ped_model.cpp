//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_model.h"
#include "ped_waypoint.h"
#include "ped_model.h"
#include <iostream>
#include <stdio.h>
#include <stack>
#include <algorithm>
#include "cuda_testkernel.h"
#include <omp.h>
#include <thread>
#include <math.h>
#include <chrono>
#include <vector>

#include "cuda_agent.h"
#include <cuda_runtime.h>

// Memory leak check with msvc++
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#define new new(_NORMAL_BLOCK, __FILE__, __LINE__)
#endif

int Ped::Model::findRegion(Ped::Tagent* agent) {

	int x = agent->getX();
	int y = agent->getY();

	int xRegionVal = x / regionWidth;
	if (xRegionVal >= xRegions) { xRegionVal = xRegions - 1; };

	int yRegionVal = y / regionHeight;
	if (yRegionVal >= yRegions) { yRegionVal = yRegions - 1; };
	return (yRegionVal)*xRegions + xRegionVal;
}

void Ped::Model::splitAgents(std::vector<Ped::Tagent*> agents) {
	for (int i = 0; i < agents.size(); i++) {
		int region = findRegion(agents[i]);
		regions[region].push_back(agents[i]);
	}
}

void Ped::Model::initBorders() {
	int yRegions = floor(sqrt(nrOfRegions));
	int xRegions = ceil(nrOfRegions / yRegions);

	const int b = (xRegions * (yRegions - 1)) + (yRegions * (xRegions - 1));  //HERE

	int* array;
	array = new int[b];

	omp_lock_t* lockB;
	lockB = new omp_lock_t[b];

	for (int i = 0; i<b; i++)
		omp_init_lock(&(lockB[i]));

	// set appropriate locks for each region
	// lock indexing: top: 0, right: 1, bottom: 2, left: 3
	for (int i = 0; i < borders.size(); i++) {
		borders[i].resize(4);
		if (i >= xRegions)					{ borders[i][0] = &lockB[i - xRegions + (yRegions * (xRegions - 1))]; }
		if ((i + 1) % xRegions != 0)		{ borders[i][1] = &lockB[i - (i / xRegions)]; }
		if (i < ((xRegions - 1)*yRegions))  { borders[i][2] = &lockB[i + (yRegions * (xRegions - 1))]; }
		if (i % xRegions != 0)				{ borders[i][3] = &lockB[i - 1 - ((i - 1) / xRegions)]; }
	}
}

void Ped::Model::setup(std::vector<Ped::Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario)
{
	// Convenience test: does CUDA work on this machine?
	cuda_test();

	// Set
	agents = std::vector<Ped::Tagent*>(agentsInScenario.begin(), agentsInScenario.end());

	// Set up destinations
	destinations = std::vector<Ped::Twaypoint*>(destinationsInScenario.begin(), destinationsInScenario.end());

	// This is the sequential implementation
	//	implementation = SEQ;  CHECK, unnecessary..?

	int N = agents.size();

	setupHeatmapSeq();
//	setupHeatmapPar();


	// setup host
	agentsDesiredPosStorageValues = (int *)calloc(SIZE*SIZE, sizeof(int));
	agentsDesiredPos = (int **)malloc(SIZE* sizeof(int *));


	for (int i = 0; i < SIZE; i++){ agentsDesiredPos[i] = agentsDesiredPosStorageValues + SIZE*i; }


	// add values to desiredpos
	for (int i = 0; i < N; i++) {
		agents[i]->computeNextDesiredPosition();
		int x = agents[i]->getDesiredX();
		int y = agents[i]->getDesiredY();
		if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) {
			agentsDesiredPos[y][x] += 0;
		}
		else { agentsDesiredPos[y][x] += 1; } // OBS: reversed x/y
	}

	//NEW for blurred heatmap
	blurred_heatmapValues = (int*)malloc(SCALED_SIZE*SCALED_SIZE*sizeof(int));
	blurred_heatmap = (int **)malloc(SCALED_SIZE* sizeof(int *));

	for (int i = 0; i < SCALED_SIZE; i++) {
		blurred_heatmap[i] = blurred_heatmapValues + SCALED_SIZE*i;
		for (int j = 0; j < SCALED_SIZE; j++) {
			blurred_heatmap[i][j] = 0;
		}
	}
	d_agentsDesiredPosStorageValues = setupCUDAAgents(d_agentsDesiredPosStorageValues, agentsDesiredPosStorageValues);

	d_blurred_heatmapValues = setupCUDABlurredHM(d_blurred_heatmapValues, blurred_heatmapValues);


	switch (implementation){
	case IMPLEMENTATION::VECTOR:
		agentsX = (float*)_mm_malloc(N* sizeof(float), 16);
		agentsY = (float*)_mm_malloc(N* sizeof(float), 16);
		agentsDestX = (float*)_mm_malloc(N* sizeof(float), 16);
		agentsDestY = (float*)_mm_malloc(N* sizeof(float), 16);
		agentsDestR = (float*)_mm_malloc(N* sizeof(float), 16);
		for (int i = 0; i < N; i++) {
			agents[i]->computeNextDesiredPosition();
			agentsX[i] = agents[i]->getX();
			agentsY[i] = agents[i]->getY();

			agentsDestX[i] = agents[i]->getDestination()->getx();
			agentsDestY[i] = agents[i]->getDestination()->gety();
			agentsDestR[i] = agents[i]->getDestination()->getr();
		}
		break;
	case IMPLEMENTATION::CUDA:
		agentsX = (float*)malloc(N* sizeof(float));
		agentsY = (float*)malloc(N* sizeof(float));
		agentsDestX = (float*)malloc(N* sizeof(float));
		agentsDestY = (float*)malloc(N* sizeof(float));
		agentsDestR = (float*)malloc(N* sizeof(float));

		agentsDestReached = (float*)malloc(N* sizeof(float));

		cudaMalloc(&d_agentsX, N*sizeof(float));
		cudaMalloc(&d_agentsY, N*sizeof(float));
		cudaMalloc(&d_agentsDestX, N*sizeof(float));
		cudaMalloc(&d_agentsDestY, N*sizeof(float));
		cudaMalloc(&d_agentsDestR, N*sizeof(float));
		cudaMalloc(&d_agentsDestReached, N*sizeof(float));

		for (int i = 0; i < N; i++) {
			agents[i]->computeNextDesiredPosition();
			agentsX[i] = agents[i]->getX();
			agentsY[i] = agents[i]->getY();

			agentsDestX[i] = agents[i]->getDestination()->getx();
			agentsDestY[i] = agents[i]->getDestination()->gety();
			agentsDestR[i] = agents[i]->getDestination()->getr();
			agentsDestReached[i] = 0;
		}

		cudaMemcpy(d_agentsX, agentsX, N*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_agentsY, agentsY, N*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_agentsDestX, agentsDestX, N*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_agentsDestY, agentsDestY, N*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_agentsDestR, agentsDestR, N*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_agentsDestReached, agentsDestReached, N*sizeof(float), cudaMemcpyHostToDevice);

		break;
	case IMPLEMENTATION::OMPMOVE:
		for (int i = 0; i < N; i++) {
			agents[i]->computeNextDesiredPosition();
		}
		regions.resize(nrOfRegions);
		splitAgents(agents);
		borders.resize(nrOfRegions);
		initBorders();
		movedAgents.resize(nrOfRegions);
		break;
	default:
		break;
	}
}

void Ped::Model::change_mode(int mode){
	if (mode == 0){
		implementation = SEQ;
	}
	else if (mode == 1){
		implementation = OMP;
	}
	else if (mode == 2){
		implementation = PTHREAD;
	}
	else if (mode == 3){
		implementation = VECTOR;
	}
	else if (mode == 4){
		implementation = OMPMOVE;
	}
	else if (mode == 5){
		implementation = CUDA;
	}
}

void Ped::Model::tick() {
	switch (implementation){
	case IMPLEMENTATION::OMP:
		setAgentsPositionOMP(nthreads);
		break;
	case IMPLEMENTATION::PTHREAD:
		spawnThreads(nthreads);
		break;
	case IMPLEMENTATION::VECTOR:
		setAgentsPositionSIMD();
		break;
	case IMPLEMENTATION::CUDA:
		setAgentsPositionCUDA();
		break;
	case IMPLEMENTATION::OMPMOVE:
		setAgentsPositionOMPMove(nthreads);
		break;
	default:
		setAgentsPositionSerial();
		break;
	}

}

void Ped::Model::setAgentsPositionSerial() {
	for (int i = 0; i < agents.size(); ++i) {
		agents[i]->computeNextDesiredPosition();
		agents[i]->setX(agents[i]->getDesiredX());
		agents[i]->setY(agents[i]->getDesiredY());
		move(agents[i]);
	}
	updateHeatmapSeq();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - start);
}

void Ped::Model::setAgentsPositionOMP(int nthreads) {
	omp_set_num_threads(nrOfRegions);
#pragma omp parallel for
	for (int i = 0; i < agents.size(); ++i) {
		agents[i]->computeNextDesiredPosition();
		agents[i]->setX(agents[i]->getDesiredX());
		agents[i]->setY(agents[i]->getDesiredY());
	}
}

void Ped::Model::setAgentsPositionOMPMove(int nthreads) {

	omp_set_num_threads(nthreads);
#pragma omp parallel for
	for (int i = 0; i < agents.size(); ++i) {
		agents[i]->computeNextDesiredPosition();
	}

	// Setting the agentsDesiredPos with the number of agents desired to go to position (x,y) in the map.
	for (int i = 0; i < agents.size(); ++i) {
		int x = (int) agents[i]->getDesiredX();
		int y = agents[i]->getDesiredY();
		if (x < 0 || x >= SIZE || y < 0 || y >= SIZE){
			agentsDesiredPos[y][x] += 0;
	 }
		else { agentsDesiredPos[y][x] += 1;    // OBS: reversed x/y!
		}
	}

	// Calling Update Heatmap using CUDA
	callUpdateHeatMap(d_agentsDesiredPosStorageValues, agentsDesiredPosStorageValues, d_blurred_heatmapValues, blurred_heatmapValues);
	// Re-Initializing the agentsDesiredPos with 0's for next tick.
	for (int i = 0; i < SIZE; i++) {
		for (int j = 0; j < SIZE; j++) {
			agentsDesiredPos[i][j] = 0;
		}
	}


	// TODO: parallelfix
#pragma omp parallel for
	// move the agents, update the movedAgents vector if they change region
	for (int i = 0; i < regions.size(); ++i) {

		moveOMP(i); }

	// reallocate agents according to the movedAgents vector
	for (int i = 0; i < regions.size(); i++) {
		if (movedAgents[i].size() != 0) {
			for (int j = 0; j < movedAgents[i].size(); j++) {
				int oldRegIndex = movedAgents[i][j].first;
				int agentIndex = movedAgents[i][j].second;
				regions[i].push_back(regions[oldRegIndex][agentIndex]);
				regions[oldRegIndex][agentIndex] = NULL;
			}
		}
	}
	for (int i = 0; i < regions.size(); i++) {
		for (int j = 0; j < regions[i].size(); j++) {
			if (regions[i][j] == NULL) {
				regions[i].erase(regions[i].begin() + j);
				j--;
			}
		}
	}
	movedAgents.clear();
	movedAgents.resize(nrOfRegions);
	cudaDeviceSynchronize();
}

void Ped::Model::spawnThreads(int nthreads) {
	int numOfThreads = nthreads;
	int chunkSize = int(ceil(float(agents.size() / numOfThreads)));  // floor -> ceil
	std::thread *threads = new std::thread[nthreads];
	int from;
	int to;
	// Launch threads
	for (int i = 0; i < numOfThreads; ++i) {
		from = i * chunkSize;
		to = ((i + 1) * chunkSize);
		if (i + 1 == numOfThreads) {
			to = int(agents.size());
		}
		threads[i] = std::thread(&Model::setAgentsPositionThreads, this, from, to);
	}

	// Join threads with the main thread
	for (int j = 0; j < nthreads; j++){
		threads[j].join();
	}
}

void Ped::Model::setAgentsPositionThreads(int from, int to) {
	for (int i = from; i < to; ++i) {
		agents[i]->computeNextDesiredPosition();
		agents[i]->setX(agents[i]->getDesiredX());
		agents[i]->setY(agents[i]->getDesiredY());
	}
}


void Ped::Model::setAgentsPositionSIMD() {
	for (int i = 0; i < agents.size(); i += 4) {

		__m128 mmAgentsX, mmAgentsY, mmDestX, mmDestY, mmDestR;
		mmAgentsX = _mm_load_ps(&agentsX[i]);
		mmAgentsY = _mm_load_ps(&agentsY[i]);
		mmDestX = _mm_load_ps(&agentsDestX[i]);
		mmDestY = _mm_load_ps(&agentsDestY[i]);
		mmDestR = _mm_load_ps(&agentsDestR[i]);

		__m128 diffX = _mm_sub_ps(mmDestX, mmAgentsX);
		__m128 diffY = _mm_sub_ps(mmDestY, mmAgentsY);
		__m128 tmpX = _mm_mul_ps(diffX, diffX);
		__m128 tmpY = _mm_mul_ps(diffY, diffY);
		__m128 tmp = _mm_add_ps(tmpX, tmpY);
		__m128 len = _mm_sqrt_ps(tmp);

		__m128 agentReachedDestination = _mm_cmplt_ps(len, mmDestR);
		uint16_t test = _mm_movemask_ps(agentReachedDestination);

		if (test != 0) {
			float* fix;
			fix = (float*)_mm_malloc(4 * sizeof(float), 16);
			_mm_store_ps(fix, agentReachedDestination);
			for (int j = 0; j < 4; j++) {
				if (fix[j]) {
					Ped::Twaypoint* nextDestination = NULL;
					agents[i + j]->waypoints.push_back(agents[i + j]->destination);
					nextDestination = agents[i + j]->waypoints.front();
					agents[i + j]->waypoints.pop_front();

					if (nextDestination != NULL) {
						agentsDestX[i + j] = nextDestination->getx();
						agentsDestY[i + j] = nextDestination->gety();
						agentsDestR[i + j] = nextDestination->getr();
					}
					agents[i + j]->destination = nextDestination;
				}
			}
		}

		__m128 newDesiredX = _mm_round_ps(_mm_add_ps(mmAgentsX, _mm_div_ps(diffX, len)), 0);
		__m128 newDesiredY = _mm_round_ps(_mm_add_ps(mmAgentsY, _mm_div_ps(diffY, len)), 0);

		_mm_store_ps(&agentsX[i], newDesiredX);
		_mm_store_ps(&agentsY[i], newDesiredY);

		for (int j = 0; j < 4; j++) {
			if (i + j < agents.size()) {

				agents[i + j]->setDesiredX((agentsX[i + j]));
				agents[i + j]->setDesiredY((agentsY[i + j]));

				agents[i + j]->setX(agents[i + j]->getDesiredX());
				agents[i + j]->setY(agents[i + j]->getDesiredY());
			}
		}
	}
}

void Ped::Model::setAgentsPositionCUDA() {
	int N = agents.size();

	// update values in device
	callCUDA(d_agentsX, d_agentsY, d_agentsDestX, d_agentsDestY, d_agentsDestR, d_agentsDestReached, N);

	// copy updated values from device to host
	cudaMemcpy(agentsX, d_agentsX, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(agentsY, d_agentsY, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(agentsDestReached, d_agentsDestReached, N*sizeof(float), cudaMemcpyDeviceToHost);

	// set agents new positions, check if agents reached destination and update accordingly
	bool reached = false;
	for (int i = 0; i < N; i++) {
		agents[i]->setX(agentsX[i]);
		agents[i]->setY(agentsY[i]);

		if (agentsDestReached[i] == 1) {
			reached = true;
			agents[i]->waypoints.push_back(agents[i]->destination);
			agents[i]->destination = agents[i]->waypoints.front();
			agents[i]->waypoints.pop_front();
			if (agents[i]->destination != NULL) {
				agentsDestX[i] = agents[i]->destination->getx();
				agentsDestY[i] = agents[i]->destination->gety();
				agentsDestR[i] = agents[i]->destination->getr();
			}
			agentsDestReached[i] = 0;
		}
	}
	// copy from host to device if atleast one agent reached its destination
	if (reached) {
		cudaMemcpy(d_agentsDestX, agentsDestX, N*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_agentsDestY, agentsDestY, N*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_agentsDestR, agentsDestR, N*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_agentsDestReached, agentsDestReached, N*sizeof(float), cudaMemcpyHostToDevice);
	}

}


////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::move(Ped::Tagent *agent) {
	// Search for neighboring agents
	set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);

	// Retrieve their positions
	std::vector<std::pair<int, int> > takenPositions;
	for (std::set<const Ped::Tagent*>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt) {
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
	}

	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	std::vector<std::pair<int, int> > prioritizedAlternatives;
	std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
	prioritizedAlternatives.push_back(pDesired);

	int diffX = pDesired.first - agent->getX();
	int diffY = pDesired.second - agent->getY();
	std::pair<int, int> p1, p2;
	if (diffX == 0 || diffY == 0)
	{
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
	}
	else {
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agent->getY());
		p2 = std::make_pair(agent->getX(), pDesired.second);
	}
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);

	// Find the first empty alternative position
	for (std::vector<pair<int, int> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {

		// If the current position is not yet taken by any neighbor
		if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end()) {

			// Set the agent's position
			agent->setX((*it).first);
			agent->setY((*it).second);

			break;
		}
	}
}


void Ped::Model::moveOMP(int RegionsIndex) {
	for (int i = 0; i < regions[RegionsIndex].size(); i++) {

		Ped::Tagent *agent = regions[RegionsIndex][i];
		std::vector<int> locked;							// keep track of locked borders
		std::vector<std::pair<int, int> > takenPositions;
		std::vector<std::pair<int, int> > takenPositionsBorder;

		// check if agent is close to a border, close appropriate locks
#pragma omp critical (locks)
		{	// checks if the agent is close to its regions top border and that the region has a top border
			// if so, lock that border and add the agents neighbors in the above region to the list of neighbors
			if (((agent->getY() % regionHeight) <= 2) && borders[RegionsIndex][0] != NULL) {  // top
				omp_set_lock(borders[RegionsIndex][0]);
				takenPositionsBorder = getNeighbors(regions[RegionsIndex - xRegions], agent);
				takenPositions.insert(takenPositions.end(), takenPositionsBorder.begin(), takenPositionsBorder.end());
				locked.push_back(0);
			}
			if (agent->getX() % (regionWidth) >= (regionWidth - 2) && borders[RegionsIndex][1] != NULL) { // right
				omp_set_lock(borders[RegionsIndex][1]);
				takenPositionsBorder = getNeighbors(regions[RegionsIndex + 1], agent);
				takenPositions.insert(takenPositions.end(), takenPositionsBorder.begin(), takenPositionsBorder.end());
				locked.push_back(1);
			}

			if (((agent->getY() % (regionHeight)) >= (regionHeight - 2)) && borders[RegionsIndex][2] != NULL) { // bottom
				omp_set_lock(borders[RegionsIndex][2]);
				takenPositionsBorder = getNeighbors(regions[RegionsIndex + xRegions], agent);
				takenPositions.insert(takenPositions.end(), takenPositionsBorder.begin(), takenPositionsBorder.end());
				locked.push_back(2);
			}
			if ((agent->getX() % regionWidth <= 2) && borders[RegionsIndex][3] != NULL) { // left
				omp_set_lock(borders[RegionsIndex][3]);
				takenPositionsBorder = getNeighbors(regions[RegionsIndex - 1], agent);
				takenPositions.insert(takenPositions.end(), takenPositionsBorder.begin(), takenPositionsBorder.end());
				locked.push_back(3);
			}
			if ((agent->getX() % regionWidth <= 2) && ((agent->getY() % (regionHeight)) >= (regionHeight - 2))) {
			}

			// lock corners, incase an agent is close to a corner of its current region
			if (locked.size() >= 2) {
				if (std::find(locked.begin(), locked.end(), 0) != locked.end()) {

					// if the agent is in the top right corner,
					// lock the right border of the region above the current one
					// and add the agents neighbors in the region above and to the top to the list of neighbors
					if (std::find(locked.begin(), locked.end(), 1) != locked.end()) { //{ lock(above right) }
						takenPositionsBorder = getNeighbors(regions[RegionsIndex - xRegions + 1], agent);
						takenPositions.insert(takenPositions.end(), takenPositionsBorder.begin(), takenPositionsBorder.end());
						omp_set_lock(borders[RegionsIndex - xRegions][1]);
						locked.push_back(4);
					}

					if (std::find(locked.begin(), locked.end(), 3) != locked.end()) {  //{ lock(above left) }
						takenPositionsBorder = getNeighbors(regions[RegionsIndex - xRegions - 1], agent);
						takenPositions.insert(takenPositions.end(), takenPositionsBorder.begin(), takenPositionsBorder.end());
						omp_set_lock(borders[RegionsIndex - xRegions][3]);
						locked.push_back(5);
					}
				}
				if (std::find(locked.begin(), locked.end(), 2) != locked.end()) {
					if (std::find(locked.begin(), locked.end(), 1) != locked.end()) {  	//{ lock(below right) }
						takenPositionsBorder = getNeighbors(regions[RegionsIndex + xRegions + 1], agent);
						takenPositions.insert(takenPositions.end(), takenPositionsBorder.begin(), takenPositionsBorder.end());
						omp_set_lock(borders[RegionsIndex + xRegions][1]);
						locked.push_back(6);
					}

					if (std::find(locked.begin(), locked.end(), 3) != locked.end()) {  //{ lock(below left) }
						takenPositionsBorder = getNeighbors(regions[RegionsIndex + xRegions - 1], agent);
						takenPositions.insert(takenPositions.end(), takenPositionsBorder.begin(), takenPositionsBorder.end());
						omp_set_lock(borders[RegionsIndex + xRegions][3]);
						locked.push_back(7);
					}
				}
			}
		}

		// find neighbors in current region
		takenPositionsBorder = getNeighbors(regions[RegionsIndex], agent);
		takenPositions.insert(takenPositions.end(), takenPositionsBorder.begin(), takenPositionsBorder.end());

		// Compute the  alternative positions that would bring the agent
		// closer to his desiredPosition, starting with the desiredPosition itself
		std::vector<std::pair<int, int> > prioritizedAlternatives;
		std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
		prioritizedAlternatives.push_back(pDesired);

		int diffX = pDesired.first - agent->getX();
		int diffY = pDesired.second - agent->getY();
		std::pair<int, int> p1, p2, p3;

		if (diffX == 0 || diffY == 0)
		{
			// Agent wants to walk straight to North, South, West or East
			p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
			p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
		}
		else {
			// Agent wants to walk diagonally
			p1 = std::make_pair(pDesired.first, agent->getY());
			p2 = std::make_pair(agent->getX(), pDesired.second);
		}
		// move the agent backwards if no other option
		p3 = std::make_pair(agent->getX() - diffX, agent->getY() - diffY);

		prioritizedAlternatives.push_back(p1);
		prioritizedAlternatives.push_back(p2);
		prioritizedAlternatives.push_back(p3);

		// Find the first empty alternative position
		for (std::vector<pair<int, int> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {

			// If the current position is not yet taken by any neighbor
			if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end()) {

				// Set the agent's position
				agent->setX((*it).first);
				agent->setY((*it).second);

				// if agent changed region, assign it to be assigned a new region later
				int newRegion = findRegion(agent);
				if (newRegion != RegionsIndex) {
#pragma omp critical (push_back)
					{movedAgents[newRegion].push_back(pair <int, int>(RegionsIndex, i)); }
				}
				break;
			}
		}
		// unset locks
		for (int j = 0; j < locked.size(); j++) {
			if (locked[j] <= 3) { omp_unset_lock(borders[RegionsIndex][locked[j]]); }
			else {
				if (locked[j] == 4) { omp_unset_lock(borders[RegionsIndex - xRegions][1]); }
				if (locked[j] == 5) { omp_unset_lock(borders[RegionsIndex - xRegions][3]); }
				if (locked[j] == 6) { omp_unset_lock(borders[RegionsIndex + xRegions][1]); }
				if (locked[j] == 7) { omp_unset_lock(borders[RegionsIndex + xRegions][3]); }
			}
		}
	}
}

/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// \date    2012-01-29
/// \return  The list of neighbors
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents (search field is a square in the current implementation)
set<const Ped::Tagent*> Ped::Model::getNeighbors(int x, int y, int dist) const {

	// create the output list
	// ( It would be better to include only the agents close by, but this programmer is lazy.)
	return set<const Ped::Tagent*>(agents.begin(), agents.end());
}

// get agents in the given region within 2 pixels of the given agents position
std::vector<std::pair<int, int> > Ped::Model::getNeighbors(std::vector<Ped::Tagent*>  localAgents, Ped::Tagent* agent){
	std::vector<std::pair<int, int> > takenPositions;
	for (int k = 0; k < localAgents.size(); k++) {
		int x = localAgents[k]->getX();
		int y = localAgents[k]->getY();
		if (abs(agent->getX() - x) <= 2 && abs(agent->getY() - y) <= 2) {
			std::pair<int, int> position(x, y);
			takenPositions.push_back(position);
		}  //HERE
	}
	return takenPositions;
}


void Ped::Model::cleanup() {
	for (int i = 0; i < borders.size(); i++) {
		for (int j = 0; j < 4; j++) {
			omp_destroy_lock(borders[i][j]);
		}
	}

	// Nothing to do here right now.
}

Ped::Model::~Model()
{
	std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent){delete agent; });
	std::for_each(destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination){delete destination; });
}

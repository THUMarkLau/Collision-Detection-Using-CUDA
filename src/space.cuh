#ifndef _SPACE_H_
#define _SPACE_H_
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <unordered_map>
#include <unordered_set>
#include "ball.cuh"
#include "params.h"

class SubSpace {
	// 子空间类
public:
	SubSpace() {}
	int* balls;
	int ball_num;
};


class Space {
	// 空间类
	
public:
	Space( ) {}
	SubSpace subspaces[SUBSPACE_X][SUBSPACE_Y][SUBSPACE_Z];
};

__global__ void collisionDetectionAndUpdate(Space* space, Ball* balls);

__device__ void update(Space* space, Ball* ball);

__device__ void collisionDetection(SubSpace* subspace, Ball* balls);

__device__ void checkBound(Ball* ball);

__device__ float getDistance(Ball* first_ball, Ball* second_ball);

__device__ int binarySearch(int val, int length,  int* list);

__device__ void insert(int val, int length, int* list);

__device__ void remove(int val, int length, int* list);

__device__ int sign(float x);
#endif // !_SPACE_H_
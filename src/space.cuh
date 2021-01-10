#ifndef _SPACE_H_
#define _SPACE_H_
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <unordered_map>
#include <unordered_set>
#include "ball.cuh"
#include "params.h"
#include <math.h>

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

__global__ void collisionDetectionAndUpdate(Ball* balls, int B, int T, int ballNums, float delta_time);

__device__ void update(Ball* ball, float delta_time);

__device__ void collisionDetection(Ball* balls);

__device__ void checkBound(Ball* ball);

__device__ float getDistance(Ball* first_ball, Ball* second_ball);

__device__ int binarySearch(int val, int length,  int* list);

__device__ void insert(int val, int length, int* list);

__device__ void remove(int val, int length, int* list);

__device__ int sign(float x);

int findCollisionRecords(SpaceRecord* space_records, int space_records_length, CollisionRecord* collisionRecords);

__global__ void doCollision(int round, Ball* balls, int ball_num, SpaceRecord* space_records, CollisionRecord* collision_records, int collision_records_length, int B, int T);

__global__ void getSpaceRecords(Ball* balls, int ball_num, int B, int T);

__device__ void calIJKGPU(int& i, int& j, int& k, float x, float y, float z);

__device__ void makeBallRecordGPU(Ball* ball);

__device__ bool containsSpaceRecord(Ball& ball, SpaceRecord& sr);

__device__ bool recordEquals(SpaceRecord& r1, SpaceRecord& r2);

__device__ void collide(Ball& ball1, Ball& ball2);

// __global__ void findCollisionRecords(SpaceRecord* space_records, int space_records_length, CollisionRecord* collisionRecords);
#endif // !_SPACE_H_
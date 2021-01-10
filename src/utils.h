#ifndef _UTILS_H_
#define _UTILS_H_
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <random>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "ball.cuh"
#include "space.cuh"
#include "params.h"

// 从文件中读取球的坐标
bool readBalls(const std::string& filename, std::vector<Ball>& balls);

bool initData(Space** s, std::vector<Ball>& balls);

void makeBallRecord(Ball& ball);

void calIJK(int& i, int& j, int& k, float x, float y, float z);

#endif // !_UTILS_H_

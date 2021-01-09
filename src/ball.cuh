#ifndef _BALL_H_
#define _BALL_H_
#include "params.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class SubSpace;

typedef struct SpaceRecord {
	int i, j, k;
	int ball_idx;
	bool is_home;
} SpaceRecord;

class Ball {
public:
	// x、y、z 轴的坐标
	float x, y, z;
	// x、y、z 轴的速度
	float speed_x, speed_y, speed_z;
	// x、y、z 轴的加速度
	float acc_x, acc_y, acc_z;
	// 碰撞记录
	int record_num;
	SpaceRecord records[8];
	// 球的编号
	int idx;
	Ball(float _x, float _y, float _z);
	glm::vec3 getPosVec() const;
};

#endif // !_BALL_H_

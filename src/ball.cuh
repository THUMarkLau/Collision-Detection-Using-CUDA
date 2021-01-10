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
	int space_idx;
	bool operator ==(const SpaceRecord& sr)const;
	bool operator >=(const SpaceRecord& sr)const;
	bool operator <=(const SpaceRecord& sr)const;
	bool operator >(const SpaceRecord& sr)const;
	bool operator !=(const SpaceRecord& sr)const;
} SpaceRecord;

typedef struct CollisionRecord {
	// 在 records 中开始的位置
	int idx;
	// 维持的长度
	int length;
	// H 和 P 的个数
	int H;
	int P;
	// 子空间的 ID，在 0 - 7 之间
	char cell_id;
} CollisionRecord;

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
	bool containsRecord(SpaceRecord& record)const;
	glm::vec3 getPosVec() const;
};

#endif // !_BALL_H_

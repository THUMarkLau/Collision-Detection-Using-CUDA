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
	// �� records �п�ʼ��λ��
	int idx;
	// ά�ֵĳ���
	int length;
	// H �� P �ĸ���
	int H;
	int P;
	// �ӿռ�� ID���� 0 - 7 ֮��
	char cell_id;
} CollisionRecord;

class Ball {
public:
	// x��y��z �������
	float x, y, z;
	// x��y��z ����ٶ�
	float speed_x, speed_y, speed_z;
	// x��y��z ��ļ��ٶ�
	float acc_x, acc_y, acc_z;
	// ��ײ��¼
	int record_num;
	SpaceRecord records[8];
	// ��ı��
	int idx;
	Ball(float _x, float _y, float _z);
	bool containsRecord(SpaceRecord& record)const;
	glm::vec3 getPosVec() const;
};

#endif // !_BALL_H_

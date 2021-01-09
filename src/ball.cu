#include "ball.cuh"

Ball::Ball(float _x, float _y, float _z) {
	x = _x;
	y = _y;
	z = _z;
	speed_x = 0;
	speed_y = 0;
	speed_z = 0;
	acc_x = 0;
	acc_y = ACCLERATION_G;
	acc_z = 0;
	i = 0;
	j = 0;
	k = 0;
	idx = 0;
}

glm::vec3 Ball::getPosVec() const {
	return glm::vec3(x, y, z);
}
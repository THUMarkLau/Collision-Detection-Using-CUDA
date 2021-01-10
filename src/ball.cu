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
	idx = 0;
	record_num = 0;
}

glm::vec3 Ball::getPosVec() const {
	return glm::vec3(x, y, z);
}

bool Ball::containsRecord(SpaceRecord& record) const {
	for (int idx = 0; idx < record_num; ++idx) {
		if (record == records[idx]) {
			return true;
		}
	}
	return false;
}

bool SpaceRecord::operator ==(const SpaceRecord& record)const {
	return i == record.i && j == record.j && k == record.k;
}

bool SpaceRecord::operator >=(const SpaceRecord& sr)const {
	if (i < sr.i) {
		return false;
	}
	else if (i > sr.i) {
		return true;
	}
	if (j < sr.j) {
		return false;
	}
	else if (j > sr.j) {
		return true;
	}
	if (k < sr.k) {
		return false;
	}
	else if (k > sr.k) {
		return true;
	}
	return true;
}

bool SpaceRecord::operator <=(const SpaceRecord& sr)const {
	if (i < sr.i) {
		return true;
	}
	else if (i > sr.i) {
		return false;
	}
	if (j < sr.j) {
		return true;
	}
	else if (j > sr.j) {
		return false;
	}
	if (k < sr.k) {
		return true;
	}
	else if (k > sr.k) {
		return false;
	}
	return true;
}

bool SpaceRecord::operator >(const SpaceRecord& sr)const {
	return !((*this) <= sr);
}

bool SpaceRecord::operator !=(const SpaceRecord& sr)const {
	return !((*this) == sr);
}
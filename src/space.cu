#include "space.cuh"

__global__ void collisionDetectionAndUpdate(Ball* balls, int B, int T, int ballNums, float delta_time) {
	int i = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
	int j = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	for (int idx = 0; i * B + j + idx * T < ballNums; ++idx) {
		collisionDetection(&balls[i * B + j + idx * T]);
	}
	for (int idx = 0; i * B + j + idx * T < ballNums; ++idx) {
		update(&balls[i * B + j + idx * T], delta_time);
	}
}

__device__ void update(Ball* ball, float delta_time) {
	float acc_fri_x = ball->speed_x > 0 ? -ACC_FRICTION : ACC_FRICTION;
	ball->speed_x += ball->acc_x * delta_time;
	if (ball->speed_x + acc_fri_x * delta_time != 0 && sign(ball->speed_x + acc_fri_x * delta_time) * sign(ball->speed_x) > 0)
		ball->speed_x += acc_fri_x * delta_time;
	else
		ball->speed_x = 0;
	ball->x += ball->speed_x * delta_time;

	float acc_fri_z = ball->speed_z > 0 ? -ACC_FRICTION : ACC_FRICTION;
	ball->speed_z += ball->acc_z * delta_time;
	if (ball->speed_z + acc_fri_z * delta_time != 0 && sign(ball->speed_z + acc_fri_z * delta_time) * sign(ball->speed_z) > 0)
		ball->speed_z += acc_fri_z * delta_time;
	else
		ball->speed_z = 0;
	ball->z += ball->speed_z * delta_time;
	

	
	if (ball->y + ball->speed_y * delta_time < Y_LOWER_BOUND) {
		ball->y = Y_LOWER_BOUND - 1e-4;
	}
	else if (ball->y + ball->speed_y * delta_time > Y_UPPER_BOUND) {
		ball->y = Y_UPPER_BOUND + 1e-4;
	}
	else {
		ball->y += ball->speed_y * delta_time;
		ball->speed_y += ball->acc_y * delta_time;
	}	
}

__device__ void collisionDetection(Ball* balls) {
	checkBound(balls);
}

__device__ void checkBound(Ball* ball) {
	if (ball->x >= X_UPPER_BOUND) {
		ball->x = X_UPPER_BOUND - 1e-4;
		ball->speed_x *= -DECAY_FACTOR;
	}
	else if (ball->x <= X_LOWER_BOUND) {
		ball->x = X_LOWER_BOUND + 1e-4;
		ball->speed_x *= -DECAY_FACTOR;
	}
	if (ball->y >= Y_UPPER_BOUND) {
		ball->y = Y_UPPER_BOUND - 1e-4;
		ball->speed_y *= -DECAY_FACTOR;
	}
	else if (ball->y <= Y_LOWER_BOUND) {
		ball->y = Y_LOWER_BOUND + 1e-4;
		ball->speed_y *= -DECAY_FACTOR;
	}
	if (ball->z >= Z_UPPER_BOUND) {
		ball->z = Z_UPPER_BOUND - 1e-4;
		ball->speed_z *= -DECAY_FACTOR;
	}
	else if (ball->z <= Z_LOWER_BOUND) {
		ball->z = Z_LOWER_BOUND + 1e-4;
		ball->speed_z *= -DECAY_FACTOR;
	}
}

__device__ float getDistance(Ball* first_ball, Ball* second_ball) {
	return sqrt((first_ball->x - second_ball->x) * (first_ball->x - second_ball->x) +
		(first_ball->y - second_ball->y) * (first_ball->y - second_ball->y) +
		(first_ball->z - second_ball->z) * (first_ball->z - second_ball->z));
}

__device__ int binarySearch(int val, int length, int* list) {
	int left = 0, right = length - 1;
	while (left <= right) {
		int mid = (left + right) / 2;
		if (list[mid] == val) {
			return mid;
		}
		else if (list[mid] < val) {
			left = mid + 1;
		}
		else if (list[mid] > val) {
			right = mid - 1;
		}
	}
	return -1;
}

__device__ void insert(int val, int length, int* list) {
	int curPos = 0;
	// 效率慢的话改用二分的方法
	for (; curPos < length; curPos++) {
		if (list[curPos] <= val)
			continue;
		else {
			break;
		}
	}
	for (int i = length - 1; i >= curPos; --i) {
		list[i + 1] = list[i];
	}
	list[curPos] = val;
}

__device__ void remove(int val, int length, int* list) {
	int pos = binarySearch(val, length, list);
	if (pos == -1) {
		return;
	}
	for (int i = pos; i < length; ++i) {
		list[i] = list[i + 1];
	}
}

__device__ int sign(float x) {
	return x > 0 ? 1 : -1;
}

int findCollisionRecords(SpaceRecord* space_records, int space_records_length, CollisionRecord* collision_records) {
	int collision_time = 0;
	// 当前正在比较的块
	int cur_record_idx = 0;
	int H_cnt = 0, P_cnt = 0;
	for (int i = 1; i < space_records_length; ++i) {
		if (i == 13) {
			i;
		}
		if (space_records[i] != space_records[cur_record_idx]) {
			if (i - cur_record_idx > 1) {
				CollisionRecord cr = CollisionRecord();
				cr.idx = cur_record_idx;
				cr.length = i - cur_record_idx;
				int _i = space_records[cur_record_idx].i;
				int _j = space_records[cur_record_idx].j;
				int _k = space_records[cur_record_idx].k;
				cr.cell_id = _i % 2 * 2 + _j % 2 * 4 + _k % 2;
				cr.H = H_cnt;
				cr.P = P_cnt;
				// TODO: 记录 cell_id 和 H、P
				collision_records[collision_time++] = cr;
			}
			H_cnt = 0;
			P_cnt = 0;
			cur_record_idx = i;
			if (space_records[i].is_home) {
				H_cnt += 1;
			}
			else {
				P_cnt += 1;
			}
		}
		else {
			if (space_records[i].is_home) {
				H_cnt += 1;
			}
			else {
				P_cnt += 1;
			}
		}
	}
	return collision_time;
}

__global__ void doCollision(int round, Ball* balls, int ball_num, SpaceRecord* space_records, CollisionRecord* collision_records, int collision_records_length, int B, int T) {
	int i = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
	int j = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	float collision_bound = 2* BALL_RADIUS;
	for (int idx = 0; i * B + j + idx * T < collision_records_length; ++idx) {
		CollisionRecord& cr = collision_records[i * B + j + idx * T];
		if (cr.cell_id > round) {
			continue;
		}
		for (int l = 0; l < cr.length-1; ++l) {
			for (int m = l + 1; m < cr.length; ++m) {
				SpaceRecord& sr1 = space_records[cr.idx + l];
				SpaceRecord& sr2 = space_records[cr.idx + m];
				if (!sr1.is_home && !sr2.is_home)
					continue;
				if (balls[sr1.ball_idx].records[0].space_idx > balls[sr2.ball_idx].records[0].space_idx) {
					continue;
				}
				float delta_x = balls[sr1.ball_idx].x - balls[sr2.ball_idx].x;
				float delta_y = balls[sr1.ball_idx].y - balls[sr2.ball_idx].y;
				float delta_z = balls[sr1.ball_idx].z - balls[sr2.ball_idx].z;
				float distance = sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);
				// 碰撞了！
				if (distance < 2 * BALL_RADIUS) {
					collide(balls[sr1.ball_idx], balls[sr2.ball_idx]);
				}
			}
		}
	}
}

__global__ void getSpaceRecords(Ball* balls, int ball_num, int B, int T) {
	int i = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
	int j = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	const float X_MARGIN = ((X_UPPER_BOUND - X_LOWER_BOUND) / (float)SUBSPACE_X);
	const float Y_MARGIN = ((Y_UPPER_BOUND - Y_LOWER_BOUND) / (float)SUBSPACE_Y);
	const float Z_MARGIN = ((Z_UPPER_BOUND - Z_LOWER_BOUND) / (float)SUBSPACE_Z);
	for (int idx = 0; i * B + j + idx * T < ball_num; ++idx) {
		int cur_idx = i * B + j + idx * T;
		int _i = floor((balls[cur_idx].x - X_LOWER_BOUND) / X_MARGIN);
		int _j = floor((balls[cur_idx].y - Y_LOWER_BOUND) / Y_MARGIN);
		int _k = floor((balls[cur_idx].z - Z_LOWER_BOUND) / Z_MARGIN);
		_i = _i >= SUBSPACE_X ? SUBSPACE_X - 1 : _i;
		_j = _j >= SUBSPACE_Y ? SUBSPACE_Y - 1 : _j;
		_k = _k >= SUBSPACE_Z ? SUBSPACE_Z - 1 : _k;
		_i = _i < 0 ? 0 : _i;
		_j = _j < 0 ? 0 : _j;
		_k = _k < 0 ? 0 : _k;
		SpaceRecord sr = SpaceRecord();
		sr.ball_idx = cur_idx;
		sr.i = _i;
		sr.j = _j;
		sr.k = _k;
		sr.is_home = true;
		sr.space_idx = _i % 2 * 2 + _j % 2 * 4 + _k % 2;
		balls[cur_idx].records[0] = sr;
		balls[cur_idx].record_num = 1;

		makeBallRecordGPU(&balls[cur_idx]);
	}
}

__device__ void makeBallRecordGPU(Ball* ball) {
	int i = ball->records[0].i;
	int j = ball->records[0].j;
	int k = ball->records[0].k;
	float x, y, z;
	for (int l = 0; l < 3; ++l) {
		if (l == 0) {
			x = ball->x;
		}
		else if (l == 1) {
			x = ball->x + BALL_RADIUS;
			if (x > X_UPPER_BOUND) {
				continue;
			}
		}
		else {
			x = ball->x - BALL_RADIUS;
			if (x < X_LOWER_BOUND) {
				continue;
			}
		}
		for (int m = 0; m < 3; ++m) {
			if (m == 0) {
				y = ball->y;
			}
			else if (m == 1) {
				y = ball->y + BALL_RADIUS;
				if (y > Y_UPPER_BOUND)
					continue;
			}
			else {
				y = ball->y - BALL_RADIUS;
				if (y < Y_LOWER_BOUND)
					continue;
			}
			for (int n = 0; n < 3; ++n) {
				if (n == 0) {
					z = ball->z;
				}
				else if (n == 1) {
					z = ball->z + BALL_RADIUS;
					if (z > Z_UPPER_BOUND)
						continue;
				}
				else {
					z = ball->z - BALL_RADIUS;
					if (z < Z_LOWER_BOUND)
						continue;
				}
				int _i, _j, _k;
				calIJKGPU(_i, _j, _k, x, y, z);
				SpaceRecord sr = SpaceRecord();
				sr.i = _i;
				sr.j = _j;
				sr.k = _k;
				sr.is_home = false;
				if (!containsSpaceRecord(*ball, sr)) {
					sr.ball_idx = ball->idx;
					ball->records[ball->record_num++] = sr;
				}
			}
		}
	}
}

__device__ void calIJKGPU(int& i, int& j, int& k, float x, float y, float z) {
	int _i = (x - X_LOWER_BOUND) / ((X_UPPER_BOUND - X_LOWER_BOUND) / SUBSPACE_X);
	int _j = (y - Y_LOWER_BOUND) / ((Y_UPPER_BOUND - Y_LOWER_BOUND) / SUBSPACE_Y);
	int _k = (z - Z_LOWER_BOUND) / ((Z_UPPER_BOUND - Z_LOWER_BOUND) / SUBSPACE_Z);
	i = _i >= SUBSPACE_X ? SUBSPACE_X - 1 : _i;
	j = _j >= SUBSPACE_Y ? SUBSPACE_Y - 1 : _j;
	k = _k >= SUBSPACE_Z ? SUBSPACE_Z - 1 : _k;
}

__device__ bool containsSpaceRecord(Ball& ball, SpaceRecord& sr) {
	for (int i = 0; i < ball.record_num; ++i) {
		if (recordEquals(ball.records[i], sr)) {
			return true;
		}
	}
	return false;
}

__device__ bool recordEquals(SpaceRecord& r1, SpaceRecord& r2) {
	return r1.i == r2.i && r1.j == r2.j && r1.k == r2.k;
}

// 模拟两个球碰撞之后的反应
__device__ void collide(Ball& ball1, Ball& ball2) {
	// 两个球之间的连线
	float delta_x = ball1.x - ball2.x;
	float delta_y = ball1.y - ball2.y;
	float delta_z = ball1.z - ball2.z;
	float length = sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);
	delta_x /= length;
	delta_z /= length;
	delta_y /= length;
	glm::vec3 ball_direct = glm::vec3(delta_x, delta_y, delta_z);

	glm::vec3 ball1_speed = glm::vec3(ball1.speed_x, ball1.speed_y, ball1.speed_z);
	glm::vec3 ball2_speed = glm::vec3(ball2.speed_x, ball2.speed_y, ball2.speed_z);

	glm::vec3 ball1_speed_projection = glm::dot(ball1_speed, ball_direct) * ball_direct;
	glm::vec3 ball2_speed_projection = glm::dot(ball2_speed, ball_direct) * ball_direct;

	ball1.speed_x -= ball1_speed_projection.x;
	ball1.speed_y -= ball1_speed_projection.y;
	ball1.speed_z -= ball1_speed_projection.z;

	ball1.speed_x += ball2_speed_projection.x * DECAY_FACTOR;
	ball1.speed_y += ball2_speed_projection.y * DECAY_FACTOR;
	ball1.speed_z += ball2_speed_projection.z * DECAY_FACTOR;
	
	ball2.speed_x -= ball2_speed_projection.x;
	ball2.speed_y -= ball2_speed_projection.y;
	ball2.speed_z -= ball2_speed_projection.z;

	ball2.speed_x += ball1_speed_projection.x * DECAY_FACTOR;
	ball2.speed_y += ball1_speed_projection.y * DECAY_FACTOR;
	ball2.speed_z += ball1_speed_projection.z * DECAY_FACTOR;

	ball1.x -= ball_direct.x * BALL_RADIUS * 2;
	ball1.y -= ball_direct.y * BALL_RADIUS * 2;
	ball1.z -= ball_direct.z * BALL_RADIUS * 2;

	ball2.x += ball_direct.x * BALL_RADIUS * 2;
	ball2.y += ball_direct.y * BALL_RADIUS * 2;
	ball2.z += ball_direct.z * BALL_RADIUS * 2;
}
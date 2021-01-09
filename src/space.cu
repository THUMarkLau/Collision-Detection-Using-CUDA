#include "space.cuh"

__global__ void collisionDetectionAndUpdate(Space* space, Ball* balls) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	SubSpace& subspace = space->subspaces[i][j][k];
	collisionDetection(&space->subspaces[i][j][k], balls);
	for (auto i = 0; i != subspace.ball_num; ++i) {
		update(space, &balls[i]);
	}
}

__device__ void update(Space* space, Ball* ball) {
	float acc_fri_x = ball->speed_x > 0 ? -ACC_FRICTION : ACC_FRICTION;
	ball->speed_x += ball->acc_x;
	if (ball->speed_x + acc_fri_x != 0 && sign(ball->speed_x + acc_fri_x) * sign(ball->speed_x) > 0)
		ball->speed_x += acc_fri_x;
	else
		ball->speed_x = 0;
	ball->x += ball->speed_x;

	float acc_fri_z = ball->speed_z > 0 ? -ACC_FRICTION : ACC_FRICTION;
	ball->speed_z += ball->acc_z;
	if (ball->speed_z + acc_fri_z != 0 && sign(ball->speed_z + acc_fri_z) * sign(ball->speed_z) > 0)
		ball->speed_z += acc_fri_z;
	else
		ball->speed_z = 0;
	ball->z += ball->speed_z;
	

	ball->speed_y += ball->acc_y;
	ball->y += ball->speed_y;

	int i = ball->x / (MAX_X / SUBSPACE_X);
	int j = ball->y / (MAX_Y / SUBSPACE_Y);
	int k = ball->z / (MAX_Z / SUBSPACE_Z);
	i = i >= SUBSPACE_X ? SUBSPACE_X - 1 : i;
	j = j >= SUBSPACE_Y ? SUBSPACE_Y - 1 : j;
	k = k >= SUBSPACE_Z ? SUBSPACE_Z - 1 : k;

	if (i != ball->i || j != ball->j || k == ball->k) {
		/*remove(ball->idx, space->subspaces[ball->i][ball->j][ball->k].ball_num, space->subspaces[ball->i][ball->j][ball->k].balls);
		insert(ball->idx, space->subspaces[i][j][k].ball_num, space->subspaces[i][j][k].balls);*/
	}
}

__device__ void collisionDetection(SubSpace* subspace, Ball* balls) {
	for (int i = 0; i < subspace->ball_num; ++i) {
		Ball* first_ball = &balls[subspace->balls[i]];
		checkBound(first_ball);
		/*for (int j = i + 1; j < subspace->ball_num; ++j) {
			Ball* second_ball = &balls[subspace->balls[j]];
			float distance = getDistance(first_ball, second_ball);

		}*/
	}
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
#include "utils.h"

bool readBalls(const std::string& filename, std::vector<Ball>& balls) {
	std::cout << "正在加载 " << filename << "..." << std::endl;
	if (filename.size() == 0) {
		std::cout << "文件名为空" << std::endl;
		return false;
	}
	FILE* file = fopen(filename.c_str(), "r");
	if (!file) {
		std::cout << "文件 " << filename << " 不存在" << std::endl;
		return false;
	}
	fseek(file, 0, SEEK_SET);
	char buffer[1000];
	fgets(buffer, 300, file);
	int ballNums = 0;
	std::uniform_real_distribution<float> u(-2, 2);
	std::default_random_engine e;
	sscanf(buffer, "%d", &ballNums);
	for (int i = 0; i < ballNums; ++i) {
		float x, y, z;
		fgets(buffer, 300, file);
		sscanf(buffer, "%f %f %f", &x, &y, &z);
		Ball ball = Ball(x, y, z);
		ball.speed_x = u(e);
		ball.speed_y = u(e);
		ball.speed_z = u(e);
		ball.idx = i;
		balls.push_back(ball);
	}
	std::cout << "加载成功，共读入 " << ballNums << " 个球" << std::endl;
	return true;
}

// 初始化用于计算的数据
// -----------------------------------
bool initData(Space** s, std::vector<Ball>& balls) {

	if (!readBalls(BALL_FILE_PATH, balls)) {
		return false;
	}
	// 为空间 Space 分配内存
	size_t space_size = sizeof(Space);
	cudaMallocManaged((void**)s, space_size);
	Space* space = *s;

	// 记录第 i 个子空间中球的数量
	int totalRecord = 0;
	for (int idx = 0, ball_nums = balls.size(); idx < ball_nums; ++idx) {
		Ball& ball = balls[idx];
		int i = (ball.x - X_LOWER_BOUND) / ((X_UPPER_BOUND - X_LOWER_BOUND) / SUBSPACE_X);
		int j = (ball.y - Y_LOWER_BOUND) / ((Y_UPPER_BOUND - Y_LOWER_BOUND) / SUBSPACE_Y);
		int k = (ball.z - Z_LOWER_BOUND) / ((Z_UPPER_BOUND - Z_LOWER_BOUND) / SUBSPACE_Z);
		i = i >= SUBSPACE_X ? SUBSPACE_X - 1 : i;
		j = j >= SUBSPACE_Y ? SUBSPACE_Y - 1 : j;
		k = k >= SUBSPACE_Z ? SUBSPACE_Z - 1 : k;
		SpaceRecord sr = SpaceRecord();
		sr.ball_idx = idx;
		sr.i = i;
		sr.j = j;
		sr.k = k;
		sr.is_home = true;
		sr.space_idx = i % 2 * 2 + j % 2 * 4 + k % 2;
		ball.records[0] = sr;
		ball.record_num = 1;

		makeBallRecord(ball);
		totalRecord += ball.record_num;
	}
	printf("Total Records: %d\n", totalRecord);

	return true;
}


void makeBallRecord(Ball& ball) {
	int i = ball.records[0].i;
	int j = ball.records[0].j;
	int k = ball.records[0].k;
	float x, y, z;
	for (int l = 0; l < 3; ++l) {
		if (l == 0) {
			x = ball.x;
		}
		else if (l == 1) {
			x = ball.x + BALL_RADIUS;
			if (x > X_UPPER_BOUND) {
				continue;
			}
		}
		else {
			x = ball.x - BALL_RADIUS;
			if (x < X_LOWER_BOUND) {
				continue;
			}
		}
		for (int m = 0; m < 3; ++m) {
			if (m == 0) {
				y = ball.y;
			}
			else if (m == 1) {
				y = ball.y + BALL_RADIUS;
				if (y > Y_UPPER_BOUND)
					continue;
			}
			else {
				y = ball.y - BALL_RADIUS;
				if (y < Y_LOWER_BOUND)
					continue;
			}
			for (int n = 0; n < 3; ++n) {
				if (n == 0) {
					z = ball.z;
				}
				else if (n == 1) {
					z = ball.z + BALL_RADIUS;
					if (z > Z_UPPER_BOUND)
						continue;
				}
				else {
					z = ball.z - BALL_RADIUS;
					if (z < Z_LOWER_BOUND)
						continue;
				}
				int _i, _j, _k;
				calIJK(_i, _j, _k, x, y, z);
				SpaceRecord sr = SpaceRecord();
				sr.i = _i;
				sr.j = _j;
				sr.k = _k;
				sr.is_home = false;
				if (!ball.containsRecord(sr)) {
					sr.ball_idx = ball.idx;
					ball.records[ball.record_num++] = sr;
				}
			}
		}
	}
}

void calIJK(int& i, int& j, int& k, float x, float y, float z) {
	int _i = (x - X_LOWER_BOUND) / ((X_UPPER_BOUND - X_LOWER_BOUND) / SUBSPACE_X);
	int _j = (y - Y_LOWER_BOUND) / ((Y_UPPER_BOUND - Y_LOWER_BOUND) / SUBSPACE_Y);
	int _k = (z - Z_LOWER_BOUND) / ((Z_UPPER_BOUND - Z_LOWER_BOUND) / SUBSPACE_Z);
	i = _i >= SUBSPACE_X ? SUBSPACE_X - 1 : _i;
	j = _j >= SUBSPACE_Y ? SUBSPACE_Y - 1 : _j;
	k = _k >= SUBSPACE_Z ? SUBSPACE_Z - 1 : _k;
}
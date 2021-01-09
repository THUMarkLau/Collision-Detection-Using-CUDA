#include "utils.h"

bool readBalls(const std::string& filename, std::vector<Ball>& balls) {
	std::cout << "���ڼ��� " << filename << "..." << std::endl;
	if (filename.size() == 0) {
		std::cout << "�ļ���Ϊ��" << std::endl;
		return false;
	}
	FILE* file = fopen(filename.c_str(), "r");
	if (!file) {
		std::cout << "�ļ� " << filename << " ������" << std::endl;
		return false;
	}
	fseek(file, 0, SEEK_SET);
	char buffer[1000];
	fgets(buffer, 300, file);
	int ballNums = 0;
	std::uniform_real_distribution<float> u(-0.01, 0.01);
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
	std::cout << "���سɹ��������� " << ballNums << " ����" << std::endl;
	return true;
}

// ��ʼ�����ڼ��������
// -----------------------------------
bool initData(Space** s, std::vector<Ball>& balls) {

	if (!readBalls("..\\..\\resources\\balls.txt", balls)) {
		return false;
	}
	// Ϊ�ռ� Space �����ڴ�
	size_t space_size = sizeof(Space);
	cudaMallocManaged((void**)s, space_size);
	Space* space = *s;

	// ��¼�� i ���ӿռ����������
	std::unordered_map<int, std::vector<int>> map;
	for (int idx = 0, ball_nums = balls.size(); idx < ball_nums; ++idx) {
		Ball ball = balls[idx];
		int i = ball.x / (MAX_X / SUBSPACE_X);
		int j = ball.y / (MAX_Y / SUBSPACE_Y);
		int k = ball.z / (MAX_Z / SUBSPACE_Z);
		i = i >= SUBSPACE_X ? SUBSPACE_X - 1 : i;
		j = j >= SUBSPACE_Y ? SUBSPACE_Y - 1 : j;
		k = k >= SUBSPACE_Z ? SUBSPACE_Z - 1 : k;

		int key = k + j * 15 + i * 15 * 15;
		if (map.find(key) != map.end()) {
			map[key].push_back(idx);
		}
		else {
			map[key] = std::vector<int>({ idx });
		}
	}
	
	// ���򴫵�ÿ���ӿռ���
	for (int i = 0; i < SUBSPACE_X; ++i) {
		for (int j = 0; j < SUBSPACE_Y; ++j) {
			for (int k = 0; k < SUBSPACE_Z; ++k) {
				int key = k + j * 15 + i * 15 * 15;
				if (map.find(key) == map.end()) {
					continue;
				}
				int ball_num = map[key].size();
				std::vector<int>& ball_idx = map[key];
				// �����ڴ�
				cudaMallocManaged((void**)&space->subspaces[i][j][k].balls, sizeof(int) * balls.size());
				space->subspaces[i][j][k].ball_num = ball_num;
				for (int l = 0; l < ball_num; ++l) {
					space->subspaces[i][j][k].balls[l] = ball_idx[l];
					balls[ball_idx[l]].i = i;
					balls[ball_idx[l]].j = j;
					balls[ball_idx[l]].k = k;
				}
				for (int l = 0; l < ball_num; ++l) {
					printf("%d ", space->subspaces[i][j][k].balls[l]);
				}
				printf("\n");
				continue;
			}
		}
	}

	return true;
}
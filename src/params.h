#ifndef _PARAMS_H_
#define _PARAMS_H_

#define SCR_WIDTH 900 // 600
#define SCR_HEIGHT 1200 // 800
#define X_SEGMENTS 50
#define Y_SEGMENTS 50
#define PI 3.14159265358979323846f

//#define VERTEX_SHADER_PATH "..\\..\\src\\shader\\vertex_shader.glsl"
//#define FRAGMENT_SHADER_PATH "..\\..\\src\\shader\\fragment_shader.glsl"

#define VERTEX_SHADER_PATH ".\\shader\\vertex_shader.glsl"
#define FRAGMENT_SHADER_PATH ".\\shader\\fragment_shader.glsl"
//#define BALL_FILE_PATH "..\\..\\resources\\balls.txt"
#define BALL_FILE_PATH ".\\ball.txt"
#define MAX_X 100.0
#define MAX_Y 100.0
#define MAX_Z 100.0

#define SUBSPACE_X 16
#define SUBSPACE_Y 16
#define SUBSPACE_Z 16

#define BALL_RADIUS 0.02f

#define ACCLERATION_G -5.0f

#define DECAY_FACTOR 0.7
#define X_UPPER_BOUND 2.25f
#define X_LOWER_BOUND -2.25f
#define Y_UPPER_BOUND 3.0f
#define Y_LOWER_BOUND 0.0f
#define Z_UPPER_BOUND 3.0f
#define Z_LOWER_BOUND -1.5f
const float X_MARGIN = ((X_UPPER_BOUND - X_LOWER_BOUND) / (float)SUBSPACE_X);
const float Y_MARGIN = ((Y_UPPER_BOUND - Y_LOWER_BOUND) / (float)SUBSPACE_Y);
const float Z_MARGIN = ((Z_UPPER_BOUND - Z_LOWER_BOUND) / (float)SUBSPACE_Z);
#define DELTA 0.05f
#define ACC_FRICTION 0.1

#endif // !_PARAMS_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "shader.h"
#include "camera.h"
#include "space.cuh"
#include "utils.h"
#include "params.h"
#include "material.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
void genBall(std::vector<float>& i, std::vector<int>& j);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);

// variables
bool firstMouse = true;
int lastX, lastY;
float delta_time;
// 上一帧渲染的时间
float last_frame_time = -1;
Camera camera;

int main()
{
    Space* space = nullptr;
    std::vector<Ball> balls_vec;
    if (!initData(&space, balls_vec)) {
        return 0;
    }
    int balls_num = balls_vec.size();

    // 将 Ball 从 vector 转为数组
    Ball* balls = nullptr;
    cudaMallocManaged((void**)&balls, sizeof(Ball) * balls_vec.size());
    for (int i = 0; i < balls_num; ++i) {
        balls[i] = balls_vec[i];
    }
    

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Collision Detection", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    // glfwSetCursorPosCallback(window, mouse_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // 生成绘制球体的各个顶点
    std::vector<float> sphereVertices;
    std::vector<int> sphereIndices;
    genBall(sphereVertices, sphereIndices);
    

    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    //生成并绑定球体的VAO和VBO
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    //将顶点数据绑定至当前默认的缓冲中
    glBufferData(GL_ARRAY_BUFFER, sphereVertices.size() * sizeof(float), &sphereVertices[0], GL_STATIC_DRAW);
    GLuint element_buffer_object;//EBO
    glGenBuffers(1, &element_buffer_object);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_object);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphereIndices.size() * sizeof(int), &sphereIndices[0], GL_STATIC_DRAW);
    //设置顶点属性指针
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    //解绑VAO和VBO
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    float backgroundVertex[] = {
        X_UPPER_BOUND + BALL_RADIUS, Y_UPPER_BOUND + BALL_RADIUS, Z_LOWER_BOUND - BALL_RADIUS,
        X_UPPER_BOUND + BALL_RADIUS, Y_LOWER_BOUND - BALL_RADIUS, Z_LOWER_BOUND - BALL_RADIUS,
        X_LOWER_BOUND - BALL_RADIUS, Y_LOWER_BOUND - BALL_RADIUS, Z_LOWER_BOUND - BALL_RADIUS,
        X_LOWER_BOUND - BALL_RADIUS, Y_UPPER_BOUND + BALL_RADIUS, Z_LOWER_BOUND - BALL_RADIUS,
        X_UPPER_BOUND + BALL_RADIUS, Y_LOWER_BOUND - BALL_RADIUS, Z_UPPER_BOUND + BALL_RADIUS,
        X_UPPER_BOUND + BALL_RADIUS, Y_UPPER_BOUND + BALL_RADIUS, Z_UPPER_BOUND + BALL_RADIUS,
        X_LOWER_BOUND - BALL_RADIUS, Y_LOWER_BOUND - BALL_RADIUS, Z_UPPER_BOUND + BALL_RADIUS,
        X_LOWER_BOUND - BALL_RADIUS, Y_UPPER_BOUND + BALL_RADIUS, Z_UPPER_BOUND + BALL_RADIUS
    };
    int backgroundIndices[] = {
        0, 1, 3,
        1, 2, 3, // Back
        0, 1, 4,
        0, 4, 5,
        2, 3, 6,
        3, 6, 7, // Side
        2, 1, 6,
        1, 4, 6, // Bottom
    };

    unsigned int backgroundVBO, backgroundVAO;
    glGenVertexArrays(1, &backgroundVAO);
    glGenBuffers(1, &backgroundVBO);
    glBindVertexArray(backgroundVAO);
    glBindBuffer(GL_ARRAY_BUFFER, backgroundVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(backgroundVertex), backgroundVertex, GL_STATIC_DRAW);

    unsigned int backgroundBackEBO;
    glGenBuffers(1, &backgroundBackEBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, backgroundBackEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(backgroundIndices), backgroundIndices, GL_STATIC_DRAW);

    //设置顶点属性指针
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    //解绑VAO和VBO
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    
    Shader shader = Shader(VERTEX_SHADER_PATH, FRAGMENT_SHADER_PATH);
    glm::vec3 specular = glm::vec3(0.8, 0.8, 0.8);
    glm::vec3 color(0.9);
    Material material = Material(color, color, specular, 128, &shader);
    Material backgroundBackMaterial = Material(glm::vec3(0.2), glm::vec3(0.2), specular, 32, &shader);
    Material backgroundSideMaterial = Material(glm::vec3(0.3, 0.2, 0.1), glm::vec3(0.3, 0.2, 0.1), specular, 128, &shader);
    Material backgroundBottomMaterial = Material(glm::vec3(0.6, 0.2, 0.4), glm::vec3(0.6, 0.2, 0.4), specular, 128, &shader);


    // 开始进行运算
    dim3 threadPerBlock(SUBSPACE_X / 4, SUBSPACE_Y / 4, SUBSPACE_Z / 4);
    dim3 numBlocks(4, 4, 4);
    while (!glfwWindowShouldClose(window))
    {
        float currentFrame = glfwGetTime();
        delta_time = currentFrame - last_frame_time;
        last_frame_time = currentFrame;
        processInput(window);
        // 碰撞检测，并且更新每个球的位置
        collisionDetectionAndUpdate << <numBlocks, threadPerBlock >> > (space, balls);
        cudaDeviceSynchronize();

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        //glEnable(GL_CULL_FACE);
        //glCullFace(GL_BACK);
        shader.use();
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        shader.setMatrix4("projection", projection);

        // 获得视角矩阵
        glm::mat4 view = camera.GetViewMatrix();
        shader.setMatrix4("view", view);

        // 设置光线信息
        shader.setVec3("light.ambient", 0.8f, 0.8f, 0.8f);
        shader.setVec3("light.diffuse", 0.6f, 0.6f, 0.6f); // 将光照调暗了一些以搭配场景
        shader.setVec3("light.specular", 1.0f, 1.0f, 1.0f);
        shader.setVec3("light.position", (X_UPPER_BOUND + X_LOWER_BOUND) / 2, Y_UPPER_BOUND, (Z_UPPER_BOUND + Z_LOWER_BOUND) / 2);
        shader.setFloat("light.constant", 1.0f);
        shader.setFloat("light.linear", 0.2f);
        shader.setFloat("light.quadratic", 0.00001f);
        shader.setVec3("viewPos", camera.Position);
        // 绘制背景
        backgroundBackMaterial.useParams();
        glm::mat4 pos = glm::mat4(1.0f);
        shader.setMatrix4("model", glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, 0)));
        glBindVertexArray(backgroundVAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        backgroundSideMaterial.useParams();
        glDrawElements(GL_TRIANGLES, 12, GL_UNSIGNED_INT, (void*)(sizeof(int) * 6));
        backgroundBottomMaterial.useParams();
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void*)(sizeof(int) * 18));

        material.useParams();
        for (int i = 0; i < balls_num; ++i) {
            auto pos = glm::translate(glm::mat4(1), balls[i].getPosVec());
            shader.setMatrix4("model", pos);
            glBindVertexArray(VAO);
            glDrawElements(GL_TRIANGLES, X_SEGMENTS * Y_SEGMENTS * 6, GL_UNSIGNED_INT, 0);
        }
        

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, delta_time);
    else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, delta_time);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, delta_time);
    else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, delta_time);
    else if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
        camera.ProcessMouseMovement(-5, 0);
    else if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
        camera.ProcessMouseMovement(5, 0);
    else if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        camera.ProcessMouseMovement(0, 1);
    else if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
        camera.ProcessMouseMovement(0, -1);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

//处理鼠标事件
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    // 鼠标的位移
    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;

    lastX = xpos;
    lastY = ypos;


    camera.ProcessMouseMovement(xoffset, yoffset);
}

void genBall(std::vector<float> &sphereVertices, std::vector<int> &sphereIndices) {
    float coff = BALL_RADIUS;
    for (int y = 0; y <= Y_SEGMENTS; y++)
    {
        for (int x = 0; x <= X_SEGMENTS; x++)
        {
            float xSegment = (float)x / (float)X_SEGMENTS;
            float ySegment = (float)y / (float)Y_SEGMENTS;
            float xPos = std::cos(xSegment * 2.0f * PI) * std::sin(ySegment * PI ) * coff;
            float yPos = std::cos(ySegment * PI) * coff;
            float zPos = std::sin(xSegment * 2.0f * PI) * std::sin(ySegment * PI) * coff;
            sphereVertices.push_back(xPos);
            sphereVertices.push_back(yPos);
            sphereVertices.push_back(zPos);
        }
    }

    //生成球的Indices
    for (int i = 0; i < Y_SEGMENTS; i++)
    {
        for (int j = 0; j < X_SEGMENTS; j++)
        {
            sphereIndices.push_back(i * (X_SEGMENTS + 1) + j);
            sphereIndices.push_back((i + 1) * (X_SEGMENTS + 1) + j);
            sphereIndices.push_back((i + 1) * (X_SEGMENTS + 1) + j + 1);
            sphereIndices.push_back(i * (X_SEGMENTS + 1) + j);
            sphereIndices.push_back((i + 1) * (X_SEGMENTS + 1) + j + 1);
            sphereIndices.push_back(i * (X_SEGMENTS + 1) + j + 1);
        }
    }

}
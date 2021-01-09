#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "Shader.h"

class Material {
private:
    glm::vec3 ambient;
    glm::vec3 diffuse;
    glm::vec3 specular;
    float shininess;
    Shader* shader;
public:
    Material(glm::vec3 _ambient = glm::vec3(0.5), glm::vec3 _diffuse = glm::vec3(0.5), glm::vec3 _specular = glm::vec3(1), float _shininess = 64, Shader* _shader = nullptr) {
        ambient = _ambient;
        diffuse = _diffuse;
        specular = _specular;
        shininess = _shininess;
        shader = _shader;
    }

    void setShader(Shader* s);
    void useParams();
};
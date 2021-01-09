#include "material.h"

void Material::setShader(Shader* s) {
    shader = s;
}

void Material::useParams() {
    shader->setVec3("material.ambient", ambient);
    shader->setVec3("material.diffuse", diffuse);
    shader->setVec3("material.specular", specular);
    shader->setFloat("material.shininess", shininess);
}
#pragma once

#include "render/base/RenderCamera.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

template<typename T>
class ScreenClick2d
{
public:
    ScreenClick2d(const RenderCamera& render, const T& min, const T& max)
        : render {render}
        , min {min}
        , max {max}
    {
    }

    bool operator()(const int x, const int y, T& intPos)
    {
        const auto size = render.GetSize();
        const float normalizedX = (x * 2.0f) / size.first - 1.0f;
        const float normalizedY = 1.0f - (y * 2.0f) / size.second;

        const glm::vec4 rayClip = glm::vec4(normalizedX, normalizedY, -1.0, 1.0);
        const glm::mat4 invProjection = glm::inverse(render.GetProjectionMatrix());
        const glm::mat4 invView = glm::inverse(render.GetViewMatrix());

        glm::vec4 rayEye = invProjection * rayClip;
        rayEye = glm::vec4(rayEye.x, rayEye.y, -1.0, 0.0);

        glm::vec4 rayWorld = invView * rayEye;
        rayWorld = glm::normalize(rayWorld);

        auto checkIntersection = [](const glm::vec3& rayOrigin, const glm::vec3& rayDirection, 
            const glm::vec3& planeNormal, const glm::vec3& planePoint,
            const T& minBounds, const T& maxBounds, 
            T& outPos) 
            {
                const float denom = glm::dot(planeNormal, rayDirection);
                if (denom < 1e-6)
                {
                    return false;
                }

                const glm::vec3 planeVec = planePoint - rayOrigin;
                const float t = glm::dot(planeVec, planeNormal) / denom;

                if (t >= 0)
                {
                    const glm::vec3 pos = rayOrigin + t * rayDirection;

                    outPos = {pos.x, pos.y};
                    outPos.x = -outPos.x;

                    return outPos.x >= minBounds.x && outPos.y >= minBounds.y &&
                            outPos.x < maxBounds.x && outPos.y < maxBounds.y;
                }

                return t >= 0;
            };

        return checkIntersection(render.GetPosition(), rayWorld, {0, 0, 1}, {0, 0, 0}, min, max, intPos);
    }

private:
    const RenderCamera& render;
    const T& min;
    const T& max;
};
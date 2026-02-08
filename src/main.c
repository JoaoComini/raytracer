#include <float.h>
#include <omp.h>
#include <raylib.h>
#include <raymath.h>
#include <stdbool.h>
#include <stdint.h>
#include <threads.h>

#define SCREEN_WIDTH 1024
#define SCREEN_HEIGHT 768

#define SPHERE_COUNT 5
#define MAX_DEPTH 10

typedef enum { MATERIAL_LAMBERTIAN, MATERIAL_METAL, MATERIAL_DIELETRIC } SphereMaterialType;

typedef struct {
    SphereMaterialType type;
    Color color;
    float refraction_index;
} SphereMaterial;

typedef struct {
    SphereMaterial material;
    Vector3 position;
    float radius;
} Sphere;

typedef struct {
    Vector3 point;
    Vector3 normal;
    float distance;
    bool front_face;
} HitResult;

Color trace_ray(Ray ray, Sphere *spheres, int depth);
bool sphere_hit(Ray, Sphere sphere, HitResult *result);

Vector3 random_unit_vector();
Vector3 random_vector();
float random_float();

Color pixels[SCREEN_HEIGHT][SCREEN_WIDTH];
Vector3 accumulator[SCREEN_HEIGHT][SCREEN_WIDTH];

int main(void) {
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Raytracer");

    SphereMaterial red = {.type = MATERIAL_DIELETRIC, .refraction_index = 1.33};
    SphereMaterial blue = {.type = MATERIAL_LAMBERTIAN, .color = BLUE};
    SphereMaterial brown = {.type = MATERIAL_METAL, .color = BROWN};
    SphereMaterial ground = {.type = MATERIAL_LAMBERTIAN, .color = DARKGREEN};

    Sphere spheres[SPHERE_COUNT];

    spheres[0] = (Sphere){.position = {-1, 0, 3}, .radius = 0.5f, .material = red};
    spheres[1] = (Sphere){.position = {1, 0, 3}, .radius = 0.5f, .material = blue};
    spheres[2] = (Sphere){.position = {0, 0, 3}, .radius = 0.5f, .material = brown};
    spheres[3] = (Sphere){.position = {0, -100.5f, 0}, .radius = 100.f, .material = ground};

    Camera camera = {.position = {2, 0.5, 1},
                     .target = {0, 0, 3},
                     .up = {0, 1, 0},
                     .fovy = 60,
                     .projection = CAMERA_PERSPECTIVE};

    Image image = {.data = pixels,
                   .width = SCREEN_WIDTH,
                   .height = SCREEN_HEIGHT,
                   .mipmaps = 1,
                   .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8};

    Texture2D texture = LoadTextureFromImage(image);

    Vector3 light_dir = Vector3Normalize((Vector3){0, 1, -0.8});

    int total_samples = 1;
    while (!WindowShouldClose()) {
        BeginDrawing();

#pragma omp parallel for collapse(2) schedule(dynamic)
        for (int x = 0; x < SCREEN_WIDTH; x++) {
            for (int y = 0; y < SCREEN_HEIGHT; y++) {
                float x_offset = random_float() - 0.5;
                float y_offset = random_float() - 0.5;

                Ray ray = GetScreenToWorldRay((Vector2){x + x_offset, y + y_offset}, camera);

                Color sample = trace_ray(ray, spheres, 0);

                accumulator[y][x].x += sample.r;
                accumulator[y][x].y += sample.g;
                accumulator[y][x].z += sample.b;

                pixels[y][x] = (Color){.r = accumulator[y][x].x / total_samples,
                                       .g = accumulator[y][x].y / total_samples,
                                       .b = accumulator[y][x].z / total_samples,
                                       .a = 255};
            }
        }

        total_samples += 1;

        UpdateTexture(texture, pixels);

        DrawTexture(texture, 0, 0, WHITE);
        DrawFPS(10, 10);
        EndDrawing();
    }

    CloseWindow();

    return 0;
}

Color trace_ray(Ray ray, Sphere *spheres, int depth) {
    if (depth >= MAX_DEPTH) {
        return BLACK;
    }

    Sphere nearest_sphere;
    HitResult nearest_result;
    nearest_result.distance = FLT_MAX;
    bool has_any_hit = false;

    for (int i = 0; i < SPHERE_COUNT; i++) {
        Sphere sphere = spheres[i];

        HitResult result;
        bool hit = sphere_hit(ray, sphere, &result);

        if (!hit || result.distance < 0.001) {
            continue;
        }

        if (result.distance < nearest_result.distance) {
            has_any_hit = true;
            nearest_result = result;
            nearest_sphere = sphere;
        }
    }

    if (!has_any_hit) {
        float factor = 0.5 * (ray.direction.y + 1);
        return ColorLerp(WHITE, (Color){.r = 255 * 0.5, .g = 255 * 0.7, .b = 255, .a = 255},
                         factor);
    }

    SphereMaterial material = nearest_sphere.material;

    Ray scattered;
    Color attenuation;

    switch (material.type) {
    case MATERIAL_LAMBERTIAN: {
        scattered = (Ray){.position = nearest_result.point,
                          .direction = Vector3Add(nearest_result.normal, random_unit_vector())};
        attenuation = material.color;
        break;
    }
    case MATERIAL_METAL:
        scattered = (Ray){.position = nearest_result.point,
                          .direction = Vector3Reflect(ray.direction, nearest_result.normal)};
        attenuation = material.color;
        break;

    case MATERIAL_DIELETRIC:
        attenuation = WHITE;

        float ri = nearest_result.front_face ? (1.0 / material.refraction_index)
                                             : material.refraction_index;

        Vector3 refracted = Vector3Refract(ray.direction, nearest_result.normal, ri);

        scattered = (Ray){.position = nearest_result.point, .direction = refracted};
    }

    return ColorTint(attenuation, trace_ray(scattered, spheres, depth + 1));
}

bool sphere_hit(Ray ray, Sphere sphere, HitResult *result) {
    Vector3 oc = Vector3Subtract(sphere.position, ray.position);
    float a = Vector3LengthSqr(ray.direction);
    float h = Vector3DotProduct(ray.direction, oc);
    float c = Vector3LengthSqr(oc) - sphere.radius * sphere.radius;
    float discriminant = h * h - a * c;

    if (discriminant < 0.f) {
        return false;
    }

    result->distance = (h - sqrtf(discriminant)) / a;
    result->point = Vector3Add(ray.position, Vector3Scale(ray.direction, result->distance));
    result->normal =
        Vector3Scale(Vector3Subtract(result->point, sphere.position), 1 / sphere.radius);
    result->front_face = true;

    if (Vector3DotProduct(result->normal, ray.direction) > 0) {
        result->normal = Vector3Negate(result->normal);
        result->front_face = false;
    }

    return true;
}

Vector3 random_unit_vector() {
    while (true) {
        Vector3 vector = (Vector3){
            .x = 2 * random_float() - 1, .y = 2 * random_float() - 1, .z = 2 * random_float() - 1};

        if (Vector3LengthSqr(vector) <= 1) {
            return Vector3Normalize(vector);
        }
    }
}

Vector3 random_vector() {
    return (Vector3){.x = random_float(), .y = random_float(), .z = random_float()};
}

static _Thread_local uint32_t rng;

static inline uint32_t xorshift32(void) {
    uint32_t x = rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return rng = x;
}

float random_float(void) {
    if (!rng) {
        rng = (uint32_t)time(NULL) ^ (uint32_t)(uintptr_t)&rng;
    }
    return (xorshift32() >> 8) * (1.0f / 16777216.0f);
}

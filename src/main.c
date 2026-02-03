#include <float.h>
#include <raylib.h>
#include <raymath.h>
#include <stdbool.h>

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600

#define SPHERE_COUNT 5

typedef struct {
    Color color;
    float specular;
    float shininess;
} SphereMaterial;

typedef struct {
    SphereMaterial material;
    Vector3 position;
    float radius;
} Sphere;

int main(void) {
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Raytracer");

    SphereMaterial red = {.color = RED, .specular = 1.f, .shininess = 256};
    SphereMaterial blue = {.color = BLUE, .specular = 0.5f, .shininess = 16};
    SphereMaterial brown = {.color = BROWN, .specular = 0.f, .shininess = 1};

    Sphere spheres[SPHERE_COUNT];

    spheres[0] = (Sphere){.position = {0, 0, 10}, .radius = 2.f, .material = red};
    spheres[1] = (Sphere){.position = {5, -5, 14}, .radius = 3.f, .material = blue};
    spheres[2] = (Sphere){.position = {0, 2, 20}, .radius = 5.f, .material = brown};
    spheres[3] = (Sphere){.position = {-2, -1, 8}, .radius = 1.f, .material = blue};
    spheres[4] = (Sphere){.position = {1, 2, 5}, .radius = 0.5f, .material = red};

    Color pixels[SCREEN_HEIGHT][SCREEN_WIDTH];

    Camera camera = {.position = {0, 0, 0},
                     .target = {0, 0, 10},
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

    while (!WindowShouldClose()) {
        BeginDrawing();

        for (int x = 0; x < SCREEN_WIDTH; x++) {
            for (int y = 0; y < SCREEN_HEIGHT; y++) {
                Ray ray = GetScreenToWorldRay((Vector2){x, y}, camera);

                Sphere nearest_sphere;
                RayCollision nearest_collision;
                nearest_collision.hit = false;
                nearest_collision.distance = FLT_MAX;
                for (int i = 0; i < SPHERE_COUNT; i++) {
                    Sphere sphere = spheres[i];

                    RayCollision collision =
                        GetRayCollisionSphere(ray, sphere.position, sphere.radius);

                    if (!collision.hit) {
                        continue;
                    }

                    if (collision.distance < nearest_collision.distance) {
                        nearest_collision = collision;
                        nearest_sphere = sphere;
                    }
                }

                if (!nearest_collision.hit) {
                    pixels[y][x] = BLACK;
                    continue;
                }

                SphereMaterial material = nearest_sphere.material;

                float ambient = 0.1f;

                float diffuse = fmax(Vector3DotProduct(nearest_collision.normal, light_dir), 0.0);

                Vector3 view_dir =
                    Vector3Normalize(Vector3Subtract(camera.position, nearest_collision.point));
                Vector3 reflect_dir =
                    Vector3Reflect(Vector3Negate(light_dir), nearest_collision.normal);

                float specular =
                    pow(fmax(Vector3DotProduct(view_dir, reflect_dir), 0.0), material.shininess) *
                    material.specular;

                float result = Clamp(ambient + diffuse + specular, 0, 1);

                pixels[y][x] = (Color){.r = material.color.r * (result),
                                       .g = material.color.g * (result),
                                       .b = material.color.b * (result),
                                       .a = material.color.a};
            }
        }

        UpdateTexture(texture, pixels);

        DrawTexture(texture, 0, 0, WHITE);
        EndDrawing();
    }

    CloseWindow();

    return 0;
}

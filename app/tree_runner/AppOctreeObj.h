#include "lib/core/TreeExplorerCpu.h"
#include "lib/tools/cuda/RandomPointsGenerator3d.cuh"
#include "render/base/RenderCamera.h"
#include "drawable/DrawableTree.h"
#include "lib/core/octree/OctreeBuilderCuda.cuh"

#include <iostream>

#define TINYOBJLOADER_IMPLEMENTATION
#include <thirdparty/tinyobjloader/tiny_obj_loader.h>

static bool LoadObjAndConvert(float3& bmin, float3& bmax,
                              std::vector<float3>& vertices,
                              const char* filename) {
  tinyobj::attrib_t inattrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;

  std::string warn;
  std::string err;
  bool ret = tinyobj::LoadObj(&inattrib, &shapes, &materials, &warn, &err, filename);
  if (!warn.empty()) 
  {
    std::cout << "WARN: " << warn << std::endl;
  }
  
  if (!err.empty()) 
  {
    std::cerr << err << std::endl;
  }

  if (!ret) 
  {
    std::cerr << "Failed to load " << filename << std::endl;
    return false;
  }

  printf("# of vertices  = %d\n", (int)(inattrib.vertices.size()) / 3);
  printf("# of normals   = %d\n", (int)(inattrib.normals.size()) / 3);
  printf("# of texcoords = %d\n", (int)(inattrib.texcoords.size()) / 2);
  printf("# of shapes    = %d\n", (int)shapes.size());

  bmin = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
  bmax = {std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), std::numeric_limits<float>::min()};

  for (size_t s = 0; s < shapes.size(); s++) 
  {
      size_t index_offset = 0;
      for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) 
      {
          size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

          if (shapes[s].mesh.num_face_vertices[f] != 3) 
          {
              index_offset += fv;
              continue;
          }

          for (size_t v = 0; v < 3; v++) 
          {
              tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
              const tinyobj::real_t vx = inattrib.vertices[3*idx.vertex_index+1];
              const tinyobj::real_t vy = inattrib.vertices[3*idx.vertex_index+2];
              const tinyobj::real_t vz = inattrib.vertices[3*idx.vertex_index+0];

              vertices.push_back({vx, vy, vz});
              bmin.x = std::min(vx, bmin.x);
              bmin.y = std::min(vy, bmin.y);
              bmin.z = std::min(vz, bmin.z);
              bmax.x = std::max(vx, bmax.x);
              bmax.y = std::max(vy, bmax.y);
              bmax.z = std::max(vz, bmax.z);
          }

          index_offset += fv;
      }
  }

  printf("bmin = %f, %f, %f\n", bmin.x, bmin.y, bmin.z);
  printf("bmax = %f, %f, %f\n", bmax.x, bmax.y, bmax.z);

  return true;
}

void RunAppOctreeObj(const TreeConfig& treeConfig, const RenderConfig& renderConfig, const AppConfig& appConfig)
{
  std::cout << "Load Model()\n";
  float3 bmin;
  float3 bmax;
  std::vector<float3> pointsVec;
  LoadObjAndConvert(bmin, bmax, pointsVec, appConfig.modelPath.c_str());

  TreeConfig treeConfigObj = treeConfig;
  treeConfigObj.origin = bmin;
  treeConfigObj.size = bmax - bmin;

  auto pointsCount = pointsVec.size();
  float3* pointsHost = pointsVec.data();
  std::cout << "RunAppOctree(): " << pointsCount << "\n";
  float3* points;
  hostToDevice(pointsHost, pointsCount, &points);
  
  auto treeBuilder = std::make_unique<OctreeBuilderCuda>(treeConfigObj);
  treeBuilder->Initialize(pointsCount);
  treeBuilder->Build(points, pointsCount);

  deviceToHost(points, pointsCount, &pointsHost);

  const auto& tree = treeBuilder->GetTree();

  if (!appConfig.enableRender)
  {
      std::cout << "SOME TEST: " << std::endl;

      std::cout << "1. kNN:" << std::endl;
      std::vector<Neighbour> result;
      result = NearestNeighbours<float3, 3>(&tree, treeConfigObj, {0.0f, 0.0f, 0.0f}, 15, pointsHost);
      std::cout << "Items found: " << result.size() << std::endl << std::endl;

      std::cout << "2. RadiusSearch:" << std::endl;
      result = RadiusSearch<float3, 3>(&tree, treeConfigObj, {0.0f, 0.0f, 0.0f}, 15, pointsHost);
      std::cout << "Items found: " << result.size() << std::endl;

      return;
  }

  RenderCamera render(renderConfig, true);

  auto drawableTree = std::make_unique<DrawableTree<float3, 3>>(&tree, pointsHost, treeConfigObj, 
      renderConfig.pointSize, renderConfig.lineWidth);
  render.AddDrawable(drawableTree.get());
  
  render.Initialize();
  render.StartLoop();
}
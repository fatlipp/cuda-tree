## CUDA based Tree builder (QuadTree, Octree)

Octree                                      | QuadTree
:------------------------------------------:|:---------------------------------------------:
![Octree](/assets/Octree1.png) | ![QuadTree](/assets/QuadTree.png)
![Octree](/assets/Skeleton.png) |

Functionality
* CUDA tree builder
* CPU kNN, Radius Search
* 2d, 3d visualization (controls: WSAD R F, Mouse). Additional functionality for QuadTree - click on a Tree for kNN and RadiusSearch
* Random Points Generator
* Configuration file
* An obj file loader

Some implementation details:
[Medium](https://medium.com/@fatlip/cuda-quadtree-octree-72e65216866c)

---
#### Build and Run
```
* mkdir build && cd build
* cmake .. `[-DCUDA_ARCH=SET_YOUR_ARCH]`
* make -j8
* `./cuda_tree_app_TYPE ./app/config/AppConfig.json`
* memory check: `compute-sanitizer --tool memcheck  ./cuda_tree_app_TYPE ./app/config/AppConfig.json`
```
* cuda_tree_app_TYPE:
    * cuda_tree_app_rng - to use random points
    * cuda_tree_app_obj - to read .obj file from `model_path` (only vertices)

#### Config
- `AppConfig.json` - contains general information (points count, pathes to Tree and Render config files)
- `TreeConfig.json`:
  * type - 'Quad' or 'Oct'
  * size - width, height, depth (if Octree)
  * threadsPerBlock - min **128** for QuadTree and **256** for Octree **(1 warp for each leaf)**
- `RenderConfig.json` - all about rendering

#### Requirements
* CUDA 11.8+
* C++ 17+
* OpenGL, GLFW
* nlohmann, tinyobjloader (included)


#### Limitations
* Tested on up to 150 mln points
* Max *stable* depth: QuadTree = 7, Octree = 6
* I recommend to disable rendering (`AppConfig.json`) if more than 50 mln points are used

#### Further development
1. CUDA based kNN, Radius Search
2. Refactoring, CPU, GPU optimization
3. OpenGL interop for faster visualization
4. Raycasting
5. KDTree

---

#### Disclaimer
* Tested on Linux: Ubuntu 22.02, CUDA: 12.2, GPU: 3060 (laptop)
* Approach based on Nvidia's QuadTree sample (cdpQuadtree)

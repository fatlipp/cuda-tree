## CUDA based Tree builder (QuadTree, Octree)

Octree                                      | QuadTree
:------------------------------------------:|:---------------------------------------------:
![Octree](/assets/Octree1.png){ width=70% } | ![QuadTree](/assets/QuadTree.png){ width=70% }

Functionality
* CUDA tree builder
* CPU kNN, Radius Search
* 2d, 3d visualization (controls: WSAD R F, Mouse). Additional functionality for QuadTree - click on a Tree for kNN and RadiusSearch
* Random Points Generator
* Configuration file

---

#### Usage
* `./cuda_tree_app ./app/config/AppConfig.json`
* memory check: `compute-sanitizer --tool memcheck  ./cuda_tree_app ./app/config/AppConfig.json`

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
* nlohmann (included)


#### Limitations
* Tested on up to 150 mln points
* Max *stable* depth: QuadTree = 7, Octree = 6
* I recommend to rendering if use more than 50 mln points

#### Further development
1. CUDA based kNN, Radius Search
2. Refactoring, CPU, GPU optimization
3. PointCloud loader
4. OpenGL interop for faster visualization
5. Raycasting
6. KDTree

---

#### Disclaimer
* Tested on Linux: Ubuntu 20.02, CUDA: 12.2, GPU: 3060 (laptop)
* Approach based on Nvidia's QuadTree sample (cdpQuadtree)
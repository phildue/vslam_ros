# TODOs

## Algorithm
- [ ] ~~Integrate kalman filter for motion prediction~~ --> only make sense with reliable uncertainty / better prior
- [x] Integrate alignment on depth error
- [x] Fix open problems in motion prior
- [x] Implement loop closure detection
- [x] Implement pose graph optimization

- [ ] Implement 3DMap (Fusion of depth maps)
  - [ ] Integrate tsdf fusion from open3D

- [ ] Implement Depth Mapping (Depth Filters)

- [ ] Implement local mapping (joint optimization of local keyframes)
  - [ ] Integrate descriptor based matching for key frames + geometric bundle adjustment --> No improvement so far
  - [ ] Implement joint photometric optimization --> No improvement so far

- [ ] Implement Localization
  - [ ] Implement landmark extraction
  - [ ] Implement online landmark extraction
  - [ ] Implement localization

## Architecture
- [ ] Common class for direct alignment with templated residual

## Framework
- [ ] Use own queue and monitoring task for replay start/stop

## Evaluation
- [x] Store results in database (wandb)
- [ ] Run on kitti dataset

## Hardware Integration
- [ ] ROS2 Driver for camera
- [ ] Runtime optimization

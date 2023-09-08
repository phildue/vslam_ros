# TODOs

## Algorithm
- [-] Integrate kalman filter for motion prediction --> only make sense with reliable uncertainty / better prior
- [x] Integrate alignment on depth error
- [x] Fix open problems in motion prior
- [x] Implement loop closure detection
- [ ] Implement pose graph optimization
- [ ] Integrate descriptor based matching for key frames
- [ ] Integrate bundle adjustment for pose graph optimization
- [ ] Implement 3D mapping that fuses depth, RGB based on trajectory (kinect fusion?)
- [ ] Implement landmark extraction
- [ ] Implement online landmark extraction
- [ ] Implement localization

## Architecture
- [ ] Common interface for aligners

## Framework
- [ ] Use own queue and monitoring task for replay start/stop

## Evaluation
- [x] Store results in database (wandb)
- [ ] Run on kitti dataset

## Hardware Integration
- [ ] ROS2 Driver for camera
- [ ] Runtime optimization

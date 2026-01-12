# Glue Width Detection System Using Computer Vision

## Technology Stack

### Hardware
- **Industrial Camera:** High-resolution camera for image acquisition
- **FANUC Robot System:** Robot manipulator with end-effector mounting
- **Lighting System:** Controlled illumination for consistent imaging
- **Calibration Board:** Chessboard pattern for camera calibration

### Software & Frameworks
- **HALCON:** Industrial machine vision software for image processing
- **OpenCV:** Open-source computer vision library
- **MATLAB:** Numerical computing environment for calibration and analysis
- **Python:** Primary programming language for algorithm implementation

### Computer Vision Libraries & Tools
- **HALCON Operators:** Image preprocessing, edge detection, skeleton extraction, blob analysis
- **OpenCV Functions:**
  - Feature detection (SIFT, FLANN)
  - Camera calibration (calibrateCamera, findChessboardCorners, cornerSubPix)
  - Homography estimation (findHomography, getPerspectiveTransform)
  - Image transformation (warpPerspective, Rodrigues)
- **MATLAB Toolboxes:**
  - Computer Vision Toolbox for camera calibration
  - Peter Corke Robotics Toolbox for hand-eye calibration
  - Image Processing Toolbox

### Algorithms & Methods
- **Zhang's Calibration Method:** Camera intrinsic and extrinsic parameter estimation
- **RANSAC:** Robust homography estimation and outlier rejection
- **Canny Edge Detection:** Precise edge localization
- **Skeleton Extraction:** Centerline extraction for width measurement
- **DLT (Direct Linear Transform):** Homography matrix computation
- **SVD (Singular Value Decomposition):** Least squares solution for calibration

---

##

Due to company confidentiality policies, the complete implementation code and proprietary algorithms cannot be publicly released. This README provides a comprehensive technical description of the methodology, system architecture, and results.

**For technical discussions or collaboration inquiries, please contact via email.**

---

## Project Overview

This project develops an automated glue width detection system for industrial applications, combining computer vision, robot manipulation, and real-time image processing to ensure product quality and consistency in adhesive application processes.

### Project Duration
October 22, 2024 - December 20, 2024 (2 months)

### Department
Intelligent Technology Development Department

---

## Project Objectives

1. **Automated Quality Control:** Develop a non-contact, automated system for real-time glue width measurement
2. **Real-time Processing:** Achieve processing speeds compatible with production line requirements (≤125ms per frame for 400mm/s application speed at 8fps)
3. **High Precision:** Ensure accurate width measurement across straight and curved sections
4. **Robustness:** Maintain detection accuracy under varying environmental conditions
5. **Trajectory Reconstruction:** Reconstruct complete glue application trajectory from sequential images

---

## Background & Motivation

### Industrial Requirements

In modern manufacturing, adhesive application quality directly impacts product reliability and performance. Traditional manual inspection methods suffer from:
- **Subjectivity:** Human judgment varies between operators
- **Low Efficiency:** Time-consuming inspection processes
- **Limited Coverage:** Cannot inspect 100% of production
- **Delayed Detection:** Quality issues discovered after application

### Technical Challenges

**Real-time Processing Requirement:**
- Application speed: 400mm/s
- Required frame rate: 8fps
- Maximum processing time: ≤125ms per frame
- Goal: Real-time inline detection (inspection while applying)

**Environmental Factors:**
- Dynamic background during robot movement
- Lighting variations
- Dust and environmental interference
- Glue flow dynamics and surface irregularities

---

## Learning Process & Knowledge Acquisition

### 3.1 Professional Books & Papers

**Core References:**

1. **Introduction to Robotics (John Craig)**
   - Robot design fundamentals
   - Kinematics and dynamics
   - Control theory

2. **14 Lectures on Visual SLAM (Gao Xiang)**
   - Camera models and calibration
   - SLAM algorithms
   - Visual odometry

3. **HALCON Digital Image Processing (Liu Guohua)**
   - Image digitization and compression
   - Template matching and blob analysis
   - OCR and dimension measurement

4. **HALCON Operator Reference (Official Documentation)**
   - Detailed operator specifications
   - Parameter meanings and usage

5. **Research Papers:**
   - Classical papers on homography-based motion estimation
   - "IHUVS: Infinite Homography-Based Uncalibrated Methodology for Robotic Visual Servoing"
   - Papers on camera calibration and pose estimation

### 3.2 Online Courses

**Robotics Courses:**

1. **Modern Robotics (Northwestern University)**
   - Configuration space
   - Forward and inverse kinematics
   - Velocity kinematics and statics
   - Trajectory generation
   - Motion planning and control

2. **Introduction to Robotics (MIT 2.12)**
   - Actuators and drive systems
   - Robot structures
   - Planar kinematics and statics

3. **Robotic Manipulation (MIT 6.4210)**
   - Pick-and-place operations
   - Geometric pose estimation

**Computer Vision Courses:**

4. **Computer Vision (CMU 16-385, Spring 2020)**
   - Image filtering
   - Feature extraction and corner detection
   - Feature descriptors and matching
   - Image homography
   - 2D transformations and camera models

5. **MATLAB Fundamentals Review**
   - Matrix operations and manipulations
   - Plotting functions (plot, fplot, ezplot)
   - 3D visualization (mesh, surf)

### 3.3 Technical Forums & Documentation

**Key Learning Resources:**

1. **OpenCV Documentation**
   - `findHomography()` function principles
   - Homogeneous coordinate transformation
   - Applications in image stitching and SLAM

2. **Zhang's Calibration Method**
   - "A Flexible New Technique for Camera Calibration"
   - Least squares and SVD decomposition
   - Intrinsic and extrinsic parameter estimation

3. **RANSAC for Image Stitching**
   - Feature detection using SIFT
   - Feature matching with FLANN
   - Homography estimation and image warping

4. **Coordinate Transformations**
   - Euler angles, rotation vectors, rotation matrices
   - Conversion relationships in OpenCV
   - Rodrigues formula applications

5. **Homography Decomposition**
   - `decomposeHomography` implementation
   - Extracting rotation and translation from homography
   - Applications in pose estimation

---

## System Architecture

### Image Processing Pipeline

```
Camera Image Acquisition
         ↓
Image Preprocessing
    ↓
Grayscale Conversion
    ↓
Noise Reduction
    ↓
Edge Detection (Canny)
    ↓
Skeleton Extraction
    ↓
Width Measurement
```

### Coordinate Systems

**Multiple Reference Frames:**
1. **World Coordinate System:** Fixed reference frame
2. **Robot Base Coordinate System:** Robot manipulator base
3. **Camera Coordinate System:** Camera optical center
4. **Image Coordinate System:** 2D image plane
5. **Calibration Board Coordinate System:** Chessboard pattern reference

---

## Implementation Approaches

### Approach A: Direct HALCON-Based Detection

#### Methodology

**Process Flow:**
1. **Image Acquisition:** Capture overhead images of glue application
2. **Image Preprocessing:**
   - Convert RGB to grayscale using `rgb1_to_gray`
   - Apply threshold using `threshold` operator
3. **Edge Detection:**
   - Apply Canny edge detector
   - Extract precise boundaries
4. **Skeleton Extraction:**
   - Use `skeleton` operator for centerline extraction
   - Apply `split_skeleton_lines` to remove branches
   - Convert to XLD format using `gen_contours_skeleton_xld`
5. **Width Calculation:**
   - Measure perpendicular distances from skeleton
   - Calculate statistical width metrics

#### Results

**Single Image Detection:**
- ✅ Excellent detection accuracy on static images
- ✅ Precise edge detection with Canny operator
- ✅ Good skeleton extraction in straight sections
- ✅ Improved quality with smoothing operators

**Continuous Image Sequence:**
- ❌ Processing speed insufficient for real-time requirements
- ❌ Current processing time: >125ms per frame
- ❌ Dynamic background introduces additional interference
- ❌ Cannot achieve inline real-time detection

#### Limitations Analysis

**Speed Constraint:**
```
Required: 400mm/s application speed at 8fps
Required processing time: ≤125ms per frame
Current performance: >125ms per frame
Gap: Cannot meet real-time inline inspection requirement
```

**Environmental Challenges:**
- Dynamic background during robot motion
- Dust and water droplets causing false detections
- Variable lighting conditions
- Glue surface reflections and irregularities

**Quality Requirements:**
- High precision measurement (sub-millimeter accuracy)
- Robustness to environmental variations
- 100% detection coverage
- Zero false negatives (missed defects)

---

### Approach B: Feature Matching-Based Homography Estimation

#### Methodology

**Concept:**
Estimate camera trajectory by computing homography matrices between consecutive frames using feature point matching.

**Process Flow:**

1. **Feature Detection:**
   - Extract SIFT (Scale-Invariant Feature Transform) keypoints
   - Generate feature descriptors for each keypoint

2. **Feature Matching:**
   - Use FLANN (Fast Library for Approximate Nearest Neighbors)
   - Apply KNN matching with distance ratio test
   - Filter matches to retain high-quality correspondences

3. **Homography Estimation:**
   - Apply RANSAC algorithm to robustly estimate homography
   - Separate inliers from outliers
   - Compute 3×3 homography matrix H

4. **Trajectory Reconstruction:**
   - Project brush position from frame n to frame n+1
   - Accumulate transformations across image sequence
   - Generate complete glue application skeleton

**Mathematical Framework:**

```
Homography Transformation:
[x']   [h11 h12 h13]   [x]
[y'] = [h21 h22 h23] × [y]
[w']   [h31 h32 h33]   [1]

where: x' = x'/w', y' = y'/w'
```

**RANSAC Algorithm:**
1. Randomly select minimum point set (4 points)
2. Compute homography hypothesis
3. Evaluate inliers using reprojection error threshold
4. Iterate for maximum inlier consensus
5. Recompute final homography from all inliers

#### Results & Analysis

**Why This Approach Failed:**

1. **Dynamic Environment:**
   - Glue application is a continuous dynamic process
   - Background changes significantly between frames
   - SIFT/FLANN designed for static scenes
   - Unstable feature detection in moving scenarios

2. **Insufficient Feature Points:**
   - Glue region lacks distinctive texture
   - Dynamic background provides unstable features
   - Feature repeatability decreases across frames
   - Matching accuracy degraded by motion blur

3. **Environmental Interference:**
   - Moving objects introduce spurious features
   - Changing textures confuse matching algorithms
   - Increased image noise from dynamic background
   - False matches from environmental elements

4. **Domain Mismatch:**
   - Feature-based methods excel in scene reconstruction
   - Not optimized for trajectory tracking in industrial applications
   - Requires stable, textured environments
   - Cannot handle featureless regions (uniform glue surface)

#### Comparison: findHomography vs getPerspectiveTransform

**Similarities:**
- Both compute homography matrices
- Both solve linear equation systems
- Minimum 4 point correspondences required

**Differences:**

| Aspect | getPerspectiveTransform | findHomography |
|--------|------------------------|----------------|
| **Input Points** | Exactly 4 points | ≥4 points |
| **Method** | SVD decomposition | DLT + RANSAC/LMEDS |
| **Robustness** | No outlier rejection | Outlier filtering |
| **Optimization** | Direct solution | Iterative refinement |
| **Use Case** | Known good points | Noisy measurements |

---

### Approach C: Calibration-Based Homography Computation

#### Methodology

**Concept:**
Compute homography matrices through explicit camera calibration, obtaining intrinsic and extrinsic parameters separately, then reconstructing homography for trajectory estimation.

**Theoretical Foundation:**

**Homography Decomposition:**
```
H = K [r1 r2 t]

where:
K = Camera intrinsic matrix (3×3)
[r1 r2] = First two columns of rotation matrix R
t = Translation vector
```

**Process Flow:**

1. **Chessboard Corner Detection:**
   ```python
   ret, corners = cv2.findChessboardCorners(gray, pattern_size)
   corners_refined = cv2.cornerSubPix(gray, corners, 
                                       winSize=(11,11), 
                                       zeroZone=(-1,-1), 
                                       criteria)
   ```

2. **Camera Calibration:**
   ```python
   ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
       object_points,    # 3D points in world coordinates
       image_points,     # 2D points in image coordinates
       image_size,
       None, None
   )
   ```
   
   **Outputs:**
   - `mtx`: Intrinsic matrix K
   - `dist`: Distortion coefficients
   - `rvecs`: Rotation vectors for each pose
   - `tvecs`: Translation vectors for each pose

3. **Rotation Matrix Conversion:**
   ```python
   R, _ = cv2.Rodrigues(rvec)
   ```

4. **Homography Computation Between Poses:**
   
   For two camera poses observing the same plane:
   ```
   H₁ = K [R₁ | t₁]
   H₂ = K [R₂ | t₂]
   
   H₁→₂ = H₂ × H₁⁻¹
   ```

5. **Trajectory Reconstruction:**
   - Project brush position through homography chain
   - Accumulate transformations across sequence
   - Generate complete application skeleton

#### Mathematical Details

**Camera Model:**
```
[u]   [fx  0  cx]   [X]
[v] = [ 0 fy  cy] × [Y]
[1]   [ 0  0   1]   [Z]
```

**Pose Transformation:**
```
P_camera = R × P_world + t
```

**Homography for Planar Scenes:**
```
Given: Plane π with normal n and distance d
H = K × (R - (t×nᵀ)/d) × K⁻¹
```

#### Implementation Steps

**Step 1: Corner Detection & Refinement**
```python
# Detect chessboard corners
ret, corners = cv2.findChessboardCorners(image, (rows, cols))

# Subpixel refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
            30, 0.001)
corners_refined = cv2.cornerSubPix(gray, corners, (11,11), 
                                    (-1,-1), criteria)
```

**Step 2: Multi-Pose Calibration**
```python
# Collect multiple views
for image in calibration_images:
    corners = detect_and_refine_corners(image)
    object_points.append(board_3d_points)
    image_points.append(corners)

# Calibrate camera
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    object_points, image_points, image_size, None, None)
```

**Step 3: Homography Computation**
```python
# Convert rotation vectors to matrices
R1, _ = cv2.Rodrigues(rvecs[pose1])
R2, _ = cv2.Rodrigues(rvecs[pose2])

# Compute homography from pose1 to pose2
# Assume plane normal n = [0, 0, 1] (board plane)
d1 = tvecs[pose1][2]  # Distance to plane
d2 = tvecs[pose2][2]

H1 = K @ np.hstack([R1[:, :2], tvecs[pose1]])
H2 = K @ np.hstack([R2[:, :2], tvecs[pose2]])

H_12 = H2 @ np.linalg.inv(H1)
```

**Step 4: Point Projection**
```python
# Project brush position from pose1 to pose2
brush_pos_homogeneous = np.array([x, y, 1])
projected_pos = H_12 @ brush_pos_homogeneous
projected_x = projected_pos[0] / projected_pos[2]
projected_y = projected_pos[1] / projected_pos[2]
```

#### Results

**Achievements:**
- ✅ Successful camera calibration with accurate intrinsic parameters
- ✅ Precise extrinsic parameter estimation for multiple poses
- ✅ Homography matrix computation between camera poses
- ✅ Corner projection validation showing transformation accuracy

**Visualization Results:**
- Displayed calibration board corners with refined positions
- Showed homography-transformed corners overlaid on target images
- Validated transformation accuracy through visual inspection

**Accuracy Analysis:**
- Small discrepancies observed between transformed and actual corners
- Requires coefficient adjustment for higher precision
- Error sources: calibration accuracy, plane assumption, numerical precision

**Progress Status:**
- ✅ Phase 1: Camera calibration - **Complete**
- ✅ Phase 2: Homography computation - **Complete**
- ✅ Phase 3: Corner transformation validation - **Complete**
- 🔄 Phase 4: Brush trajectory reconstruction - **In Progress**

#### MATLAB Verification

**Hand-Eye Calibration Validation:**

To verify homography accuracy, parallel hand-eye calibration was performed in MATLAB:

**Process:**

1. **Extrinsic Parameter Calculation:**
   - Computed external parameters for 16 different poses
   - Generated transformation matrices for each configuration

2. **Robotics Toolbox Integration:**
   - Imported Peter Corke Robotics Toolbox
   - Used `rpy2r` for roll-pitch-yaw to rotation matrix conversion
   - Applied spatial transformation utilities

3. **Hand-Eye Relationship Estimation:**
   ```matlab
   % Estimate camera intrinsics
   intrinsics = estimateCameraIntrinsics(params);
   
   % Collect hand-eye calibration images
   [images, robotPoses] = collectCalibrationData();
   
   % Estimate camera extrinsics
   [R_cam, t_cam] = estimateCameraExtrinsics(images, intrinsics);
   
   % Compute robot end-effector to base transformation
   T_base_ee = computeRobotFK(robotPoses);
   
   % Estimate camera to end-effector transformation
   [R_ee_cam, t_ee_cam] = estimateHandEyeTransformation(
       T_base_ee, R_cam, t_cam);
   ```

4. **Spatial Relationship Results:**
   - Successfully obtained intrinsic parameters
   - Computed extrinsic parameters for all poses
   - Generated rotation matrices for hand-eye relationship
   - Validated consistency with OpenCV homography results

**Verification Outcome:**
- Cross-validation between MATLAB and OpenCV implementations
- Confirmed accuracy of homography computation approach
- Spatial transformation consistency verified across methods

---

## Applications of Homography Matrix

### 1. Image Stitching
Homography transforms images from different viewpoints to a common plane, enabling seamless panoramic image creation.

### 2. Image Rectification
Corrects distortions caused by camera perspective or lens aberrations, restoring geometric accuracy.

### 3. Object Recognition & Localization
Maps objects to a standard plane for consistent recognition and dimensional measurement.

### 4. 3D Reconstruction
Recovers 3D spatial information from 2D images through multi-view geometry analysis.

### 5. Pose Estimation
Determines camera position and orientation relative to known reference planes or objects.

### 6. Visual Servoing
Provides transformation relationship for robot vision-based control and trajectory planning.

---

## Key Technical Concepts

### Camera Calibration Pipeline

```
Image Capture
    ↓
Corner Detection (findChessboardCorners)
    ↓
Subpixel Refinement (cornerSubPix)
    ↓
Calibration (calibrateCamera)
    ↓
Outputs: K (intrinsics), dist, R, t
    ↓
Undistortion & Pose Estimation
```

### Homography Transformation Chain

For a sequence of images i=1 to N:
```
Pose 1 → Homography H₁₂ → Pose 2 → Homography H₂₃ → Pose 3 → ...
```

Brush trajectory reconstruction:
```
Brush₁ → H₁₂(Brush₁) → Brush₂ → H₂₃(Brush₂) → Brush₃ → ...
```

Complete skeleton:
```
Skeleton = Union of all brush positions across frames
```

### Coordinate Transformation Relationships

**Euler Angles ↔ Rotation Matrix:**
```python
# Euler to Rotation Matrix
R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

# Rotation Matrix to Euler
yaw, pitch, roll = decompose_rotation_matrix(R)
```

**Rotation Vector ↔ Rotation Matrix (Rodrigues):**
```python
# Rotation vector to matrix
R, _ = cv2.Rodrigues(rvec)

# Rotation matrix to vector
rvec, _ = cv2.Rodrigues(R)
```

**Homography Decomposition:**
```python
# Decompose homography into R, t, n
retval, Rs, ts, normals = cv2.decomposeHomography(H, K)
```

---

## Technical Challenges & Solutions

### Challenge 1: Real-time Processing Speed

**Problem:**
- Required: ≤125ms per frame @ 8fps
- Current: >125ms per frame
- Cannot achieve inline inspection during application

**Attempted Solutions:**
- Algorithm optimization (skeleton extraction efficiency)
- Hardware acceleration considerations
- Parallel processing architecture

**Future Directions:**
- GPU acceleration for image processing
- Optimized HALCON operator sequences
- Hardware upgrade to faster processing units

### Challenge 2: Dynamic Background Interference

**Problem:**
- Moving robot creates changing background
- Environmental elements (dust, reflections) cause false detections
- Feature instability in dynamic scenes

**Solutions Explored:**
- Calibration-based approach (Approach C) eliminates feature matching dependency
- Background subtraction techniques
- Temporal filtering across frames

### Challenge 3: Homography Accuracy

**Problem:**
- Small discrepancies between projected and actual positions
- Accumulation of errors across frame sequence
- Plane assumption deviations

**Solutions:**
- Subpixel corner refinement for higher accuracy
- Multiple calibration images for robust parameter estimation
- Error compensation and coefficient adjustment
- Cross-validation with MATLAB hand-eye calibration

### Challenge 4: Skeleton Extraction in Curved Regions

**Problem:**
- Good skeleton extraction in straight sections
- Degraded performance in highly curved areas
- Branch points causing discontinuities

**Solutions:**
- Smoothing operators before skeletonization
- Branch removal algorithms (`split_skeleton_lines`)
- Curve-aware skeleton extraction methods

---

## Performance Metrics

### Detection Accuracy
- **Straight Sections:** Excellent (>95% accuracy)
- **Curved Sections:** Good (>85% accuracy in center regions)
- **Edge Regions:** Moderate (requires environmental interference mitigation)

### Processing Speed
- **Single Frame (Approach A):** >125ms (insufficient for real-time)
- **Target Performance:** ≤125ms per frame
- **Required Improvement:** ~20-30% speed increase needed

### Calibration Accuracy
- **Intrinsic Parameters:** High precision with low reprojection error
- **Extrinsic Parameters:** Consistent across multiple poses
- **Homography Transformation:** Small pixel-level discrepancies requiring refinement

---

## Future Work & Optimization

### 1. Complete Trajectory Reconstruction (Approach C)
- **Current Status:** Homography computation complete, trajectory reconstruction in progress
- **Next Steps:**
  - Implement brush position tracking across frame sequence
  - Accumulate homography transformations
  - Generate complete glue application skeleton
  - Validate against ground truth measurements

### 2. Real-time Performance Optimization
- **Algorithm Optimization:**
  - Streamline HALCON operator sequences
  - Implement region-of-interest (ROI) processing
  - Multi-threaded image processing pipeline
- **Hardware Acceleration:**
  - GPU-accelerated image processing
  - Dedicated vision processing hardware
  - Optimized memory management

### 3. Robustness Enhancement
- **Environmental Adaptation:**
  - Adaptive lighting compensation
  - Dynamic background subtraction
  - Multi-frame temporal filtering
- **Error Handling:**
  - Outlier detection and rejection
  - Automatic quality assessment
  - Fallback detection strategies

### 4. System Integration
- **Robot Communication:**
  - Real-time feedback to robot controller
  - Synchronization with robot motion
  - Adaptive control based on detection results
- **Production Line Integration:**
  - Integration with MES (Manufacturing Execution System)
  - Automated defect reporting
  - Statistical process control (SPC)

### 5. Advanced Features
- **3D Width Measurement:**
  - Stereo vision or structured light integration
  - Full 3D glue profile reconstruction
- **Predictive Analytics:**
  - Machine learning for defect prediction
  - Process parameter optimization
- **Multi-sensor Fusion:**
  - Combine vision with other sensors (laser, ultrasonic)
  - Enhanced measurement reliability

---

## Learning Outcomes

### Technical Skills Acquired

1. **Industrial Machine Vision:**
   - HALCON software proficiency
   - Image preprocessing and enhancement
   - Feature extraction and measurement
   - Skeleton extraction algorithms

2. **Computer Vision:**
   - OpenCV library mastery
   - Camera calibration theory and practice
   - Homography estimation techniques
   - Feature detection and matching

3. **Robot Vision Systems:**
   - Hand-eye calibration methodology
   - Coordinate system transformations
   - Visual servoing concepts
   - Trajectory planning and control

4. **Mathematical Foundations:**
   - Linear algebra (matrices, SVD)
   - Optimization algorithms (RANSAC, least squares)
   - Geometric transformations
   - Numerical methods

5. **Software Development:**
   - Python for computer vision
   - MATLAB for robotics analysis
   - Algorithm design and optimization
   - Debugging and validation techniques

### Problem-Solving Abilities

- **Systematic Approach:** Breaking complex problems into manageable components
- **Research Skills:** Efficiently finding and applying academic literature
- **Iterative Development:** Testing hypotheses and refining solutions
- **Critical Analysis:** Evaluating approach suitability and limitations
- **Adaptation:** Pivoting strategies when methods prove inadequate

### Professional Development

- **Industry Knowledge:** Understanding manufacturing quality control requirements
- **Team Collaboration:** Effective communication with supervisors and colleagues
- **Documentation:** Creating comprehensive technical reports
- **Time Management:** Balancing learning, experimentation, and deliverables
- **Self-Directed Learning:** Proactively acquiring necessary knowledge

---

## Conclusion

This internship provided invaluable hands-on experience in applying computer vision and robotics theory to real-world industrial applications. Through the development of three distinct approaches to glue width detection, key insights were gained:

### Key Achievements

1. **Multiple Solution Strategies:** Explored direct image processing, feature-based, and calibration-based approaches
2. **Technical Depth:** Mastered camera calibration, homography estimation, and coordinate transformations
3. **Practical Understanding:** Recognized real-world constraints (speed, robustness, accuracy trade-offs)
4. **Tool Proficiency:** Gained expertise in HALCON, OpenCV, and MATLAB

### Lessons Learned

- **Method Selection Matters:** Algorithm suitability depends on specific application constraints
- **Environmental Factors:** Real-world conditions significantly impact theoretical approaches
- **Iterative Refinement:** Complex problems require multiple attempts and continuous improvement
- **Cross-Validation:** Multiple tools and methods provide confidence in results

### Future Applications

The knowledge and skills acquired during this internship will be invaluable for:
- Academic research in computer vision and robotics
- Future industrial projects requiring vision systems
- Advanced study in robot perception and control
- Career development in automation and intelligent systems

---

## Acknowledgments

Special thanks to the supervisors and team members at the Intelligent Technology Development Department for their patient guidance, professional mentorship, and creating an excellent learning environment throughout this internship.

---

## References

### Books
- John Craig, "Introduction to Robotics: Mechanics and Control"
- Gao Xiang, "14 Lectures on Visual SLAM"
- Liu Guohua, "HALCON Digital Image Processing"

### Papers
- Z. Zhang, "A Flexible New Technique for Camera Calibration," IEEE TPAMI, 2000
- "IHUVS: Infinite Homography-Based Uncalibrated Methodology for Robotic Visual Servoing"

### Online Courses
- Modern Robotics (Northwestern University)
- MIT 2.12 Introduction to Robotics
- MIT 6.4210 Robotic Manipulation
- CMU 16-385 Computer Vision

### Documentation
- HALCON Operator Reference Manual
- OpenCV Documentation
- MATLAB Computer Vision Toolbox
- Peter Corke Robotics Toolbox

---

## License

This project was developed as part of an industry internship. The methodology and technical description are shared for educational purposes. Implementation details remain confidential per company policies.

**For technical discussions or collaboration inquiries, please contact via email.**

---

**Project Status:** Research and Development Phase  
**Implementation Status:** Approach C (Calibration-based) - Homography computation complete, trajectory reconstruction in progress  
**Last Updated:** December 2024

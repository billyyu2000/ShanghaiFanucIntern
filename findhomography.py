import numpy as np
import cv2
import glob

# ...（你之前的代码）...

# Load the two images for homography estimation
img1 = cv2.imread('../data/left00.jpg')
img2 = cv2.imread('../data/left01.jpg')

# Convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Find keypoints and descriptors in both images
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Match descriptors
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Extract location of good matches
points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Calculate homography
H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)

# Check if H is None
if H is not None:
    # Use the homography matrix to warp img1 to img2
    h, w = img2.shape[:2]
    img11 = cv2.warpPerspective(img1, H, (w, h))

    # Display the results
    cv2.imshow('img1', img11)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Homography calculation failed.")

img = cv2.imread("left01.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
ret, corners = cv2.findChessboardCorners(gray, (12 ,13), None)
corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

x, y, w, h, r, c = 15, 40, 38, 38, 12, 13
pts1 = np.int32(corners2.squeeze())
arr2 = np.tile(np.arange(c), r).reshape((r, c))
arr1 = np.tile(np.arange(r), c).reshape((c, r))
arr = np.dstack((arr1[:, ::-1] * h + y, arr2.T * w + x))
pts2 = arr.reshape((r * c, 2))

cv2.imshow("result", warp(img, np.zeros_like(img), pts1, pts2))
cv2.waitKey(0)
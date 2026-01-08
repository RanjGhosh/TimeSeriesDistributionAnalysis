import cv2
import numpy as np
from skimage import morphology, measure
import matplotlib.pyplot as plt

# Load the image
filename = "image_for_analysis_2.tif"
frame = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# Check if the file was loaded correctly
if frame is None:
    raise FileNotFoundError(f"Error: Could not load {filename}. Check file path.")

# Display the original image
plt.figure()
plt.imshow(frame, cmap='gray')
plt.title("Original Image")
plt.axis('on')
plt.show()

# Define scale
scale = 118.54  # px/mm

# Convert image to float
frame = frame.astype(np.float64) / 255.0

# Create structuring element
SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Morphological gradient
morphological_gradient = cv2.dilate(frame, SE) - cv2.erode(frame, SE)

# Thresholding
_, mask = cv2.threshold(morphological_gradient, 0.05, 1, cv2.THRESH_BINARY)
#0.05
plt.figure()
plt.imshow(mask, cmap='gray')
plt.title("mask_1")
plt.axis('on')
plt.show()
SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))

# Morphological closing
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, SE)

# Fill holes
mask = morphology.remove_small_holes(mask.astype(bool), area_threshold=100000).astype(np.uint8)

# Keep the largest connected component
labels = measure.label(mask)
mask = morphology.remove_small_objects(labels, min_size=5000).astype(np.uint8)
mask=mask>0
plt.figure()
plt.imshow(mask, cmap='gray')
plt.title("mask_2")
plt.axis('on')
plt.show()
# Invert mask and apply area filter
not_mask = np.logical_not(mask)
#mask = np.logical_or(mask, morphology.remove_small_objects(not_mask, min_size=5000).astype(np.uint8))
#print(mask[1,1])
plt.figure()
plt.imshow(not_mask, cmap='gray')
plt.title("not_mask")
plt.axis('on')
plt.show()

plt.figure()
plt.imshow(mask, cmap='gray')
plt.title("final mask")
plt.axis('on')
plt.show()

# Apply mask to the frame
frame = frame * mask
frame = frame > 0.35
#255

# Find coordinates of the thresholded frame
rw, cl = np.where(frame)
x_adj = cl - np.mean(cl)
y_adj = rw - np.mean(rw)

# Calculate the radius
r = np.sqrt(x_adj**2 + y_adj**2)

plt.figure()
plt.imshow(frame, cmap='gray')
plt.title("Final Image")
plt.axis('on')
plt.show()

# Plot histogram with axes

plt.figure()
plt.hist(r / scale, bins=100, density=True, histtype='step', linewidth=2)
plt.xlabel("Radius (mm)")
plt.ylabel("Density")
plt.title("Histogram of Extracted Radii")
plt.grid(True)
plt.show()
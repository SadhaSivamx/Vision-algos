'''
We process the video as a temporal series of frames
with dimensions (N,rows,cols).

For every individual pixel coordinate, we model the
intensity distribution over time by fitting a
Gaussian Mixture Model (GMM) with three components (K=3).

The Core Hypotheses:

1) Background (BG): These pixels are persistent and stable.
Therefore, the background component will have a
high weight (frequency) and a low spread/variance (stability).

2) Foreground (FG): These pixels represent transient changes
or moving objects. Their distributions typically exhibit lower
weights and higher spreads due to their inconsistent nature.

Classification :

a) Identify the BG Gaussian: For each pixel, we evaluate the three
fitted Gaussians and select the one that maximizes the
ratio weight/spread. This component represents our most reliable
background model, providing us with its specific
mean and standard deviation.

b) To classify the current frame, we determine if the observed
pixel value falls within a statistical "confidence interval"
defined by T standard deviations (where T is our threshold).

c) Decision :
If | pixelvalue - meanbg | > T * stdbg -> FG
else its -> BG
'''

import cv2 as cv
import numpy as np
from sklearn.mixture import GaussianMixture
import sys

# Load video file
videofile = "Shrimp.mp4"
cap = cv.VideoCapture(videofile)
frames = []
originalresized = []

print("Step 1: Collecting and Smashing Frames...")
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    # Smash dimensions to 1/4 size
    h, w, _ = frame.shape
    newsize = (w // 4, h // 4)
    framesm = cv.resize(frame, newsize)

    # Store grayscale for GMM and color for final stack
    grayframe = cv.cvtColor(framesm, cv.COLOR_BGR2GRAY)
    frames.append(grayframe)
    originalresized.append(framesm)

cap.release()

# Convert frame list to temporal array
videoarray = np.array(frames)
N, rows, cols = videoarray.shape
# Flatten spatial dims to get a timeline per pixel
pixelsequences = videoarray.reshape(N, -1).T
totalpixels = rows * cols

# Mask storage for all frames
allmasks = np.zeros((N, rows, cols), dtype=np.uint8)

print(f"Step 2: Fitting GMM for {totalpixels} pixels...")

# Core statistical loop
for i in range(totalpixels):
    # Extract historical intensity for a single pixel
    pixelhistory = pixelsequences[i].reshape(-1, 1)

    # Model the pixel using 3 Gaussians
    gmm = GaussianMixture(n_components=3, covariance_type='spherical', reg_covar=1e-2)
    gmm.fit(pixelhistory)

    # Get GMM parameters
    weights = gmm.weights_
    variances = gmm.covariances_
    means = gmm.means_.flatten()

    # Define background component using weight-to-spread ratio
    bgidx = np.argmax(weights / (variances + 1e-6))
    bgmean = means[bgidx]
    bgstd = np.sqrt(variances[bgidx])

    # Classify foreground points based on standard deviation distance
    diffs = np.abs(pixelsequences[i] - bgmean)
    isfg = diffs > (4.5 * bgstd)

    # Map results back to 2D image coordinates
    r, c = divmod(i, cols)
    allmasks[:, r, c] = isfg.astype(np.uint8) * 255

    # Display progress
    if i % 500 == 0 or i == totalpixels - 1:
        percentdone = (i / totalpixels) * 100
        sys.stdout.write(f"\r{percentdone:.1f}% done")
        sys.stdout.flush()

print("\nStep 3: Vertical Stacking and Saving Output.mp4...")

# Setup video writer for vertical output
fourcc = cv.VideoWriter_fourcc(*'mp4v')
mergedvideoout = cv.VideoWriter("Output.mp4", fourcc, 20.0, (cols, rows * 2))

for t in range(N):
    # Smooth the binary mask to remove salt-and-pepper noise
    maskclean = cv.medianBlur(allmasks[t], 5)
    maskbgr = cv.cvtColor(maskclean, cv.COLOR_GRAY2BGR)

    # Combine original frame and binary change mask vertically
    stackedrect = np.vstack((originalresized[t], maskbgr))

    mergedvideoout.write(stackedrect)

    if cv.waitKey(10) == ord('q'):
        break

mergedvideoout.release()
cv.destroyAllWindows()
print("Success! Processed video saved as Output.mp4")

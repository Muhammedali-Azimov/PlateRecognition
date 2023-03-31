import cv2
import imutils
import pytesseract

# Load the image
image = cv2.imread('plates/car3.jpg')

# Preprocess the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 30, 200)
cv2.imshow('Original Image', gray)
cv2.waitKey(0)
cv2.imshow('Original Image', edged)
cv2.waitKey(0)

# Find contours in the image
contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

# Find the largest contour (which should correspond to the license plate)
largest_contour = max(contours, key=cv2.contourArea)

# Draw the largest contour on the original image for visualization purposes
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# Display the image with contours
cv2.imshow('Image with Contours', image)
cv2.waitKey(0)

# Loop through the contours and find the license plate
for c in contours:
    # Approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    
    # Check if the contour has four points
    if len(approx) == 4:
        # Get the ROI and apply OCR
        (x, y, w, h) = cv2.boundingRect(approx)
        roi = gray[y:y + h, x:x + w]
        text = pytesseract.image_to_string(roi, config='--psm 11')
        
        # Print the license plate number
        print("License Plate Number: ", text)
        break
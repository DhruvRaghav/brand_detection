import os
import pandas as pd
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# Define the transformations to apply
transform = A.Compose([
    A.Resize(width=512, height=512),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=20, p=0.5),
    ToTensorV2()
])

# Load the bounding boxes from the CSV file
df = pd.read_csv('annotations.csv')

# Loop over the images in the folder
for filename in os.listdir('images'):
    # Load the image
    image = cv2.imread(os.path.join('images', filename))
    # Load the corresponding bounding boxes from the CSV file
    bboxes = df[df['image_id'] == filename]['bbox'].values
    # Convert the bounding boxes from YOLO format to xyxy format
    bboxes = [[float(coord) for coord in bbox.split(',')] for bbox in bboxes]
    bboxes = [[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] for bbox in bboxes]
    # Apply the transformations to the image and bounding boxes
    augmented = transform(image=image, bboxes=bboxes)
    # Extract the augmented image and bounding boxes
    augmented_image = augmented['image']
    augmented_bboxes = augmented['bboxes']
    # Convert the augmented bounding boxes back to YOLO format
    augmented_bboxes = [[(bbox[0] + bbox[2]) / 2 / 512, (bbox[1] + bbox[3]) / 2 / 512,
                         (bbox[2] - bbox[0]) / 512, (bbox[3] - bbox[1]) / 512] for bbox in augmented_bboxes]
    augmented_bboxes = [f'{bbox[0]:.6f},{bbox[1]:.6f},{bbox[2]:.6f},{bbox[3]:.6f}' for bbox in augmented_bboxes]
    # Save the augmented image and bounding boxes
    cv2.imwrite(os.path.join('augmented_images', f'augmented_{filename}'), augmented_image)
    with open(os.path.join('augmented_annotations', f'augmented_{filename[:-4]}.txt'), 'w') as f:
        for bbox in augmented_bboxes:
            f.write(f'0 {bbox}\n')

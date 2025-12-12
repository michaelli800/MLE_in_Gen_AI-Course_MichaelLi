from PIL import Image
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.foundation import FoundationPredictor

# Load the image
image = Image.open("./test_data/image/image.png")  # Replace with your image path
langs = ["en"]  # Specify the language(s)

# Initialize foundation (required)
foundation_predictor = FoundationPredictor()
# Initialize predictors
detection_predictor = DetectionPredictor()
recognition_predictor = RecognitionPredictor(foundation_predictor)
# Use the correct task
task_names = ["ocr_with_boxes"]
# Perform OCR
predictions = recognition_predictor([image], task_names, detection_predictor)

# Display results with polygon coordinates
for page in predictions:
    for line in page.text_lines:
        print(f"Text: {line.text}")
        print(f"Confidence: {line.confidence}")
        print(f"Polygon: {line.polygon}\n")

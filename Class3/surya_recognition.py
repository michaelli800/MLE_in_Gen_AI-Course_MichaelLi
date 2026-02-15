import os
import json
from PIL import Image, ImageDraw
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

image_path = "./test_data/image/image.png"
output_dir = "results1/"
os.makedirs(output_dir, exist_ok=True)
image = Image.open(image_path)
foundation_predictor = FoundationPredictor()
recognition_predictor = RecognitionPredictor(foundation_predictor)
detection_predictor = DetectionPredictor()

predictions = recognition_predictor([image], det_predictor=detection_predictor)

# 5. Process and Save Results
for i, page_result in enumerate(predictions):
    # Save the JSON data (matches --output_dir results/)
    output_file = os.path.join(output_dir, f"results_{i}.json")
    
    # Use model_dump for Pydantic v2 or .dict() for v1
    try:
        result_dict = page_result.model_dump()
    except AttributeError:
        result_dict = page_result.dict()
        
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=4)

    # 6. Replicate the --images flag (Manual Visualization)
    # Since surya.postprocess is gone, we draw polygons manually
    viz_image = image.copy()
    draw = ImageDraw.Draw(viz_image)
    
    for line in page_result.text_lines:
        # line.polygon is a list of [x, y] coordinates: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        # Flatten the list for PIL: [x1, y1, x2, y2, x3, y3, x4, y4]
        flat_poly = [coord for point in line.polygon for coord in point]
        draw.polygon(flat_poly, outline="red", width=3)

    viz_image.save(os.path.join(output_dir, f"debug_image_{i}.png"))
    print(f"OCR complete. Results saved to: {output_dir}")
from typing import List
from PIL import Image
import torch
import numpy as np
import io
from FastSAM.fastsam import FastSAM, FastSAMPrompt
from transformers import CLIPProcessor, CLIPModel

class VLMManager:
    def __init__(self):
        # Initialize the model here
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the pretrained CLIP model
        self.model = CLIPModel.from_pretrained("flax-community/clip-rsicd-v2").to(self.device)
        
        # Load the processor for the CLIP model
        self.processor = CLIPProcessor.from_pretrained("flax-community/clip-rsicd-v2")
        
        # Initialize FastSAM model
        self.fastSAM = FastSAM('./FastSAM.pt')
        
    def identify(self, image: bytes, caption: str) -> List[int]:
        # Perform object detection with a vision-language model   
        image_pil = Image.open(io.BytesIO(image)) # PIL Opener
        output_format = "PNG"
        output_filename = "saved_image"  
        output_filepath = f"{output_filename}.{output_format.lower()}"
        image_pil.save(output_filepath, format=output_format) # Temporarily save for segmentation analysis

        # Detect Segments
        everything_results = self.fastSAM(output_filepath, device=self.device, retina_masks=True, imgsz=1024, conf=0.3, iou=0.1,)
        prompt_process = FastSAMPrompt(output_filepath, everything_results, device=self.device)
        annotations = prompt_process.everything_prompt()

        cropped_boxes = []
        original_bboxes = []

        # Loop through all the Masks
        for annotation in annotations:
            cropped_image, bounding_box = self.crop_image_with_mask(image_pil, annotation, 5) # Padding = 5
            
            # Filter away masks above threshold area
            if self.calculate_bbox_area(bounding_box) < 6000:
                cropped_boxes.append(cropped_image)
                original_bboxes.append(bounding_box)

        # Retrieve query scores
        best_img_idx = self.retrieve(cropped_boxes, caption)
        best_bbox = original_bboxes[best_img_idx]
        
        return best_bbox

    def crop_image_with_mask(self, image, mask, padding):
        image = np.array(image)
        # Convert the PyTorch tensor to a NumPy array
        mask_np = mask.cpu().numpy() if mask.is_cuda else mask.numpy()

        # Find the coordinates of non-zero pixels in the binary mask
        nonzero_coords = np.argwhere(mask_np != 0)

        # Calculate the bounding box of the non-zero pixels
        min_row, min_col = np.min(nonzero_coords, axis=0)
        max_row, max_col = np.max(nonzero_coords, axis=0)

        # Define the cropping region with padding
        min_row_pad = max(0, min_row - padding)
        min_col_pad = max(0, min_col - padding)
        max_row_pad = min(image.shape[0], max_row + padding)
        max_col_pad = min(image.shape[1], max_col + padding)

        # Crop the region from the original image using the bounding box
        cropped_image = image[min_row_pad:max_row_pad, min_col_pad:max_col_pad]

        # Resize the cropped image to the desired size
        cropped_image_pil = Image.fromarray(cropped_image).resize((224, 224))

        # Adjust bounding box coordinates for the padded image
        min_row_adjusted = min_row 
        min_col_adjusted = min_col
        max_row_adjusted = max_row 
        max_col_adjusted = max_col

        # Define the bounding box coordinates
        bounding_box = self.convert_to_left_top_width_height_format((min_row_adjusted, min_col_adjusted, max_row_adjusted, max_col_adjusted))

        return cropped_image_pil, bounding_box

    def convert_to_left_top_width_height_format(self, bounding_box):
        min_row, min_col, max_row, max_col = bounding_box
        left = min_col
        top = min_row
        width = max_col - min_col
        height = max_row - min_row
        return int(left), int(top), int(width), int(height)

    @torch.no_grad()
    def retrieve(self, elements: List[Image.Image], search_text: str) -> int:
        # Preprocess images and tokenize text
        inputs = self.processor(
            text=[search_text] * len(elements), images=elements, return_tensors="pt", padding=True
        )
        
        # Move inputs to the appropriate device
        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}
        
        # Perform model inference
        outputs = self.model(
            **inputs
        )
        
        # Get similarity scores
        logits_per_image = outputs.image_embeds @ outputs.text_embeds.T
       
        # Extract the diagonal values for similarity scores
        diagonal_values = logits_per_image.diag()
        probs = diagonal_values.softmax(dim=0)  # Convert logits to probabilities
        
        #print(probs)
        
        # Find the index of the image with the highest similarity score
        most_similar_idx = torch.argmax(probs).item()
        return most_similar_idx
    
    def get_indices_of_values_above_threshold(self, values, threshold):
        return [i for i, v in enumerate(values) if v > threshold]

    def calculate_bbox_area(self, bbox):
        """
        Calculate the area of a bounding box.

        Parameters:
            left (int): The left coordinate of the bounding box.
            top (int): The top coordinate of the bounding box.
            width (int): The width of the bounding box.
            height (int): The height of the bounding box.

        Returns:
            int: The area of the bounding box.
        """
        return bbox[2] * bbox[3]
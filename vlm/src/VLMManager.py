from typing import List
from PIL import Image, ImageEnhance
import torch
import numpy as np
import io
import clip
from FastSAM.fastsam import FastSAM, FastSAMPrompt
from transformers import CLIPProcessor, CLIPModel
import cv2

class VLMManager:
    def __init__(self):
        # Initialize the model here
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model, self.processor = clip.load("ViT-B/32", device= self.device,jit=False)
        checkpoint = torch.load("clip-finetune/clip.pt")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Initialize FastSAM model
        self.fastSAM = FastSAM('./FastSAM.pt')
        
    def identify(self, image: bytes, caption: str) -> List[int]:
        # Perform object detection with a vision-language model   
        image_pil = Image.open(io.BytesIO(image)) # PIL Opener
        
        # Temporarily save for segmentation analysis
        output_format = "PNG"
        output_filename = "saved_image"  
        output_filepath = f"{output_filename}.{output_format.lower()}"
        image_pil.save(output_filepath, format=output_format) 

        # Detect Segments using FastSAM
        everything_results = self.fastSAM(output_filepath, device=self.device, retina_masks=True, imgsz=1024, conf=0.3, iou=0.65,)
        prompt_process = FastSAMPrompt(output_filepath, everything_results, device=self.device)
        annotations = prompt_process.everything_prompt()

        cropped_boxes = []
        original_bboxes = []

        # Loop through all the Segmented Masks
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
        
        # Sharpen and contrast
        cropped_image_pil = cropped_image_pil

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
        # Tokenize the search text
        text_tokens = clip.tokenize([search_text]).to(self.device)

        # Encode the text once
        text_features = self.model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

        sim_dict = {}
        for idx, image in enumerate(elements):
            # Preprocess the image
            processed_image = self.processor(image).unsqueeze(0).to(self.device)

            # Encode the image
            image_features = self.model.encode_image(processed_image).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Compute similarity
            similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

            # Store similarity in dictionary
            sim_dict[idx] = similarity.item()

        # Find the index with the highest similarity
        best_img_idx = max(sim_dict, key=sim_dict.get)
        return best_img_idx
    
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

    
    
    

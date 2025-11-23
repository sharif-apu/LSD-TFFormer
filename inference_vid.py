import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import time
from modelDefinitions.tfformer import TwoTineFormerN

# Define normalization parameters
normMean = [0.5, 0.5, 0.5]
normStd = [0.5, 0.5, 0.5]

class UnNormalize(object):
    def __init__(self):
        self.std = normStd
        self.mean = normMean

    def __call__(self, tensor, imageNetNormalize=None):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        if imageNetNormalize:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.mul_(s).add_(m)
        else:
            tensor = (tensor * 0.5) + 0.5
        
        return tensor

# Function to add text to an image
def add_text_label(image, text):
    # Set text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0  # Adjusted for 2K resolution
    font_thickness = 4  # Adjusted for 2K resolution
    text_color = (255, 255, 255)  # White color
    
    # Calculate text size
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_width, text_height = text_size
    
    # Calculate position (centered horizontally, near the top)
    x = (image.shape[1] - text_width) // 2
    y = 80  # Distance from the top
    
    # Add black background for better visibility
    cv2.rectangle(
        image, 
        (x - 15, y - text_height - 15), 
        (x + text_width + 15, y + 15), 
        (0, 0, 0), 
        -1
    )
    
    # Add text
    cv2.putText(
        image, 
        text, 
        (x, y), 
        font, 
        font_scale, 
        text_color, 
        font_thickness, 
        cv2.LINE_AA
    )
    
    return image

# Main execution code
if __name__ == '__main__':
    # Load model
    model = TwoTineFormerN().cuda()
    cp = "/home/sharif/LSD/TFFormer/cp/TFFormer_checkpoint.pth"
    checkpoint = torch.load(cp)
    model.load_state_dict(checkpoint['stateDictEG'])
    model.eval()  # Set model to evaluation mode
    model.half() 
    
    # Create unnormalizer
    unNormalize = UnNormalize()
    
    # Define input and output video paths
    input_video_path = "/home/sharif/LSD/TFFormer/viddha/input_dhaka4.mp4"
    output_video_path = input_video_path.replace("input", "output_demo")
    
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        exit()
    
    # Get video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 25  # Set output fps to 25 as requested
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define fixed dimensions
    fixed_width = 1280  # Half of 2K width
    fixed_height = 1440  # Full 2K height
    output_width = fixed_width * 2  # 2K width
    output_height = fixed_height  # 2K height
    
    print(f"Processing video: {input_video_path}")
    print(f"Total frames: {total_frames}, Original Resolution: {original_width}x{original_height}")
    print(f"Output Resolution: {output_width}x{output_height}, FPS: {fps}")
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))
    
    # Define transform for preprocessing
    transform = transforms.Compose([
        transforms.Resize((1024, 1024), interpolation=Image.BICUBIC), 
        transforms.ToTensor(),
        transforms.Normalize(normMean, normStd),
    ])
    
    # Create display window
    cv2.namedWindow('Processing Preview', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Processing Preview', 1280, 720)  # Resize window for better viewing
    
    # Process each frame
    start_time = time.time()
    frame_count = 0
    
    with torch.no_grad():  # Disable gradient calculation for inference
        for _ in tqdm(range(total_frames), desc="Processing frames"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert OpenCV frame to PIL Image for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Resize and prepare for model
            pil_image = pil_image.resize((1024, 1024))
            input_tensor = transform(pil_image).unsqueeze(0).cuda()
            
            # Run inference
            with torch.no_grad():
                output = model(input_tensor.half())
            
            # Process output tensor to image
            if isinstance(output, (list, tuple)):
                output_tensor = output[0]  # Take the first element if it's a list/tuple
            else:
                output_tensor = output
                
            if output_tensor.dim() == 4:  # [B, C, H, W]
                output_img = unNormalize(output_tensor[0].clone())
            else:
                output_img = unNormalize(output_tensor.clone())
            
            output_np = output_img.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
            output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
            
            # Resize the input and output frames to fixed dimensions
            input_resized = cv2.resize(frame, (fixed_width, fixed_height))
            output_resized = cv2.resize(output_np, (fixed_width, fixed_height))
            
            # Create a black canvas of 2K resolution
            comparison_frame = np.zeros((fixed_height, fixed_width * 2, 3), dtype=np.uint8)
            
            # Place the resized frames on the canvas
            comparison_frame[:, 0:fixed_width] = input_resized
            comparison_frame[:, fixed_width:fixed_width * 2] = output_resized
            
            # Add text labels to both sides
            left_half = comparison_frame[:, :fixed_width].copy()
            right_half = comparison_frame[:, fixed_width:].copy()
            
            left_half = add_text_label(left_half, "Input (Fold-5)")
            right_half = add_text_label(right_half, "Enhanced (Ours)")
            
            comparison_frame[:, :fixed_width] = left_half
            comparison_frame[:, fixed_width:] = right_half
            
            # Display the comparison frame (resized for display)
            cv2.imshow('Processing Preview', cv2.resize(comparison_frame, (1280, 720)))
            
            # Write frame to output video
            out.write(comparison_frame)
            frame_count += 1
            
            # Press 'q' to exit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    elapsed_time = time.time() - start_time
    fps_processing = frame_count / elapsed_time
    
    print(f"Video processing complete. Output saved to {output_video_path}")
    print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds ({fps_processing:.2f} FPS)")
    print(f"Output video has {frame_count} frames at {fps} FPS in 2K resolution ({output_width}x{output_height})")

import cv2
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import logging
import time
from PIL import Image
import sys
from threading import Thread, Lock
from queue import Queue
from googletrans import Translator
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def setup_logging():
    """Configure logging with basic formatting"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class CaptionGenerator:
    def __init__(self, processor, model, device):
        self.processor = processor
        self.model = model
        self.device = device
        self.current_caption = f"Initializing caption... ({device.upper()})"
        self.caption_queue = Queue(maxsize=1)
        self.lock = Lock()
        self.running = True
        self.thread = Thread(target=self._caption_worker)
        self.thread.daemon = True
        self.thread.start()
        self.translator = Translator()  

    def _caption_worker(self):
        while self.running:
            try:
                if not self.caption_queue.empty():
                    frame = self.caption_queue.get()
                    caption = self._generate_caption(frame)
                    with self.lock:
                        self.current_caption = caption
            except Exception as e:
                logging.error(f"Caption worker error: {str(e)}")
            time.sleep(0.1)  # Prevent busy waiting

    def _generate_caption(self, image):
        try:
            # Resize to 640x480 (or any other size)
            image_resized = cv2.resize(image, (640, 480))

            # Convert to RGB
            rgb_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            # Process the image for captioning
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=30,
                    num_beams=5,
                    num_return_sequences=1
                )

            caption = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

            translated = self.translator.translate(caption, src='en', dest='ko')
            caption_ko = translated.text

            return f"BLIP: {caption_ko} ({self.device.upper()})"
        except Exception as e:
            logging.error(f"Caption generation error: {str(e)}")
            return f"BLIP: Caption generation failed ({self.device.upper()})"

    def update_frame(self, frame):
        if self.caption_queue.empty():
            try:
                self.caption_queue.put_nowait(frame.copy())
            except:
                pass  # Queue is full, skip this frame

    def get_caption(self):
        with self.lock:
            return self.current_caption

    def stop(self):
        self.running = False
        self.thread.join()

def get_gpu_usage():
    """Get the GPU memory usage and approximate utilization"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)  # MB

        memory_used_percent = (memory_allocated / memory_total) * 100
        gpu_info = f"GPU Memory Usage: {memory_used_percent:.2f}% | Allocated: {memory_allocated:.2f} MB / {memory_total:.2f} MB"
        
        return gpu_info
    else:
        return "GPU not available"

def load_models():
    """Load BLIP model"""
    try:
        blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        blip_model = AutoModelForImageTextToText.from_pretrained("Salesforce/blip-image-captioning-large")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            # Set GPU memory usage limit to 90%
            torch.cuda.set_per_process_memory_fraction(0.99)
            blip_model = blip_model.to('cuda')

        return blip_processor, blip_model, device
    except Exception as e:
        logging.error(f"Failed to load models: {str(e)}")
        return None, None, None

def live_stream_with_caption(processor, model, device, display_width=1200, display_height=800):
    """Stream webcam feed with live captions and FPS"""
    video_path = "video/video06.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f'Failed to open video file: {video_path}')
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 코덱
    out = cv2.VideoWriter('result/result_video06.mp4', fourcc, fps, (width, height))

    logger.info(f"Video feed started successfully using {device.upper()}.")
    caption_generator = CaptionGenerator(processor, model, device)

    prev_time = time.time()  # Track time to calculate FPS

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from webcam.")
                break

            # Update caption and track FPS
            caption_generator.update_frame(frame)
            current_caption = caption_generator.get_caption()

            # Get GPU memory usage
            gpu_info = get_gpu_usage()

            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # Break caption into lines if it overflows
            max_width = 40  # Adjust max width for caption as needed
            caption_lines = [current_caption[i:i + max_width] for i in range(0, len(current_caption), max_width)]

            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)
            font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 17)

            draw.rectangle(
                (35,35, 600,150),
                fill=(255,255,255)
            )                           

            y_offset = 40
            for line in caption_lines:
                draw.text((50, y_offset), line, font=font, fill=(0, 0, 0))
                y_offset += 30


            #GPU
            draw.text((50, y_offset), gpu_info, font=font, fill=(0, 0, 0))
            y_offset += 30
            
            # FPS
            draw.text((50, y_offset), f"FPS: {fps:.2f}", font=font, fill=(0, 0, 0))

            # PIL
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

            out.write(frame)

            # Display the video frame
            cv2.imshow("BLIP: Unified Vision-Language Captioning", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    finally:
        caption_generator.stop()
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    logger = setup_logging()

    logger.info("Loading BLIP model...")
    blip_processor, blip_model, device = load_models()
    if None in (blip_processor, blip_model):
        logging.error("Failed to load the BLIP model. Exiting.")
        sys.exit(1)

    logger.info(f"Using {device.upper()} for inference.")
    logger.info("Starting live stream with BLIP captioning and FPS display...")
    live_stream_with_caption(blip_processor, blip_model, device)

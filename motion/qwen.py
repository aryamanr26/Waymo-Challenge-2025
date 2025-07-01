from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from qwen_vl_utils import process_vision_info
import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

# Load model + processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


print("Using CUDA:", torch.cuda.is_available())

# Load tokenizer and model to GPU
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     "Qwen/Qwen-VL",
#     trust_remote_code=True,
#     device_map="auto",
#     torch_dtype=torch.float16
# )

# Input
prompt = "What is the cat doing in the image?"
image_path = "blue-tabby-maine-coon-cat-running-outdoors_Nils-Jacobi_Shutterstock.jpg"
image = Image.open(image_path).convert("RGB")

# Qwen messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt}
        ]
    }
]

# Format input
chat_text = processor.apply_chat_template(messages, tokenize = False, add_generation_prompt=True)

# Process image
# image_inputs, video_inputs = process_vision_info(messages)
# image_inputs = [img.to(model.device) for img in image_inputs]  # ✅ Send images to GPU

# # Tokenize and send text to GPU
# inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)
# inputs.update({"images": image_inputs})  # ✅ Combine with image tensors

inputs = processor(text=chat_text, images=image, return_tensors="pt").to(model.device)
# Generate
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=512)

# Decode output
# output = tokenizer.decode(
#     output_ids[0][inputs["input_ids"].shape[1]:], 
#     skip_special_tokens=True
# )

# print("Answer:", output)

response = processor.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
print("Answer:", response)


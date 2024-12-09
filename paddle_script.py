# Testing paddle ocr

import os
from paddleocr import PaddleOCR, draw_ocr # For OCR processing
from groq import Groq  # For calling Groq LLaMA 3.3 API
import os  # For accessing environment variables like GROQ_API_KEY
from PIL import Image
# import matplotlib.pyplot as plt
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()


# Initialize the OCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Initialize with angle classification and English language

# Path to the input image
img_path = 'images_sample\image.png'

# Perform OCR
results = ocr.ocr(img_path, cls=True)

# Print the recognized text along with confidence scores
print("Recognized Text with Confidence Scores:")
for idx, res in enumerate(results[0]):
    text, confidence = res[1]
    print(f"{idx + 1}. {text} (Confidence: {confidence:.2f})")

# Draw OCR results on the image
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in results[0]]
txts = [line[1][0] for line in results[0]]
scores = [line[1][1] for line in results[0]]

# Path to font file (adjust path to your environment)
font_path = "\\font\\arial.ttf"

# Generate annotated image
annotated_image = draw_ocr(image, boxes, txts, scores, font_path=font_path)

# Convert and save/display the result
annotated_image = Image.fromarray(annotated_image)
annotated_image.save('result.jpg')

# Perform OCR
results = ocr.ocr(img_path, cls=True)

# Extract the recognized text
recognized_text = [res[1][0] for res in results[0]]

# Save the text to a file
output_file = 'op.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    for line in recognized_text:
        f.write(line + '\n')

print(f"Recognized text has been saved to {output_file}")

# # Display the image with bounding boxes using matplotlib
# plt.figure(figsize=(10, 10))
# plt.imshow(annotated_image)
# plt.axis('off')
# plt.title("OCR Results")
# plt.show()

# Initialize Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),  # Fetch the key from environment variables
)


user_prompt = input("Enter your question or prompt: ")

final_prompt = f"{recognized_text}\n\n{user_prompt}"

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": final_prompt,
        }
    ],
    model="llama-3.3-70b-specdec",
)

# Retrieve and display the response
groq_response = chat_completion.choices[0].message.content
print("Response from Groq LLaMA 3.3:")
print(groq_response)

with open('groq_response.txt', 'w', encoding='utf-8') as f:
    f.write(groq_response)

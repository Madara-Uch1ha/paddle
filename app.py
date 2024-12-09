import os
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Paths
FONT_PATH = "font/arial.ttf"
os.makedirs("output", exist_ok=True)

def perform_ocr(image_path):
    results = ocr.ocr(image_path, cls=True)
    text = [line[1][0] for line in results[0]]
    return results, text

def annotate_image(image_path, results):
    image = Image.open(image_path).convert('RGB')
    boxes = [line[0] for line in results[0]]
    txts = [line[1][0] for line in results[0]]
    scores = [line[1][1] for line in results[0]]
    annotated_img = draw_ocr(image, boxes, txts, scores, font_path=FONT_PATH)
    return Image.fromarray(annotated_img)

def save_text(text, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in text:
            f.write(line + '\n')

def query_groq(recognized_text, user_prompt):
    final_prompt = f"{recognized_text}\n\n{user_prompt}"
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": final_prompt}],
        model="llama-3.2-3b-preview",
    )
    return chat_completion.choices[0].message.content

def main():
    st.title("OCR and Ingredient Analysis Pipeline")
    st.write("Upload or capture an image to perform OCR and query a language model.")

    # Choose input method
    input_method = st.radio("Choose input method:", ["Upload an Image", "Take a Photo"])

    # Handle input
    if input_method == "Upload an Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    else:
        uploaded_file = st.camera_input("Take a photo")

    if uploaded_file is not None:
        # Display input image
        st.image(uploaded_file, caption="Input Image", use_column_width=True)

        # Save uploaded file to a temporary path
        img_path = "temp_image.jpg"
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Perform OCR
        st.write("Performing OCR...")
        results, recognized_text = perform_ocr(img_path)
        st.write("**Recognized Text:**")
        st.write(recognized_text)

        # Annotate Image
        st.write("Annotating Image...")
        annotated_image = annotate_image(img_path, results)
        st.image(annotated_image, caption="Annotated Image", use_column_width=True)

        # Save recognized text
        text_file = os.path.join("output", "recognized_text.txt")
        save_text(recognized_text, text_file)
        st.download_button(
            "Download Recognized Text", data="\n".join(recognized_text), file_name="recognized_text.txt"
        )

        # Query LLM
        st.write("Querying the LLM...")
        user_prompt = "Can you list all the benefits and potential health risks of the ingredients? Ensure the data is unbiased."
        response = query_groq("\n".join(recognized_text), user_prompt)
        st.write("**LLM Response:**")
        st.write(response)

        # Save response to file
        response_file = os.path.join("output", "groq_response.txt")
        with open(response_file, "w", encoding="utf-8") as f:
            f.write(response)
        st.download_button(
            "Download LLM Response", data=response, file_name="groq_response.txt"
        )
    # if st.button("Submit Query"):


if __name__ == "__main__":
    main()

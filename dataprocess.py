import os
import json
from pathlib import Path
from docx import Document


def process_docx(file_path):
    try:
        doc = Document(file_path)
        content = []
        for para in doc.paragraphs:
            content.append(para.text)
        return "\n".join(content)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return ""


def create_json_structure(image_files, observations, diagnosis):
    json_data = []
    for image in image_files:
        if image.name == ".DS_Store":  # Skip processing .DS_Store files
            continue
        entry = {
            "messages": [
                {
                    "content": "<image>\n请描述检查所见。",
                    "role": "user"
                },
                {
                    "content": observations,
                    "role": "assistant"
                },
                {
                    "content": "根据上述检查所见，得到的诊断结果是什么？",
                    "role": "user"
                },
                {
                    "content": diagnosis,
                    "role": "assistant"
                }
            ],
            "images": [
                str(image)
            ]
        }
        json_data.append(entry)
    return json_data


def main():
    base_path = Path('ABUS_HHUS')
    output_file = Path('ABUS_HHUS/output.json')
    all_data = []

    if not base_path.exists():
        print(f"Base path {base_path} does not exist.")
        return

    for dirpath, dirnames, filenames in os.walk(base_path):
        print(f"Current directory: {dirpath}")
        print(f"Subdirectories: {dirnames}")
        print(f"Files: {filenames}")

        folder_name = Path(dirpath).parts[-2]

        if 'images' in dirnames:
            image_path = Path(dirpath) / 'images'
            images = [image for image in image_path.glob('*') if image.is_file()]
            docx_files = [f for f in filenames if f.endswith('.docx')]

            print(f"Found images: {images}")
            print(f"Found docx files: {docx_files}")

            for docx_file in docx_files:
                docx_path = Path(dirpath) / docx_file
                print(f"Processing docx file: {docx_path}")
                text = process_docx(docx_path)

                if not text:
                    print(f"No content found in {docx_path}")
                    continue

                # Split text into observations and diagnosis
                if '诊断：' in text:
                    observations, diagnosis = text.split('诊断：', 1)
                    observations = observations.strip()
                    diagnosis = "诊断：" + diagnosis.strip()
                else:
                    observations = text.strip()
                    diagnosis = ""

                print(f"Observations: {observations}")
                print(f"Diagnosis: {diagnosis}")

                # Create JSON structure for each image
                if images:  # Ensure there are images in the folder
                    json_entries = create_json_structure(images, observations, diagnosis)
                    all_data.extend(json_entries)
                else:
                    print(f"No images found in: {image_path}")

    if all_data:
        # Ensure the directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Write all data to a single JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)
        print(f"Data written to {output_file}")
    else:
        print("No data to write")


if __name__ == "__main__":
    main()

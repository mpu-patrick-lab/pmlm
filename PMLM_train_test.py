from vllm import LLM, SamplingParams
from PIL import Image
from io import BytesIO
import requests


# llm = LLM(model="/root/autodl-tmp/llama-1.5-7b-hf-ours", max_model_len=2048,image_input_type="pixel_values",image_token_id=32000,image_input_shape="1,3,336,336",image_feature_size=576)
llm = LLM(model="/root/autodl-tmp/llama-1.5-7b-hf-ours", max_model_len=2048)



initial_prompt = "USER: <image>\n请描述检查所见。\nASSISTANT:"
image = Image.open("/root/autodl-tmp/test.jpg")


sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)


outputs = llm.generate(
    {
        "prompt": initial_prompt,
        "multi_modal_data": {"image": image}
    },
    sampling_params=sampling_params
)

generated_text = ""
for o in outputs:
    generated_text += o.outputs[0].text

print(f"LLM output: {generated_text}")


def chat(prompt, image_path):
    image = Image.open(image_path)

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)

    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {"image": image}
        },
        sampling_params=sampling_params
    )

    generated_text = ""
    for o in outputs:
        generated_text += o.outputs[0].text
    return generated_text


def chat_with_history(image_path):
    res = []
    prompt1 = "USER: <image>\n请描述检查所见。\nASSISTANT:"
    prompt2 = "\nUSER: 根据上述检查所见，得到的诊断结果是什么？\nASSISTANT:"
    res1 = chat(prompt=prompt1, image_path=image_path)
    res.append(res1)
    ful_prompt = prompt1 + " \n" + res1 + prompt2
    res2 = chat(prompt=ful_prompt, image_path=image_path)
    res.append(res2)
    return res

import os
import json
def process_images(base_path):
    results = []

    for folder in ['b', 'm']:
        folder_path = os.path.join(base_path, folder)
        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            if os.path.isdir(subfolder_path):
                images_folder = os.path.join(subfolder_path, 'images')
                if os.path.exists(images_folder):
                    image_files = os.listdir(images_folder)
                    if image_files:
                        first_image_path = os.path.join(images_folder, image_files[0])
                        res = chat_with_history(first_image_path)
                        json_entry = {
                            "messages": [
                                {
                                    "content": "请描述检查所见。",
                                    "role": "user"
                                },
                                {
                                    "content": res[0],
                                    "role": "assistant"
                                },
                                {
                                    "content": "根据上述检查所见，得到的诊断结果是什么？",
                                    "role": "user"
                                },
                                {
                                    "content": res[1],
                                    "role": "assistant"
                                }
                            ],
                            "images": [
                                first_image_path
                            ]
                        }
                        results.append(json_entry)

    return results


base_path = 'OCR_classification'
output_file = 'output.json'

results = process_images(base_path)


with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
"""
res=chat_with_history("/root/autodl-tmp/test.jpg")
image_path="/root/autodl-tmp/test.jpg"
prompt1 = "USER: <image>\n请描述检查所见。\nASSISTANT:"
prompt2 = "\nUSER: 根据上述检查所见，得到的诊断结果是什么？\nASSISTANT:"
res1 = chat(prompt=prompt1,image_path=image_path)
ful_prompt = prompt1+" \n"+res1+prompt2

res2 = chat(prompt=ful_prompt,image_path=image_path)
sec_prompt=initial_prompt+" \n"+generated_text

ful_prompt=sec_prompt+"\nUSER: 根据上述检查所见，得到的诊断结果是什么？\nASSISTANT:"

outputs3 = llm.generate(
    {
        "prompt": ful_prompt,
        "multi_modal_data": {"image": image}
    },
    sampling_params=sampling_params
)

generated_text3 = ""
for o in outputs3:
    generated_text3 += o.outputs[0].text

print(f"LLM output: {generated_text3}")

history = ["请描述检查所见","检查所见：US+ABVS：双乳大小对称，外形无殊，腺体回声不均，结构紊乱，以双乳外上象限为著，双乳内可见多个囊性小暗区，透声可。于右乳9点钟位置（距乳头5.8cm）可见一大小约1.0*0.6*0.9cm的低回声结节，形态不规则，边缘光整，内回声欠均，后伴浅声衰，CDFI未见明显血流信号，弹性评分3分。右乳可见手术疤痕。双侧腋下未见明显肿大淋巴结回声。"]

full_prompt = f"USER: {history[0]}\nASSISTANT: {history[1]}"

"""
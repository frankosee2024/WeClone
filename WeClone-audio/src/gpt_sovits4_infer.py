import os
import torch
from gpt_sovits4 import GPTSoVITS4

model = GPTSoVITS4("WeClone-audio/pretrained_models/GPT-SoVITS4", device="cuda")

text = "晚上好啊,小可爱们，该睡觉了哦"
reference = os.path.join(os.path.dirname(__file__), "sample.wav")

with torch.no_grad():
    wav = model.inference(text=text, reference_audio=reference)
    out_path = os.path.join(os.path.dirname(__file__), "output_gpt_sovits4.wav")
    model.save(wav, out_path)
    print("生成成功！")

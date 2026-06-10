# Speak-mk1: Multimodal Speech Therapy SALM
 
Speak-mk1 is an advanced Speech-Language Pathology (SLP) AI system designed for highly accurate articulation therapy, real-time error detection, and corrective feedback. By combining state-of-the-art Speech Audio Language Models (SALMs) with computer vision tracking, it offers clinical-grade feedback natively through a standard webcam and microphone.
 
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-blue)](https://huggingface.co/SakhrML/SpeakMK1_early)
[![GitHub License](https://img.shields.io/github/license/ableflyer/Speak-mk1)](LICENSE.md)
 
---
 
## 📂 Repository Structure & Core Code
 
The heart of the project lives inside the `Model/Code` directory. If you are looking for the actual model architecture, tokenizers, and custom layer implementations, navigate there:
 
👉 **[Go directly to the Source Code (`Model/Code`)](https://github.com/ableflyer/Speak-mk1/tree/main/Model/Code)**
 
### Project Map
* **`Model/Code/`** - Houses the core machine learning stack, including custom Mamba layers, the audio-to-text projection Q-Former, tokenizers, and training configurations.
* **`requirements.txt`** - System-wide Python dependencies.
* **`app.py && wav2vec2test.py`** - testing files before we started working on the model.
 
---
 
## 🚀 Key Features
 
* **Advanced Audio Feature Extraction:** Utilizing a mamba-attention hybrid to evaluate speech voicing, manner, and place of articulation.
* **Real-time Oral Kinematics (WIP):** Computer vision tracking powered by MediaPipe FaceLandmarker to capture mouth opening geometry, lip protrusion, and jaw/tongue tip position.
* **The Hybrid Reasoning engine:** An ultra-efficient 70M parameter Mamba-attention hybrid model.
* **Multi-Stage Training Curriculum:** Pretrained on generalized text, domain-adapted via CHILDES, injected with clinical knowledge from PubMed Central, and final instruction-tuned on specialized SLP datasets.
 
---
 
## 🛠️ System Architecture
 
The system consists of three fundamental components operating in tandem (defined across the codebase in `Model/Code`):
 
1. **Audio Encoder:** A custom Mamba SSM-based audio stack coupled with a BLIP-2 style Q-Former to project audio features straight into the language model.
2. **Video Pipeline (WIP):** Frame-by-frame visual tracking extracting vital articulatory metrics straight from a traditional webcam.
3. **Core LLM (SpeakMK1LLM):** Accepts both acoustic and visual tokens to spot exact phonological errors and immediately generate natural language hints.
 
---
 
## 📦 Installation & Setup
 
### 1. Clone the Repository
```bash
git clone [https://github.com/ableflyer/Speak-mk1.git](https://github.com/ableflyer/Speak-mk1.git)
cd Speak-mk1
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
## 🤗 Model Zoo
The specialized weights for our SLP-adapted reasoning engines are hosted on Hugging Face.
👉 [Download the Speak-mk1 Models on Hugging Face](https://huggingface.co/SakhrML/SpeakMK1_early)
## 📝 Citation & Research Paper
This software implementation accompanies our upcoming research paper on speak-mk1. If you use this code or model in your academic work, please cite it:
```bibtex
bibtex coming soon
```
## 📄 License
The code in this repository is licensed under the **MIT License**; allowing free academic and commercial reuse, modification, and distribution, provided attribution is maintained. See the LICENSE file for details.

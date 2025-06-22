# 🎵 Violin-to-Flute Timbre Transfer with DDSP

This repository implements a **timbre transfer system** using the [**Differentiable Digital Signal Processing (DDSP)**](https://arxiv.org/abs/2001.04643) framework. Given a **violin recording** as input, our model generates a **flute-like version** of the same musical phrase, preserving pitch and dynamics while altering timbre.

---

## 📖 Overview

### 🔍 Original Framework
DDSP combines **classic digital signal processing (DSP)** elements (oscillators, noise, filters, reverb) with **deep learning**, enabling **interpretable and high-quality audio synthesis**.

### 💡 Our Contribution
We adapt the DDSP architecture to a **low-data timbre transfer setting**, using only a few minutes of training audio:
1. **Pitch + Loudness Extraction**:  
   Extract conditioning features using a pretrained **CREPE model** for F₀ and standard loudness estimation.
2. **Temporal Encoder + Synthesizer**:  
   A GRU-based model predicts parameters for a **harmonic + noise synthesizer**.
3. **Timbre Transfer from Violin to Flute**:  
   Trained only on monophonic violin and flute datasets, the model learns to resynthesize the input **as if played by a flute**.

---

## 🛠️ System Architecture

The diagram below illustrates the inference-time pipeline:

![Architecture Diagram](Diagrams/detailed_training_diagram.png)

### Steps:
1. **Input Audio**:  
   A monophonic violin recording.
2. **Feature Extraction**:  
   - Pitch: via **CREPE**  
   - Loudness: A-weighted power + log compression
3. **Temporal Encoder**:  
   A GRU model processes features and outputs synthesis parameters.
4. **Harmonic + Noise Synthesizer**:  
   Generates audio conditioned on F₀, amplitude, and noise content.
5. **Reverb Module**:  
   Adds room acoustics using learned or fixed **impulse response**.

---

## ✨ Features

- ✅ **Interpretable audio synthesis** using DDSP components
- 🎯 **Timbre transfer** without requiring paired audio or MIDI
- 🧠 **Efficient learning** from just a few minutes of data
- 🔁 Modular DSP blocks: harmonic synthesis, noise shaping, and reverb

---

## 📊 Dataset

We use **monophonic solo recordings** of:
- 🎻 **Violin** (13 minutes)
- 🎶 **Flute** (13 minutes)  
Sourced from [**MusOpen**](https://musopen.org/) — royalty-free and consistent in acoustic environment.

- Sample rate: **16 kHz**, mono
- Segmentation: **4-second clips** (64,000 samples)
- Consistent performer and room acoustics per instrument

> This minimal dataset setup contrasts with large-scale datasets like NSynth (300k notes) and CocoChorales (50+ hours), highlighting DDSP’s strength in **low-resource learning**.

---

## 🔊 Example Output

All generated audio examples from our timbre transfer model can be found in the [`recordings`](./reccordings) folder.

These include violin inputs and their corresponding flute-converted outputs, organized by experiment and timestamp.


## 👥 Contributors

This project was developed by:

- **Ronel Davidov**  
- **Aviv Ben Ari**  
- **Yuval veiman**  
- **Dolev Dahan**

---

## 📂 References

- **DDSP Paper**:  
  [DDSP: Differentiable Digital Signal Processing (ICLR 2020)](https://arxiv.org/abs/2001.04643)

- **Pitch Estimation (CREPE)**:  
  [CREPE: A Convolutional Representation for Pitch Estimation](https://ieeexplore.ieee.org/document/8461329)

- **MusOpen Dataset Source**:  
  [https://musopen.org](https://musopen.org)

---

📫 For access to trained models or further questions, feel free to contact:  
**davidovr@post.bgu.ac.il, beaviv@post.bgu.ac.il,  veimany@post.bgu.ac.il,  dahandol@post.bgu.ac.il**

We welcome contributions, feedback, and questions! 🚀🎶

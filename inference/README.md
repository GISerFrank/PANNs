# PANNs 音频分析与可视化工具

这是一个基于预训练音频神经网络（PANNs）的自动化工具，用于对音频文件进行深入分析。脚本能够批量处理文件夹中的音频，为每个文件生成详细的推断数据和多样化的可视化图表，涵盖了音频内容标签（Audio Tagging）和声音事件检测（Sound Event Detection）两个核心任务。

![示例输出图像](https://i.imgur.com/kY64t1N.png)
*图：声音事件检测热力图（Pianoroll）示例*

---

## ✨ 主要功能

- **批量处理**: 自动扫描指定文件夹内的所有音频文件（支持 `.wav`, `.mp3`, `.flac`, `.ogg` 等格式）。
- **随机抽样**: 可设定处理文件的数量，并从文件夹中随机选取样本进行分析，便于快速测试和验证。
- **双重任务分析**:
    - **音频内容标签 (Audio Tagging)**: 总结整个音频片段包含哪些声音事件。
    - **声音事件检测 (Sound Event Detection)**: 精确到每一帧，判断特定声音事件在何时发生。
- **全面的结果输出**: 为每个音频文件生成一个独立的文件夹，包含：
    - **原始推断数据**: `clipwise_output.npy`, `framewise_output.npy`, `embedding.npy`。
    - **元数据**: `metadata.json`，记录了处理时间、模型路径、音频信息等，保证实验可复现性。
- **多样化可视化**:
    - Top-10 音频标签条形图
    - 声音事件检测曲线图
    - 声音事件检测热力图（钢琴卷帘图）
    - 事件发生时间轴图

---

## 🚀 环境准备

本项目依赖于 Conda 进行环境管理，以确保所有依赖库的版本兼容性。

1.  **创建并激活 Conda 环境**
    建议使用 Python 3.7 或 3.8。PANNs 的原始环境是基于 Python 3.7 的。

    ```bash
    # 创建一个名为 panns_env 的新环境
    conda create -n panns_env python=3.7

    # 激活环境
    conda activate panns_env
    ```

2.  **安装依赖库**
    项目所需的依赖库通常在 `requirements.txt` 文件中定义。请确保已安装所有必要的库。核心依赖包括：

    ```bash
    # 安装 PyTorch (GPU版本)
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

    # 安装 PANNs 推断库及其他依赖
    pip install panns-inference librosa matplotlib numpy
    ```
    *注意：请根据你的 CUDA 版本选择合适的 PyTorch 安装命令。*

---

## 🛠️ 如何使用

1.  **克隆或下载项目**
    将本项目代码下载到你的本地或服务器上。

2.  **准备音频文件**
    将所有你需要分析的音频文件放入一个文件夹中，例如 `my_audio_files/`。

3.  **配置并运行脚本**
    打开 `generate_all_results.py` 文件，修改文件末尾的两个主要参数：

    ```python
    if __name__ == '__main__':
        # --- 1. 修改为你的音频文件夹路径 ---
        INPUT_AUDIO_DIR = 'my_audio_files/' 
        
        # --- 2. 控制随机处理的文件数量 ---
        # 设置为 None 表示处理所有文件
        MAX_FILES_TO_PROCESS = 10 
    ```

4.  **执行脚本**
    在终端中运行以下命令：

    ```bash
    python generate_all_results.py
    ```

脚本将开始处理，并在终端打印出每个文件的处理进度。

---

## 📊 输出结果说明

处理完成后，对于每一个音频文件（例如 `sample.wav`），你都会得到一个对应的结果文件夹（`sample_wav_results/`），其中包含：

-   **`clipwise_output.npy`**: 片段级预测结果，形状为 `(1, 527)`，用于 Audio Tagging。
-   **`embedding.npy`**: 音频嵌入向量，形状为 `(1, 2048)`，代表音频的“声学指纹”。
-   **`framewise_output.npy`**: 帧级别预测结果，形状为 `(帧数, 527)`，用于 Sound Event Detection。
-   **`metadata.json`**: 本次分析的元数据。
-   **`1_audio_tagging_barchart.png`**: Top-10 音频标签的可视化条形图。
-   **`2_sed_line_plot.png`**: 最显著的几个声音事件的概率曲线图。
-   **`3_sed_pianoroll.png`**: 声音事件的热力图，直观展示事件的时序和重叠。
-   **`4_event_timeline.png`**: 声音事件的发生时间轴图。

---

## 🔗 致谢

本项目基于 [Qiuqiang Kong](https://github.com/qiuqiangkong) 等人出色的工作：
-   **PANNs**: [PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://arxiv.org/abs/1912.10211)
-   **代码库**: [qiuqiangkong/audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn)
-   **推断库**: [qiuqiangkong/panns_inference](https://github.com/qiuqiangkong/panns_inference)


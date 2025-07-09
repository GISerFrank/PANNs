import os
import json
import librosa
import numpy as np
import matplotlib.pyplot as plt
from panns_inference import AudioTagging, SoundEventDetection, labels
from datetime import datetime
from pathlib import Path
import glob # 导入 glob 模块用于文件匹配
import random # 导入 random 模块用于随机选取

def generate_all_results(audio_path):
    """
    对单个音频文件执行推断，并生成所有数据和可视化结果。
    (此函数内容保持不变)
    """
    print(f"--- Processing: {audio_path} ---")
    
    # --- 1. 设置 ---
    output_dir = os.path.splitext(os.path.basename(audio_path))[0] + '_results'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}/")

    # --- 2. 加载音频 ---
    try:
        (audio, sr) = librosa.core.load(audio_path, sr=32000, mono=True)
        audio = audio[None, :]  # 增加批次维度
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    # --- 3. 设置模型路径并执行推断 ---
    device = 'cuda' # 或者 'cpu'
    at_checkpoint_name = 'Cnn14_mAP=0.431.pth'
    sed_checkpoint_name = 'Cnn14_DecisionLevelMax.pth'
    default_path_dir = Path.home() / 'panns_data'
    at_model_path = default_path_dir / at_checkpoint_name
    sed_model_path = default_path_dir / sed_checkpoint_name

    print("Initializing models...")
    try:
        at = AudioTagging(checkpoint_path=None, device=device)
        sed = SoundEventDetection(checkpoint_path=None, device=device)
        
        print("Running inference...")
        (clipwise_output, embedding) = at.inference(audio)
        framewise_output = sed.inference(audio)
    except Exception as e:
        print(f"Error during model inference: {e}")
        return

    # --- 4. 保存所有原始数据 ---
    print("Saving raw .npy data and metadata.json...")
    np.save(os.path.join(output_dir, 'clipwise_output.npy'), clipwise_output)
    np.save(os.path.join(output_dir, 'embedding.npy'), embedding)
    np.save(os.path.join(output_dir, 'framewise_output.npy'), framewise_output)

    metadata = {
        'source_audio_path': os.path.abspath(audio_path),
        'sample_rate': sr,
        'audio_duration_seconds': len(audio[0]) / sr,
        'model_checkpoint_at': str(at_model_path),
        'model_checkpoint_sed': str(sed_model_path),
        'analysis_timestamp': datetime.now().isoformat()
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    # --- 5. 生成并保存多样化的可视化结果 ---
    print("Generating and saving visualizations...")

    # 5.1 音频标签（片段级）- Top-K 条形图
    plt.figure(figsize=(10, 8))
    top_k = 10
    topk_indexes = np.argsort(clipwise_output[0])[::-1][:top_k]
    topk_probs = clipwise_output[0][topk_indexes]
    topk_labels = np.array(labels)[topk_indexes]
    plt.barh(np.arange(top_k), topk_probs, align='center')
    plt.yticks(np.arange(top_k), topk_labels)
    plt.gca().invert_yaxis()
    plt.xlabel('Probability')
    plt.title('Top-10 Audio Tags (Clip-wise)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_audio_tagging_barchart.png'))
    plt.close()

    # 5.2 声音事件检测（帧级）- 曲线图
    frame_wise_prob = framewise_output[0]
    top_k_plot = 5
    top_k_indexes_plot = np.mean(frame_wise_prob, axis=0).argsort()[::-1][:top_k_plot]
    plt.figure(figsize=(10, 5))
    for i in top_k_indexes_plot:
        plt.plot(frame_wise_prob[:, i], label=labels[i])
    plt.ylim(0, 1)
    plt.xlabel('Frames')
    plt.ylabel('Probability')
    plt.title('Sound Event Detection (Line Plot)')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_sed_line_plot.png'))
    plt.close()

    # 5.3 声音事件检测（帧级）- 钢琴卷帘图 (Pianoroll)
    plt.figure(figsize=(10, 8))
    top_k_pianoroll = 15
    top_indices_pianoroll = np.mean(frame_wise_prob, axis=0).argsort()[::-1][:top_k_pianoroll]
    plt.imshow(frame_wise_prob[:, top_indices_pianoroll].T, aspect='auto', interpolation='nearest', origin='lower')
    plt.yticks(np.arange(top_k_pianoroll), np.array(labels)[top_indices_pianoroll])
    plt.xlabel('Frames')
    plt.ylabel('Classes')
    plt.title('Sound Event Detection (Pianoroll/Heatmap)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_sed_pianoroll.png'))
    plt.close()

    # 5.4 声音事件检测（帧级）- 事件时间轴图
    plt.figure(figsize=(10, 8))
    threshold = 0.5
    top_k_timeline = 10
    top_indices_timeline = np.mean(frame_wise_prob, axis=0).argsort()[::-1][:top_k_timeline]
    
    for i, label_idx in enumerate(top_indices_timeline):
        detected_frames = np.where(frame_wise_prob[:, label_idx] > threshold)[0]
        if len(detected_frames) > 0:
            blocks = np.split(detected_frames, np.where(np.diff(detected_frames) != 1)[0] + 1)
            for block in blocks:
                if len(block) > 0:
                    start_frame = block[0]
                    end_frame = block[-1]
                    plt.barh(y=i, width=end_frame-start_frame, left=start_frame, height=0.6)

    plt.yticks(np.arange(top_k_timeline), np.array(labels)[top_indices_timeline])
    plt.gca().invert_yaxis()
    plt.xlabel('Frames')
    plt.ylabel('Classes')
    plt.title(f'Event Timeline (Threshold > {threshold})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_event_timeline.png'))
    plt.close()

    print(f"--- Finished processing: {audio_path} ---\n")


# ====================================================================
# --- 主要改动部分 ---
# ====================================================================
if __name__ == '__main__':
    # --- 1. 在这里修改你要处理的音频文件夹路径 ---
    # 例如: '/scratch/bliao6/test_audios' 或 'examples/'
    INPUT_AUDIO_DIR = '/scratch/bliao6/test_audios/' 
    
    # --- 2. 新增：控制最大处理文件数 ---
    # 设置为 None 表示处理所有找到的文件
    # 设置为一个数字（如 5）则只处理前5个文件，方便快速测试
    MAX_FILES_TO_PROCESS = 5 

    # --- 3. 定义支持的音频文件格式 ---
    supported_formats = ['*.wav', '*.mp3', '*.flac', '*.ogg']
    
    # --- 4. 自动查找文件夹内所有音频文件 ---
    audio_files_to_process = []
    for fmt in supported_formats:
        search_path = os.path.join(INPUT_AUDIO_DIR, fmt)
        audio_files_to_process.extend(glob.glob(search_path))

    # --- 5. 【新改动】随机打乱文件列表 ---
    random.shuffle(audio_files_to_process)

    # --- 6. 根据 MAX_FILES_TO_PROCESS 截取文件列表 ---
    if MAX_FILES_TO_PROCESS is not None:
        # 确保不会因为文件总数少于MAX_FILES_TO_PROCESS而出错
        num_to_process = min(MAX_FILES_TO_PROCESS, len(audio_files_to_process))
        audio_files_to_process = audio_files_to_process[:num_to_process]

    # --- 7. 循环处理找到的所有文件 ---
    if not audio_files_to_process:
        print(f"No audio files found in the directory: {INPUT_AUDIO_DIR}")
    else:
        print(f"Randomly selected {len(audio_files_to_process)} audio files to process.")
        for audio_file in audio_files_to_process:
            generate_all_results(audio_file)
            
    print("===================================")
    print("All tasks finished.")
    print("===================================")


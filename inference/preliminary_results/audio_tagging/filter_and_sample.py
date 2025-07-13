import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt

def filter_and_sample_by_label(input_csv_path, output_csv_path, samples_per_label=200):
    """
    读取一个包含音频标签的CSV文件，为每个标签随机选取指定数量的样本，
    并将结果保存到一个按标签排序的新CSV文件中。
    同时，生成并保存相关的统计数据和可视化图表。

    Args:
        input_csv_path (str): 输入的CSV文件路径。
        output_csv_path (str): 输出的CSV文件路径。
        samples_per_label (int): 每个标签要选取的最大样本数。
    """
    print(f"--- Starting to process file: {input_csv_path} ---")

    # --- 1. 检查输入文件是否存在 ---
    if not os.path.exists(input_csv_path):
        print(f"Error: Input file not found at {input_csv_path}")
        return

    # --- 2. 使用 pandas 读取CSV文件 ---
    try:
        print("Reading CSV into DataFrame...")
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
        
    print("CSV file loaded successfully. Columns found:", df.columns.tolist())
    
    if 'Label' not in df.columns:
        print("Error: The input CSV must contain a 'Label' column.")
        return

    # --- 3. 按标签分组并进行随机抽样 ---
    print(f"Sampling up to {samples_per_label} records for each label...")
    sampled_df = df.groupby('Label', group_keys=False).apply(lambda x: x.sample(n=min(len(x), samples_per_label)))
    
    print("Sorting the final results...")
    sorted_df = sampled_df.sort_values(by=['Label', 'Probability'], ascending=[True, False])

    # --- 4. 保存筛选后的主要数据到CSV文件 ---
    try:
        print(f"Saving sampled data to: {output_csv_path}")
        sorted_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    except Exception as e:
        print(f"Error saving output CSV file: {e}")
        return

    # ====================================================================
    # --- 新增功能：生成统计数据与可视化 ---
    # ====================================================================
    print("\n--- Generating statistics and visualization ---")
    
    # --- 5. 计算抽样前后的统计数据 ---
    print("Calculating statistics...")
    original_counts = df['Label'].value_counts().reset_index()
    original_counts.columns = ['Label', 'OriginalCount']
    
    sampled_counts = sampled_df['Label'].value_counts().reset_index()
    sampled_counts.columns = ['Label', 'SampledCount']
    
    # 合并统计数据
    stats_df = pd.merge(original_counts, sampled_counts, on='Label', how='left').fillna(0)
    stats_df['SampledCount'] = stats_df['SampledCount'].astype(int)
    
    # --- 6. 保存统计数据到CSV文件 ---
    base_output_name = os.path.splitext(output_csv_path)[0]
    stats_csv_path = f"{base_output_name}_stats.csv"
    try:
        print(f"Saving statistics to: {stats_csv_path}")
        stats_df.to_csv(stats_csv_path, index=False, encoding='utf-8')
    except Exception as e:
        print(f"Error saving stats CSV file: {e}")

    # --- 7. 生成并保存可视化图表 ---
    try:
        print("Generating comparison chart...")
        # 只可视化最常见的25个标签，避免图表过于拥挤
        top_n_labels = 25
        plot_df = stats_df.head(top_n_labels)

        plt.figure(figsize=(12, 10))
        
        # 设置条形图的位置
        bar_width = 0.4
        index = range(len(plot_df))
        
        # 绘制条形图
        plt.bar(index, plot_df['OriginalCount'], bar_width, label='Original Count', color='skyblue')
        plt.bar([i + bar_width for i in index], plot_df['SampledCount'], bar_width, label=f'Sampled Count (Max {samples_per_label})', color='orange')
        
        # 设置图表样式
        plt.xlabel('Label', fontsize=12)
        plt.ylabel('Number of Records', fontsize=12)
        plt.title(f'Original vs. Sampled Record Counts (Top {top_n_labels} Labels)', fontsize=16)
        plt.xticks([i + bar_width / 2 for i in index], plot_df['Label'], rotation=90)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout() # 自动调整布局，防止标签重叠
        
        # 保存图表
        plot_path = f"{base_output_name}_stats.png"
        print(f"Saving chart to: {plot_path}")
        plt.savefig(plot_path, dpi=300)
        plt.close()

    except Exception as e:
        print(f"Error generating visualization: {e}")

    print("\n--- Processing finished successfully! ---")


# ====================================================================
# --- 主要执行区域 ---
# ====================================================================
if __name__ == '__main__':
    # --- 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(
        description="从音频标签CSV文件中，为每个标签选取指定数量的记录，并按标签整理输出。\n同时会自动生成统计数据和可视化图表。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--input_csv', 
        type=str, 
        help='输入的 consolidated_audio_tags.csv 文件路径。'
    )
    
    parser.add_argument(
        '--output_csv', 
        type=str, 
        default='sampled_by_label.csv',
        help='输出的CSV文件名。统计和图表文件将基于此命名。\n(默认: sampled_by_label.csv)'
    )
    
    parser.add_argument(
        '--num_samples', 
        type=int, 
        default=200,
        help='每个标签最多选取的记录数量。\n(默认: 200)'
    )
    
    args = parser.parse_args()

    # --- 调用主函数 ---
    filter_and_sample_by_label(
        input_csv_path=args.input_csv, 
        output_csv_path=args.output_csv, 
        samples_per_label=args.num_samples
    )

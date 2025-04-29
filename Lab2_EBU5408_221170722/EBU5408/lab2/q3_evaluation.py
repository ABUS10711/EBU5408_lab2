import os
import glob
import soundfile as sf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import pearsonr
from pathlib import Path

# Evaluation folder path
tuning_results_dir = "q3_tuning_results"
output_dir = "q3_evaluation"

# Create output directory
os.makedirs(output_dir, exist_ok=True)


# 计算两个信号之间的互信息
def compute_mutual_information(x, y, n_bins=50, subsample_factor=10):
    """
    使用直方图方法计算两个连续信号之间的互信息，优化计算效率。

    Parameters:
        x (numpy.ndarray): First signal
        y (numpy.ndarray): Second signal
        n_bins (int): Number of bins for histogram
        subsample_factor (int): Subsampling factor to reduce computation

    Returns:
        float: Mutual information value
    """
    x = x[::subsample_factor]
    y = y[::subsample_factor]

    try:
        x_binned = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform').fit_transform(x.reshape(-1, 1))
        y_binned = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform').fit_transform(y.reshape(-1, 1))
        mi = mutual_info_score(x_binned.ravel(), y_binned.ravel())
        return mi
    except Exception as e:
        print(f"Error computing mutual information: {e}")
        return 0.0


# 评估函数：计算互信息和相关系数
def evaluate_separation_results(separated_sources_dir, n_components):
    """
    Evaluate audio source separation quality using mutual information and correlation coefficient.

    Parameters:
        separated_sources_dir (Path): Directory path of separated signals
        n_components (int): Number of separated signals

    Returns:
        dict: Evaluation results including average mutual information and correlation
    """
    separated_files = sorted([f for f in separated_sources_dir.glob("source_*.wav")])

    if len(separated_files) != n_components:
        print(
            f"Warning: Number of separated signals ({len(separated_files)}) in {separated_sources_dir} does not match n_components ({n_components})")
        return None

    separated_sources = []
    for f in separated_files:
        try:
            s, sr = sf.read(f)
            separated_sources.append(s)
        except Exception as e:
            print(f"Error reading file {f}: {e}")
            return None

    separated_sources_np = np.array(separated_sources)

    # Calculate mutual information matrix
    mi_matrix = np.zeros((n_components, n_components))
    for i in range(n_components):
        for j in range(n_components):
            if i != j:
                mi_matrix[i, j] = compute_mutual_information(separated_sources_np[i], separated_sources_np[j])

    # Calculate correlation coefficient matrix
    corr_matrix = np.zeros((n_components, n_components))
    for i in range(n_components):
        for j in range(n_components):
            if i != j:
                corr_matrix[i, j] = abs(pearsonr(separated_sources_np[i], separated_sources_np[j])[0])

    avg_mi = np.mean(mi_matrix[mi_matrix != 0])
    avg_corr = np.mean(corr_matrix[corr_matrix != 0])

    return {
        "Average Mutual Information": avg_mi,
        "Average Correlation": avg_corr
    }


# 可视化函数：折线图
def visualize_results(df, output_dir):
    """
    Visualize the impact of each parameter on separation quality with one line plot per parameter,
    showing both avg_mi and avg_corr.

    Parameters:
        df (pd.DataFrame): DataFrame containing evaluation results
        output_dir (str): Path to save visualization results
    """
    sns.set(style="whitegrid", font_scale=1.2)

    # 参数列表
    params = ['n_components', 'max_iter', 'tol', 'bp_lo', 'bp_hi']

    # 为每个参数绘制折线图，包含 avg_mi 和 avg_corr
    for param in params:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # 左 y 轴：avg_mi
        sns.lineplot(x=param, y='avg_mi', data=df, marker='o', color='blue', ax=ax1, legend=False)
        ax1.set_xlabel(param.replace('_', ' ').title())
        ax1.set_ylabel('Average Mutual Information', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # 右 y 轴：avg_corr
        ax2 = ax1.twinx()
        sns.lineplot(x=param, y='avg_corr', data=df, marker='s', color='orange', ax=ax2, legend=False)
        ax2.set_ylabel('Average Correlation', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        # 手动创建图例
        blue_line = plt.Line2D([0], [0], color='blue', marker='o', linestyle='-', label='Avg MI')
        orange_line = plt.Line2D([0], [0], color='orange', marker='s', linestyle='-', label='Avg Corr')
        ax1.legend(handles=[blue_line, orange_line], loc='best')

        # 标题
        plt.title(f'Impact of {param.replace("_", " ").title()} on Separation Quality')

        # 保存
        save_path = os.path.join(output_dir, f'lineplot_{param}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Line plot saved to {save_path}")

# 主评估和可视化函数
def evaluate_and_visualize():
    """
    Evaluate all parameter combinations and visualize the impact of each parameter.
    """
    results_list = []

    # 处理标准参数组合
    standard_folder = Path(tuning_results_dir) / "standard"
    if standard_folder.exists():
        params = {
            'n_components': 3,
            'max_iter': 5,
            'tol': 1e-4,
            'bp_lo': 50,
            'bp_hi': 10000
        }
        print(f"Evaluating standard parameters: {standard_folder}")
        results = evaluate_separation_results(standard_folder, params['n_components'])
        if results:
            results_list.append({
                'n_components': params['n_components'],
                'max_iter': params['max_iter'],
                'tol': params['tol'],
                'bp_lo': params['bp_lo'],
                'bp_hi': params['bp_hi'],
                'avg_mi': results['Average Mutual Information'],
                'avg_corr': results['Average Correlation']
            })

    # 处理其他参数组合
    for param_folder in Path(tuning_results_dir).iterdir():
        if param_folder.name == 'standard' or not param_folder.is_dir():
            continue

        param_name = param_folder.name
        if param_name not in ['n_components', 'max_iter', 'tol', 'bp_lo', 'bp_hi']:
            print(f"Skipping unknown parameter folder: {param_folder}")
            continue

        for value_folder in param_folder.iterdir():
            if not value_folder.is_dir():
                continue

            try:
                value = value_folder.name
                params = {
                    'n_components': 3,
                    'max_iter': 5,
                    'tol': 1e-4,
                    'bp_lo': 50,
                    'bp_hi': 10000
                }
                if param_name == 'n_components':
                    params[param_name] = int(value)
                elif param_name in ['max_iter', 'bp_lo', 'bp_hi']:
                    params[param_name] = int(value)
                elif param_name == 'tol':
                    params[param_name] = float(value)

                print(f"Evaluating {param_name}={value}: {value_folder}")
                results = evaluate_separation_results(value_folder, params['n_components'])
                if results:
                    results_list.append({
                        'n_components': params['n_components'],
                        'max_iter': params['max_iter'],
                        'tol': params['tol'],
                        'bp_lo': params['bp_lo'],
                        'bp_hi': params['bp_hi'],
                        'avg_mi': results['Average Mutual Information'],
                        'avg_corr': results['Average Correlation']
                    })
            except Exception as e:
                print(f"Error processing folder {value_folder}: {e}")

    # 保存结果到 CSV
    if results_list:
        df = pd.DataFrame(results_list)
        # 确保参数列为适当的类型
        df['n_components'] = df['n_components'].astype(int)
        df['max_iter'] = df['max_iter'].astype(int)
        df['bp_lo'] = df['bp_lo'].astype(int)
        df['bp_hi'] = df['bp_hi'].astype(int)
        df['tol'] = df['tol'].astype(float)
        df.to_csv(os.path.join(output_dir, 'evaluation_results.csv'), index=False)
        print(f"Evaluation results saved to {os.path.join(output_dir, 'evaluation_results.csv')}")

        # 可视化
        visualize_results(df, output_dir)


# 调用主函数
if __name__ == "__main__":
    evaluate_and_visualize()

import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import pickle
import scipy.sparse.linalg as spla
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd  # 导入pandas库用于excel导出
import os  # 导入os库用于创建文件夹

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def main():
    """主函数，实现stPCA分析感染数据并绘制Z的标准差随时间变化图"""
    # 参数设置 - 优化计算效率
    win_m = 30          # 窗口大小（用户已调整）
    embeddings_num = 2  # 降维后的维度（从3减小到2，大幅减少H矩阵大小）
    a = 0.01            # alpha参数
    b = 1 - a           # beta参数
    window_step = 5     # 窗口步长，从2增加到5，进一步减少窗口数量
    
    # 设置数据维度 (用户已调整节点数为3000)
    node_num = 2000       # 保持3000个节点
    time_points_total = 700

    # 加载保存的模拟数据
    pickle_file = 'last_simulation_data_ws.pkl'
    print(f"正在加载模拟数据: {pickle_file}")

    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    # 提取数据
    last_sim_all_time_data = data['last_sim_all_time_data']
    
    # 检查数据形状并处理可能的维度问题
    print(f"原始数据形状: {last_sim_all_time_data.shape}")
    
    try:
        # 处理三维数据 (时间点, 节点数, 状态数)
        if len(last_sim_all_time_data.shape) == 3:
            print("数据是三维的 (时间点, 节点数, 状态数)")
            time_points, nodes, states = last_sim_all_time_data.shape
            print(f"时间点: {time_points}, 节点数: {nodes}, 状态数: {states}")
            
            # 限制时间点数量，避免处理过多数据
            if time_points > time_points_total:
                print(f"限制时间点数量从 {time_points} 到 {time_points_total}")
                last_sim_all_time_data = last_sim_all_time_data[:time_points_total, :, :]
                time_points = time_points_total
            
            # 提取I状态(索引2)的数据
            if states >= 3:
                print("提取I状态数据 (索引2)")
                I_data = last_sim_all_time_data[:, :, 2]  # 提取I状态
            else:
                print("状态数不足，使用最后一个状态")
                I_data = last_sim_all_time_data[:, :, -1]  # 使用最后一个状态
            
            # 计算每个时间点的总感染人数
            infection_counts = I_data.sum(axis=1)
            print(f"总感染人数数据形状: {infection_counts.shape}")
            
            # 为了适应stPCA算法，我们需要将数据重塑为 (节点数, 时间点)
            # 由于我们只有总感染人数，需要将其扩展为节点数行
            infection_counts = infection_counts.reshape(1, -1)  # 形状: (1, 时间点)
            infection_counts = np.repeat(infection_counts, node_num, axis=0)  # 形状: (节点数, 时间点)
            
            # 添加微小的随机扰动以避免数据完全相同
            infection_counts += np.random.normal(0, 0.001, infection_counts.shape)
            
        elif len(last_sim_all_time_data.shape) == 4:
            print("数据是四维的，使用第一个模拟数据")
            # 取第一个模拟数据
            last_sim_all_time_data = last_sim_all_time_data[0]
            print(f"处理后的三维数据形状: {last_sim_all_time_data.shape}")
            
            # 限制时间点数量
            if last_sim_all_time_data.shape[0] > time_points_total:
                last_sim_all_time_data = last_sim_all_time_data[:time_points_total, :, :]
            
            # 提取I状态(索引2)的数据
            I_data = last_sim_all_time_data[:, :, 2]  # 提取I状态
            
            # 计算每个时间点的总感染人数
            infection_counts = I_data.sum(axis=1)
            print(f"总感染人数数据形状: {infection_counts.shape}")
            
            # 重塑为 (节点数, 时间点)
            infection_counts = infection_counts.reshape(1, -1)  # 形状: (1, 时间点)
            infection_counts = np.repeat(infection_counts, node_num, axis=0)  # 形状: (节点数, 时间点)
            
        else:
            print(f"警告: 数据维度({len(last_sim_all_time_data.shape)})不符合预期")
            # 使用原始数据处理逻辑
            infection_counts = last_sim_all_time_data[:, :, -1].sum(axis=1).T
            # 限制时间点数量
            if infection_counts.shape[1] > time_points_total:
                infection_counts = infection_counts[:, :time_points_total]
            
        print(f"最终处理后的数据形状: {infection_counts.shape}")
    except Exception as e:
        print(f"数据处理出错: {e}")
        print("使用模拟数据生成器...")
        infection_counts = generate_simulation_data(node_num, time_points_total)
        print(f"模拟数据形状: {infection_counts.shape}")

    # 更新实际时间点总数
    time_points_total = infection_counts.shape[1]

    # 创建滑动窗口 - 增加步长以减少窗口数量
    print("创建滑动窗口...")
    num_windows = (time_points_total - win_m) // window_step + 1
    print(f"窗口数量: {num_windows} (步长: {window_step})")
    
    myzones = np.zeros((num_windows, win_m), dtype=int)
    for ni in tqdm(range(num_windows), desc="创建滑动窗口"):
        myzones[ni, :] = np.arange(ni * window_step, ni * window_step + win_m)
    
    # 初始化存储变量
    temp_var_flat_y = np.zeros(num_windows)
    
    # 对每个窗口进行处理
    print(f"开始处理数据窗口... 共{num_windows}个窗口，节点数={node_num}，嵌入维度={embeddings_num}")
    start_time = time.time()
    
    # 使用并行处理来加速窗口计算
    num_cores = multiprocessing.cpu_count() - 1  # 留一个核心给系统
    print(f"使用 {num_cores} 个CPU核心进行并行计算")
    
    # 创建任务列表
    tasks = [(infection_counts, myzones[ii, :], win_m, embeddings_num, a, b) for ii in range(num_windows)]
    
    # 并行处理窗口并显示进度条
    results = Parallel(n_jobs=num_cores)(
        delayed(process_window)(*task)
        for task in tqdm(tasks, desc="处理数据窗口", unit="窗口")
    )
    
    # 收集结果
    for ii, result in enumerate(results):
        temp_var_flat_y[ii] = result
    
    elapsed_time = time.time() - start_time
    print(f"数据处理完成，耗时: {elapsed_time:.2f}秒")
    
    # 绘制z的标准差随时间变化图
    print("正在绘制图表...")
    plot_z_std_over_time(temp_var_flat_y, num_windows, window_step)
    
    print("分析完成！图像已保存为 z_std_over_time.png")

def process_window(infection_counts, window_indices, win_m, embeddings_num, a, b):
    """处理单个窗口的数据 - 用于并行计算"""
    # 提取当前窗口的数据
    xx = infection_counts[:, window_indices]
    
    # 数据预处理
    traindata = xx - np.mean(xx, axis=1, keepdims=True)  # 去中心化
    X = traindata.T  # 转置数据格式
    
    # 获取维度信息
    m, n = X.shape  # m: 时间点数量, n: 节点数量
    L = embeddings_num  # 嵌入维度
    
    # 构造矩阵P和Q
    P = X[1:, :]  # 去掉第一行
    Q = X[:-1, :]  # 去掉最后一行
    
    # 求解Z
    H = construct_H_matrix(X, P, Q, n, L, a, b)
    
    # 特征值分解 - 只计算最大的几个特征值
    # 使用eigsh计算最大的10个特征值，这比计算所有特征值快得多
    eigenvalues, eigenvectors = spla.eigsh(H, k=10, which='LM')
    
    # 排序特征值和特征向量
    idx = eigenvalues.argsort()[::-1]  # 降序排列的索引
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 找到第一个正特征值
    ci = 0
    for i in range(len(eigenvalues)):
        if eigenvalues[i] > 0:
            ci = i
            break
    
    # 重构权重矩阵W和Z矩阵
    cW = eigenvectors[:, ci]
    W = cW.reshape(n, L)
    Z = X @ W * np.max(np.abs(eigenvalues))
    
    # 计算flat_z和其标准差
    flat_z, _ = calculate_flat_z(Z)
    
    # 返回当前窗口的结果
    return np.std(flat_z)

def generate_simulation_data(node_num, time_points_total):
    """生成模拟感染数据"""
    infection_counts = np.zeros((node_num, time_points_total))
    
    # 模拟感染过程的基本趋势
    for i in tqdm(range(node_num), desc="生成模拟数据"):
        # 每个节点有不同的感染起始时间和强度
        start_time = np.random.randint(1, 301)
        peak_time = start_time + np.random.randint(100, 301)
        peak_value = 0.8 + 0.2 * np.random.rand()
        
        # 创建感染曲线
        t = np.arange(time_points_total)
        infection_counts[i, :] = peak_value * np.exp(-((t - peak_time) / (0.3 * peak_time))**2) * (t >= start_time)
        
        # 添加随机波动
        infection_counts[i, :] += 0.02 * np.random.randn(time_points_total)
        infection_counts[i, infection_counts[i, :] < 0] = 0
    
    return infection_counts

def construct_H_matrix(X, P, Q, n, L, a, b):
    """构造H矩阵 - 优化计算效率"""
    # 预计算公共矩阵乘积以减少重复计算
    XTX = X.T @ X
    PTP = P.T @ P
    QTQ = Q.T @ Q
    PTQ = P.T @ Q
    QTP = Q.T @ P
    
    # 直接构建H矩阵的对角块和次对角块
    D1 = a * XTX - b * PTP
    D2 = a * XTX - b * (PTP + QTQ)
    DL = a * XTX - b * QTQ
    A = b * PTQ
    AT = b * QTP
    
    # 创建H矩阵
    H = np.zeros((n * L, n * L))
    
    # 填充第一块
    H[:n, :n] = D1
    if L > 1:
        H[:n, n:2*n] = A
    
    # 填充中间块
    for j in range(2, L):
        start_row = n * (j - 1)
        end_row = n * j
        start_col_prev = start_row - n
        end_col_next = end_row + n
        
        H[start_row:end_row, start_col_prev:start_row] = AT
        H[start_row:end_row, start_row:end_row] = D2
        
        if j < L:
            H[start_row:end_row, end_row:end_col_next] = A
    
    # 填充最后一块
    if L > 1:
        start_row_last = n * (L - 1)
        end_row_last = n * L
        start_col_prev_last = start_row_last - n
        
        H[start_row_last:end_row_last, start_col_prev_last:start_row_last] = AT
        H[start_row_last:end_row_last, start_row_last:end_row_last] = DL
    
    return H

def calculate_flat_z(Z):
    """计算flat_z及其标准差 - 优化版"""
    m, L = Z.shape
    flat_z = np.zeros(m)
    
    # 向量化计算，避免嵌套循环
    for zi in range(m):
        window_size = min(L, zi + 1)
        # 计算当前窗口内的平均值
        flat_z[zi] = Z[zi - window_size + 1:zi + 1, -window_size:].diagonal().mean()
    
    sd_flat_z = flat_z.std()
    return flat_z, sd_flat_z

def save_plot_data(temp_var_flat_y, num_windows, window_step, filename_prefix='stPCA_plot_DRC'):
    """保存绘图用的数据为pkl和excel文件
    
    参数:
        temp_var_flat_y: Z的标准差数据
        num_windows: 窗口数量
        window_step: 窗口步长
        filename_prefix: 输出文件名前缀
    """
    # 创建分数图文件夹
    excel_dir = '北京数据'
    os.makedirs(excel_dir, exist_ok=True)
    
    # 计算x轴时间点
    x = np.arange(1, num_windows + 1) * window_step
    
    # 找到最高分数的时间点位置
    max_idx = np.argmax(temp_var_flat_y)
    max_time_point = x[max_idx]
    max_score = temp_var_flat_y[max_idx]
    
    # 准备要保存的数据
    data_to_save = {
        'z_std_values': temp_var_flat_y,
        'time_points': x,
        'num_windows': num_windows,
        'window_step': window_step,
        'max_time_point': max_time_point,
        'max_score': max_score,
        'max_idx': max_idx
    }
    
    # 保存为pkl文件
    pkl_file = f'{filename_prefix}.pkl'
    with open(pkl_file, 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f"绘图数据已保存为pkl文件: {pkl_file}")
    
    # 保存为excel文件
    excel_file = os.path.join(excel_dir, f'{filename_prefix}.xlsx')
    
    # 创建DataFrame
    df_data = {
        '时间点': x,
        'Z的标准差': temp_var_flat_y,
        '最高分数标记': [1 if t == max_time_point else 0 for t in x]
    }
    
    df = pd.DataFrame(df_data)
    
    # 保存到excel
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='stPCA数据', index=False)
    
    print(f"绘图数据已保存为excel文件: {excel_file}")


def plot_z_std_over_time(temp_var_flat_y, num_windows, window_step):
    """绘制Z的标准差随时间变化图"""
    plt.figure(figsize=(12, 6))
    
    # 调整x轴以反映实际时间点（考虑步长）
    x = np.arange(1, num_windows + 1) * window_step
    plt.plot(x, temp_var_flat_y, '-b', linewidth=2)
    
    # 找到最高分数的时间点位置
    max_idx = np.argmax(temp_var_flat_y)
    max_time_point = x[max_idx]
    max_score = temp_var_flat_y[max_idx]
    
    # 添加垂直虚线标记最高分数位置
    plt.axvline(x=max_time_point, color='r', linestyle='--', linewidth=2, 
                label=f'最高分数: {max_score:.4f} (时间点: {max_time_point})')
    
    plt.title('Z的标准差随时间的变化', fontsize=18)
    plt.xlabel('时间点', fontsize=16)
    plt.ylabel('Z的标准差', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('z_std_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 调用保存数据函数
    save_plot_data(temp_var_flat_y, num_windows, window_step, filename_prefix='stPCA_北京')

if __name__ == "__main__":
    main()
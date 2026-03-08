import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import scipy.sparse as sp
from tqdm import tqdm  # 导入进度显示库
import pandas as pd  # 导入pandas库用于excel导出

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 加载seir_network_simulation.py生成的BA网络数据
def load_ba_network_data(pickle_file='last_simulation_data_ws.pkl', simulation_file='reference_samples_ws.txt'):
    """从pickle文件和模拟结果文件加载BA网络数据"""
    # 加载pickle文件中的网络结构和第一次模拟的完整时间序列数据
    if not os.path.exists(pickle_file):
        print(f"错误：找不到数据文件 {pickle_file}")
        return None, None, None, None, None, None
    
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    # 提取网络结构数据（只有物理层邻接矩阵）
    adjacency_matrix = data['adjacency_matrix']  # 物理层邻接矩阵
    node_num = data['node_num']
    
    # 提取第一次模拟的所有时间点数据（I状态，索引2）
    # 确保数据格式正确
    if 'last_sim_all_time_data' in data and data['last_sim_all_time_data'].ndim == 3:
        first_sim_data = data['last_sim_all_time_data'][:, :, 2]  # 形状: (time_steps, node_num)，I状态
        print(f"成功提取时间序列数据，形状: {first_sim_data.shape}")
    else:
        print("警告：pickle文件中的时间序列数据格式不正确，使用随机生成的数据")
        # 使用随机生成的时间序列数据作为备选
        time_steps = data.get('time_scale', 2000)  # 从数据中获取时间尺度，默认为2000
        first_sim_data = np.random.rand(time_steps, node_num)
    
    # 加载模拟结果文件中的数据
    if not os.path.exists(simulation_file):
        print(f"警告：找不到模拟结果文件 {simulation_file}")
        print("将使用第一次模拟的时间序列数据生成模拟结果")
        # 返回None表示没有找到模拟结果文件
        return adjacency_matrix, first_sim_data, None, node_num, data['parameters'], data.get('time_scale', 2000)
    
    # 读取模拟结果数据
    all_simulations_data = []
    with open(simulation_file, 'r') as f:
        lines = f.readlines()
        # 跳过表头
        for line in tqdm(lines[1:], desc="加载模拟结果数据"):
            parts = line.strip().split('\t')
            if len(parts) > 1:
                # 转换为浮点数
                node_data = [float(x) for x in parts[1:]]
                all_simulations_data.append(node_data)
    
    # 转换为numpy数组，形状: (node_num, simulations_num)
    all_simulations_data = np.array(all_simulations_data)
    
    print(f"成功加载BA网络数据：")
    print(f"- 节点数: {node_num}")
    print(f"- 物理层邻接矩阵: {type(adjacency_matrix)}, 非零元素数: {adjacency_matrix.count_nonzero() if hasattr(adjacency_matrix, 'count_nonzero') else np.count_nonzero(adjacency_matrix)}")
    print(f"- 第一次模拟时间步数: {first_sim_data.shape[0]}")
    print(f"- 模拟次数: {all_simulations_data.shape[1] if all_simulations_data.size > 0 else '无'}")
    
    return adjacency_matrix, first_sim_data, all_simulations_data, node_num, data['parameters'], data.get('time_scale', 2000)

# 将邻接矩阵转换为邻接表格式
def adjacency_matrix_to_list(adj_matrix, node_num):
    """将邻接矩阵转换为邻接表格式"""
    adj_list = []
    
    for i in tqdm(range(node_num), desc="构建邻接表"):
        # 找到与节点i相连的所有节点
        if hasattr(adj_matrix, 'tocoo'):
            # 对于稀疏矩阵
            coo_matrix = adj_matrix.tocoo()
            connected_nodes = coo_matrix.col[coo_matrix.row == i].tolist()
        else:
            # 对于普通矩阵
            connected_nodes = np.where(adj_matrix[i] > 0)[0].tolist()
        
        # 邻接表格式：[节点ID, 连接节点1, 连接节点2, ...]（节点ID从1开始）
        adj_list.append([i+1] + [node+1 for node in connected_nodes])
    
    return adj_list

# 生成模块ID
def generate_module_ids(node_num):
    """生成模块ID"""
    # 使用更合理的模块分配方式，基于节点度或其他网络特性
    # 这里使用简单的随机分配，实际应用中可以根据网络结构优化
    np.random.seed(42)  # 固定随机种子以确保结果可重复
    module_id = np.random.randint(1, 11, size=node_num)  # 随机分配1-10个模块
    
    return module_id

# 主要的NIG计算函数，使用模拟数据和时间序列
def calculate_NIG_with_multiple_simulations(adjacency_matrix, first_sim_data, all_simulations_data, node_num, module_id):
    """使用单层网络结构、模拟数据和时间序列计算网络信息增益(NIG)"""
    # 如果没有找到模拟结果文件，使用第一次模拟的时间序列数据生成模拟结果
    if all_simulations_data is None:
        print("使用第一次模拟的时间序列数据生成模拟结果")
        # 使用时间序列数据的不同时间点作为不同的模拟结果
        time_steps = first_sim_data.shape[0]  # 时间点数量
        # 选择35个时间点作为模拟结果（与seir_network_simulation.py中的模拟次数一致）
        if time_steps >= 35:
            selected_time_points = np.linspace(0, time_steps-1, 35, dtype=int)
            all_simulations_data = first_sim_data[selected_time_points, :].T
        else:
            # 如果时间点不足，重复使用时间点
            selected_time_points = np.arange(0, time_steps)
            repeated_indices = np.tile(selected_time_points, (35 // time_steps) + 1)[:35]
            all_simulations_data = first_sim_data[repeated_indices, :].T
        
        print(f"生成的模拟结果形状: {all_simulations_data.shape}")
    
    # 数据标准化
    pprofile = (all_simulations_data - np.mean(all_simulations_data, axis=1, keepdims=True)) / (np.std(all_simulations_data, axis=1, keepdims=True) + 1e-9)

    # 计算每个模块在每个样本中的表达均值和方差 - 向量化版本
    psize = pprofile.shape
    gene_mod_mean = np.zeros(psize)
    gene_mod_var = np.zeros(psize)
    
    print("计算模块均值和方差...")
    # 预计算所有模块的均值和方差
    unique_modules = np.unique(module_id)
    for mod in tqdm(unique_modules, desc="计算模块统计量"):
        indices = np.where(module_id == mod)[0]
        if len(indices) == 0:
            continue
        
        # 计算该模块在所有样本中的均值
        mod_mean = np.mean(pprofile[indices, :], axis=0, keepdims=True)  # 形状: (1, simulations_num)
        # 计算该模块在所有样本中的方差
        mod_var = np.var(pprofile[indices, :], axis=0, keepdims=True) + 1e-9  # 形状: (1, simulations_num)
        
        # 将结果赋值给属于该模块的所有节点
        gene_mod_mean[indices, :] = mod_mean
        gene_mod_var[indices, :] = mod_var
    
    # 计算每个网络节点的p(i,j)和网络流熵Hn(x) - 向量化版本
    print("计算p_numr和节点熵...")
    # 计算p_numr
    p_numr = np.exp(-np.abs(pprofile - gene_mod_mean) ** 2 / (2 * gene_mod_var))
    
    # 计算p_node
    row_sums = np.sum(p_numr, axis=1, keepdims=True) + 1e-9
    p_node = p_numr / row_sums
    
    # 计算熵节点 - 修正版本
    entropy_node = np.zeros(psize[0])
    # 创建掩码，防止log(0)
    mask = (pprofile * p_node) > 0
    masked_values = np.where(mask, pprofile * p_node * np.log(pprofile * p_node), 0)
    # 对每个节点求和（沿着样本轴）并取负数
    entropy_node = -np.sum(masked_values, axis=1)
    
    # 从邻接矩阵构建邻接表
    adj_list = adjacency_matrix_to_list(adjacency_matrix, node_num)

    # 计算权重：使用物理层连接数
    weight_sum = np.zeros(psize[0])
    for i in tqdm(range(psize[0]), desc="计算节点连接数"):
        # 物理层连接数
        connections = len(adj_list[i]) - 1
        weight_sum[i] = connections

    # 计算局部信息增益
    print("计算局部信息增益...")
    IG_node = np.zeros(psize[0])
    
    # 使用tqdm显示进度
    for i in tqdm(range(psize[0]), desc="计算节点信息增益"):
        if weight_sum[i] == 0:
            continue

        # 获取物理层邻居信息
        neighbors = adj_list[i][1:]  # 跳过第一个元素（节点本身）

        if not neighbors:
            continue

        # 确保邻居索引有效并转换为0-based
        valid_neighbors = [n-1 for n in neighbors if 0 <= n-1 < psize[0]]
        
        if not valid_neighbors:
            continue

        # 计算条件熵和信息增益 - 向量化版本
        IG_cond = np.abs(entropy_node[valid_neighbors] - entropy_node[i])
        IG_node[i] = np.mean(IG_cond)

    # 计算引入时间序列案例样本后的NIG
    if first_sim_data is not None:
        time_points = first_sim_data.shape[0]  # 时间点数量
        print(f"计算时间序列信息增益变化，共 {time_points} 个时间点...")

        # 标准化时间序列数据
        first_sim_data_std = (first_sim_data - np.mean(first_sim_data, axis=0, keepdims=True)) / (np.std(first_sim_data, axis=0, keepdims=True) + 1e-9)

        # 计算信息增益变化 - 部分向量化版本
        delt_IG = np.zeros((psize[0], time_points))
        
        # 预计算每个节点的模块信息
        mod_info = {i: {'module': module_id[i], 'indices': np.where(module_id == module_id[i])[0]} for i in range(psize[0])}
        
        for t in tqdm(range(time_points), desc="计算时间序列信息增益"):
            # 获取当前时间点的案例样本
            case_sample = first_sim_data_std[t, :]

            # 为每个节点计算信息增益变化 - 向量化版本
            for i in range(psize[0]):
                mod = mod_info[i]['module']
                indices = mod_info[i]['indices']
                
                # 计算案例样本的模块均值和方差
                case_mod_mean = np.mean(case_sample[indices])
                case_mod_var = np.var(case_sample[indices]) + 1e-9

                # 计算案例样本的熵
                if case_sample[i] > 0:  # 防止log(0)
                    case_entropy = -case_sample[i] * np.log(case_sample[i])
                else:
                    case_entropy = 0

                # 计算信息增益变化
                delt_IG[i, t] = np.abs(IG_node[i] - case_entropy)

        # 计算每个时间点的平均NIG值
        NIG = np.mean(delt_IG, axis=0)

        return np.arange(time_points), NIG, delt_IG
    else:
        # 如果没有时间序列数据，返回默认值
        print("警告：没有时间序列数据，无法计算NIG随时间变化")
        s = np.linspace(0, 10, 10)
        NIG = np.zeros(10)
        delt_IG = np.zeros((psize[0], 10))
        return s, NIG, delt_IG

# 保存NIG绘图数据为pkl和excel文件
def save_NIG_plot_data(s, NIG, delt_IG, excel_dir='分数图'):
    """保存用于绘图的NIG数据为pkl和excel文件
    
    参数:
        s: 时间点数组
        NIG: 每个时间点的平均NIG值
        delt_IG: 每个节点在每个时间点的信息增益变化
        excel_dir: excel文件保存目录
    """
    # 创建分数图文件夹（如果不存在）
    os.makedirs(excel_dir, exist_ok=True)
    
    # 保存为pkl文件
    pkl_file_path = 'NIG.pkl'
    data_to_save = {
        'time_points': s,
        'average_NIG': NIG,
        'node_NIG_changes': delt_IG
    }
    with open(pkl_file_path, 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f"NIG绘图数据已保存为pkl文件: {pkl_file_path}")
    
    # 保存为excel文件
    excel_file_path = os.path.join(excel_dir, 'NIG.xlsx')
    
    # 创建包含时间点和平均NIG值的DataFrame
    df_avg = pd.DataFrame({
        '时间点': s,
        '平均NIG值': NIG
    })
    
    # 创建包含节点NIG变化的DataFrame
    df_nodes = pd.DataFrame(delt_IG.T, columns=[f'节点{i}' for i in range(delt_IG.shape[0])])
    df_nodes.insert(0, '时间点', s)  # 在第一列插入时间点
    
    # 使用ExcelWriter保存多个工作表
    with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
        df_avg.to_excel(writer, sheet_name='平均NIG值', index=False)
        df_nodes.to_excel(writer, sheet_name='节点NIG变化', index=False)
    
    print(f"NIG绘图数据已保存为excel文件: {excel_file_path}")

# 绘制结果图
def plot_results(s, NIG, delt_IG):
    """绘制基于三层网络、40次模拟和时间序列的NIG分析结果图"""
    # 绘制NIG曲线
    plt.figure(figsize=(10, 6))
    plt.plot(s[:len(NIG)], NIG, 'r-*', linewidth=2.5)
    
    # 标记特定时间点（NIG峰值附近）
    peak_idx = np.argmax(NIG)
    if peak_idx >= 0 and peak_idx < len(s):
        plt.axvline(x=s[peak_idx], linestyle='--', color='g', linewidth=1.5, label=f'峰值位置: t={s[peak_idx]}')
        plt.legend(fontsize=20)
    
    # 调整x轴范围
    plt.xlim(min(s) - 5, max(s) + 5)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 根据参数类型设置标题和标签
    if len(s) > 100:  # 如果是时间序列数据
        plt.title('基于时间序列的三层网络NIG分析结果')
        plt.xlabel('时间点')
    else:
        plt.title('三层网络NIG分析结果')
        plt.xlabel('参数')
    
    plt.ylabel('网络信息增益 (NIG)')
    plt.tight_layout()
    plt.savefig('NIG_simulation_result.png', dpi=300)
    plt.show()
    
    # 如果节点数过多，采样部分节点进行3D可视化，避免图表过于拥挤
    nodes_num, params_num = delt_IG.shape
    sample_ratio = min(1.0, 200 / nodes_num)  # 最多显示200个节点
    
    if sample_ratio < 1.0:
        # 随机采样节点
        np.random.seed(42)  # 固定种子以确保结果可重复
        sample_indices = np.random.choice(nodes_num, int(nodes_num * sample_ratio), replace=False)
        sample_delt_IG = delt_IG[sample_indices, :]
        print(f"为3D可视化采样了{len(sample_indices)}个节点（总节点数: {nodes_num}）")
    else:
        sample_delt_IG = delt_IG
        sample_indices = np.arange(nodes_num)
    
    # 绘制3D条形图
    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(111, projection='3d')
    
    # 获取采样后的数据尺寸
    # sample_nodes_num, sample_params_num = sample_delt_IG.shape
    #
    # 生成x, y数据
    # x_data, y_data = np.meshgrid(np.arange(sample_params_num), np.arange(sample_nodes_num))
    # x_data = x_data.flatten()
    # y_data = y_data.flatten()
    # z_data = np.zeros_like(x_data)
    
    # 条形图的宽度和深度
    # width = depth = 0.7
    
    # 绘制3D条形图
    # bars = ax.bar3d(x_data, y_data, z_data, width, depth, sample_delt_IG.flatten(), shade=True)
    
    # 根据参数类型设置坐标轴标签
    # if len(s) > 100:  # 如果是时间序列数据
    #     ax.set_xlabel('时间点')
    #     plt.title('基于时间序列的三层网络节点信息增益分布')
    # else:
    #     ax.set_xlabel('参数')
    #     plt.title('三层网络节点信息增益分布')
    #
    # ax.set_ylabel('网络节点')
    # ax.set_zlabel('节点信息增益变化')
    
    # 设置刻度标签
    # step = max(1, int(sample_params_num / 6))  # 确保不超过6个刻度标签
    # ax.set_xticks(np.arange(0, sample_params_num, step))
    
    # 根据参数类型设置刻度标签格式
    # if len(s) > 100:  # 如果是时间序列数据
    #     ax.set_xticklabels([f'{int(s[i])}' for i in np.arange(0, params_num, step)])
    # else:
    #     ax.set_xticklabels([f'{s[i]:.2f}' for i in np.arange(0, params_num, step)])
    #
    # # 调整视角以获得更好的可视化效果
    # ax.view_init(elev=30, azim=45)
    # 
    # plt.tight_layout()
    # plt.savefig('NIG_3D_distribution.png', dpi=300)
    # plt.show()


# 获取NIG分数最高的时间点中分数最高的k个节点
def get_top_k_nodes(s, NIG, delt_IG, k=10, output_file='NIG_nodes.txt', pkl_file='NIG_nodes.pkl'):
    """获取NIG分数最高的时间点中，分数最高的k个节点

    参数:
        s: 时间点数组
        NIG: 每个时间点的平均NIG值
        delt_IG: 每个节点在每个时间点的信息增益变化
        k: 需要返回的节点数量
        output_file: txt输出文件路径
        pkl_file: pkl输出文件路径

    返回:
        top_time_point: 最高NIG分数的时间点
        top_nodes: 分数最高的k个节点ID列表
        top_scores: 对应的NIG分数列表
    """
    # 找到NIG分数最高的时间点
    peak_idx = np.argmax(NIG)
    top_time_point = s[peak_idx]

    # 获取该时间点所有节点的分数
    node_scores = delt_IG[:, peak_idx]

    # 获取分数最高的k个节点的索引
    top_k_indices = np.argsort(node_scores)[::-1][:k]

    # 获取对应的节点ID和分数
    top_nodes = top_k_indices.tolist()
    top_scores = node_scores[top_k_indices].tolist()

    print(f"\n在NIG分数最高的时间点 t={top_time_point}，分数最高的{k}个节点:")
    print("节点ID\tNIG分数")
    print("-" * 20)

    # 准备保存到文件的内容
    file_content = []
    file_content.append(f"NIG分数最高的时间点: t={top_time_point}")
    file_content.append(f"分数最高的{k}个节点:")
    file_content.append("节点ID\tNIG分数")
    file_content.append("-" * 20)

    for node_id, score in zip(top_nodes, top_scores):
        print(f"{node_id}\t{score:.6f}")
        file_content.append(f"{node_id}\t{score:.6f}")

    # 保存到txt文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(file_content))

    print(f"\n节点信息已保存到txt文件: {output_file}")

    # 保存到pkl文件
    pkl_data = {
        'top_time_point': top_time_point,
        'vital_nodes': top_nodes,
        'top_scores': top_scores,
        'k_value': k
    }

    with open(pkl_file, 'wb') as f:
        pickle.dump(pkl_data, f)

    print(f"节点信息已保存到pkl文件: {pkl_file}")

    return top_time_point, top_nodes, top_scores

def main():
    """主函数"""
    print("开始NIG计算")
    
    # 1. 加载BA网络数据
    adjacency_matrix, first_sim_data, all_simulations_data, node_num, params, time_scale = load_ba_network_data()
    
    if adjacency_matrix is None:
        print("无法加载网络数据，程序终止")
        return
    
    # 2. 生成模块ID
    print("生成模块ID...")
    module_id = generate_module_ids(node_num)
    
    # 3. 计算NIG
    s, NIG, delt_IG = calculate_NIG_with_multiple_simulations(adjacency_matrix, first_sim_data, all_simulations_data, node_num, module_id)
    
    # 4. 显示和保存结果
    print(f"NIG计算完成")
    print(f"最大NIG值: {np.max(NIG):.4f}")
    print(f"最小NIG值: {np.min(NIG):.4f}")
    print(f"平均NIG值: {np.mean(NIG):.4f}")
    
    # 5. 可视化结果
    print("生成可视化结果...")
    plot_results(s, NIG, delt_IG)
    
    # 6. 保存NIG绘图数据为pkl和excel文件
    print("保存NIG绘图数据...")
    save_NIG_plot_data(s, NIG, delt_IG, excel_dir='数据')
    
    # 7. 获取并保存top-k节点
    get_top_k_nodes(s, NIG, delt_IG, k=20, output_file='NIG_ba_nodes.txt', pkl_file='NIG_ba_nodes.pkl')
    
    print("NIG计算和结果保存完成")

if __name__ == "__main__":
    main()
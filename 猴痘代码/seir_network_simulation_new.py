import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pickle
import scipy.sparse as sp
import networkx as nx
import random
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd  # 导入pandas库用于Excel文件处理

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# -------------------参数设置---------------------------
# SEIR模型参数，beta待改变
beta, sigma, gamma, mu = 1.696e-1, 0.20, 0.15, 0.0  # 使用network_new.py中的参数

# 拓扑连接参数
node_num = 2000  # 保持原程序的网络规模
network_types = ['ws']  # 使用Watts-Strogatz小世界网络

# 模拟时间尺度
time_scale = 700  # 与network_new.py保持一致

# 每种网络结构的模拟次数
simulations_per_network = 35  # 保持原程序的模拟次数

# ---------------------网络生成函数---------------------------
def generate_graph(node_num, graph_type, seed=123):
    """生成不同类型的网络拓扑，替换为Watts-Strogatz小世界网络"""
    random.seed(seed)
    np.random.seed(seed)
    
    if graph_type == 'ws':
        k = 6  # 每个节点的近邻数量
        p = 0.1  # 重连概率
        G = nx.watts_strogatz_graph(node_num, k, p, seed=seed)
        
    else:
        raise ValueError("不支持的网络类型。请选择'ws'。")
    
    # 转换为邻接矩阵
    adj_matrix = nx.to_numpy_array(G)
    
    return adj_matrix

# ---------------------SEIR模型实现---------------------------
class SEIRModel:
    """基于network_new.py的SEIR模型实现，包含平均场计算"""
    def __init__(self, adjacency_matrix, beta, sigma, gamma):
        self.A = adjacency_matrix
        self.N = self.A.shape[0]  # Number of nodes

        # Calculate average degree <k> of the network
        avg_degree = np.mean(np.sum(self.A, axis=1))
        shift_degree = np.mean(np.sum(self.A, axis=1) ** 2) / avg_degree
        self.beta_tilde = beta
        self.beta = beta / shift_degree
        self.sigma = sigma
        self.gamma = gamma
        self.solution = None
        self.t = None
        self.mean_field_solution = None
        self.t_tilde = None

    def _dynamics(self, t, y):
        # Unpack the state vector
        S = y[0:self.N]
        E = y[self.N:2 * self.N]
        I = y[2 * self.N:3 * self.N]

        # Calculate the infection force for each node: sum_j(a_ji * p_j^I)
        neighbor_infection = self.A.T @ I

        # Differential equations
        dSdt = -S * neighbor_infection * self.beta
        dEdt = S * neighbor_infection * self.beta - self.sigma * E
        dIdt = self.sigma * E - self.gamma * I
        dRdt = self.gamma * I

        return np.concatenate([dSdt, dEdt, dIdt, dRdt])
    
    def seir_mean_field_dynamics(self, t, y):
        """
        Deterministic SEIR Compartmental Model (Mean Field Approximation).
        dS/dt = -beta * S * I
        dE/dt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dR/dt = gamma * I
        """
        S, E, I, R = y
        dSdt = -self.beta_tilde * S * I
        dEdt = self.beta_tilde * S * I - self.sigma * E
        dIdt = self.sigma * E - self.gamma * I
        dRdt = self.gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    def run(self, EI0, t_max=100, dt=0.1):
        # Initial conditions
        E0 = EI0[0]
        I0 = EI0[1]
        R0 = 0
        S0 = 1 - E0 - I0 - R0
        y0 = np.concatenate([
            np.full(self.N, S0),  # Susceptible
            np.full(self.N, E0),  # Exposed
            np.full(self.N, I0),  # Infected
            np.full(self.N, R0)  # Recovered
        ])
        # Time grid
        self.t = np.linspace(0, t_max, int(t_max / dt) + 1)
        self.solution = solve_ivp(self._dynamics, (0, t_max), y0, t_eval=self.t, method='RK45', atol=1e-6,
                                  rtol=1e-6).y.T

        return self.t, self.solution
    
    def solve_mean_field(self, EI0, t_max=100, dt=0.1):
        """
        Solves the mean-field SEIR equations.
        """
        E0 = EI0[0]
        I0 = EI0[1]
        R0 = 0
        S0 = 1 - E0 - I0 - R0
        y0 = [S0, E0, I0, R0]
        self.t_tilde = np.linspace(0, t_max, int(t_max / dt) + 1)
        self.mean_field_solution = solve_ivp(self.seir_mean_field_dynamics, (0, t_max), y0, t_eval=self.t_tilde,
                                             method='RK45', atol=1e-6, rtol=1e-6).y.T
        return self.t_tilde, self.mean_field_solution

    def get_average_states(self):
        if self.solution is None:
            raise ValueError("Model has not been run yet. Call .run() first.")

        sol_reshaped = self.solution.reshape((len(self.t), 4, self.N))
        S_avg = np.mean(sol_reshaped[:, 0, :], axis=1)
        E_avg = np.mean(sol_reshaped[:, 1, :], axis=1)
        I_avg = np.mean(sol_reshaped[:, 2, :], axis=1)
        R_avg = np.mean(sol_reshaped[:, 3, :], axis=1)

        return S_avg, E_avg, I_avg, R_avg

# ---------------------主模拟函数---------------------------
def run_simulation(network_type):
    """对特定网络类型运行模拟，保持原程序的输出格式"""
    print(f"\n开始对 {network_type} 网络进行 {simulations_per_network} 次模拟...")
    
    # 输出文件路径
    output_file = f'reference_samples_{network_type}.txt'
    first_sim_output_file = f'stage_i_{network_type}.txt'
    
    # 初始化存储所有模拟结果的矩阵
    all_results = np.zeros((node_num, simulations_per_network))
    
    # 存储第一次模拟的初始感染节点
    first_sim_infected_nodes = None
    
    for sim in range(simulations_per_network):
        print(f"\n模拟 {sim+1}/{simulations_per_network} 开始...")
        
        # ---------------------数据初始化---------------------------
        # 生成网络拓扑
        adj_matrix = generate_graph(node_num, network_type, seed=sim+3)
        
        # 初始条件，不同国家参数不同
        E0 = 0.0009982475
        I0 = 0.003010708

        EI0 = (E0, I0)
        
        # ---------------------准备模拟---------------------------
        # 初始化SEIR模型
        model = SEIRModel(
            adjacency_matrix=adj_matrix,
            beta=beta,
            sigma=sigma,
            gamma=gamma
        )
        
        # ---------------------运行模拟---------------------------
        print(f"开始模拟感染传播，时间尺度: {time_scale}...")
        t, solution = model.run(EI0=EI0, t_max=time_scale, dt=1)
        # 计算平均场解
        model.solve_mean_field(EI0=EI0, t_max=time_scale, dt=0.1)
        print(f"模拟完成! solution形状: {solution.shape}")
        
        # ---------------------提取感病人数并找到峰值时间点---------------------------
        # 提取I状态
        I_states = solution[:, 2*node_num:3*node_num]
        infection_counts = np.sum(I_states, axis=1)
        
        # 找到感染人数最高值的时间点
        peak_time = np.argmax(infection_counts)
        
        # 计算最高值2%的时间范围
        peak_value = infection_counts[peak_time]
        threshold = peak_value * 0.05
        
        # 找到所有在峰值2%范围内的时间点
        valid_times = np.where(np.abs(infection_counts) >= threshold)[0]
        
        # 根据valid_times的长度选择合适的时间点
        if len(valid_times) > 1:
            selected_time_idx = valid_times[1]  # 如果有多个时间点，选择第二个
        elif len(valid_times) > 0:
            selected_time_idx = valid_times[0]  # 如果只有一个时间点，选择第一个
        else:
            selected_time_idx = peak_time  # 如果没有符合条件的时间点，使用峰值时间

        if selected_time_idx == 1:
            selected_time_idx = random.randint(10, 20)
        
        print(f"模拟 {sim+1}: 峰值时间 = {peak_time}, 选择的时间点 = {selected_time_idx}")
        
        # 确保selected_time_idx在有效范围内
        if selected_time_idx >= solution.shape[0]:
            selected_time_idx = solution.shape[0] - 1
        
        # ---------------------存储模拟结果---------------------------
        # 提取所选时间点的所有节点数据
        time_data = solution[selected_time_idx].reshape(4, node_num).T
        
        # 存储到结果矩阵中 (I状态)
        for node in range(node_num):
            all_results[node, sim] = time_data[node][2]  # I状态是索引2
        
        # 保存第一次模拟的所有时间点数据
        if sim == 0:
            first_sim_all_time_data = solution.reshape(-1, 4, node_num).transpose(0, 2, 1)
            first_sim_infected_nodes = None  # network_new.py中没有特定的初始感染节点设置
        
        print(f"模拟 {sim+1} 的数据已存储")
    
    # ---------------------将所有结果写入文件---------------------------
    with open(output_file, 'w') as f:
        # 第一行：表头
        header = ['ID']
        for s in range(simulations_per_network):
            header.append(f'模拟{s+1}')
        f.write('\t'.join(header) + '\n')
        
        # 写入每个节点的数据
        for node in range(node_num):
            row = [f'{node}']
            for s in range(simulations_per_network):
                row.append(f'{all_results[node, s]:.4e}')
            f.write('\t'.join(row) + '\n')
    
    # ---------------------将第一次模拟的所有时间点数据写入文件---------------------------
    with open(first_sim_output_file, 'w') as f:
        # 获取实际的时间点数量
        actual_time_points = first_sim_all_time_data.shape[0]
        
        # 第一行：表头（时间点）
        header = ['时间点\\节点ID']
        for time_point in range(actual_time_points):
            header.append(f'时间点{time_point}')
        f.write('\t'.join(header) + '\n')
        
        # 写入每个节点的数据
        for node in range(node_num):
            row = [f'{node}']
            for time_point in range(actual_time_points):
                i_prob = first_sim_all_time_data[time_point, node, 2]  # I状态
                row.append(f'{i_prob:.4e}')
            f.write('\t'.join(row) + '\n')
    
    # ---------------------保存为pickle文件---------------------------
    first_sim_pickle_file = f'last_simulation_data_{network_type}.pkl'
    with open(first_sim_pickle_file, 'wb') as f:
        pickle.dump({
            'last_sim_all_time_data': first_sim_all_time_data,
            'adjacency_matrix': adj_matrix,
            'node_num': node_num,
            'time_scale': time_scale,
            'initial_infected_nodes': first_sim_infected_nodes,
            'parameters': {
                'beta': beta, 'sigma': sigma, 'gamma': gamma, 'mu': mu
            }
        }, f)
    
    print(f"第一次模拟的数据已保存为pickle格式到 {first_sim_pickle_file}")
    
    # ---------------------绘制第一次模拟的感病人数随时间变化图---------------------------
    # 提取I状态并求和
    infection_counts = np.sum(first_sim_all_time_data[:, :, 2], axis=1)
    
    time_points = range(len(infection_counts))
    
    # 提取平均场的I状态
    if hasattr(model, 'mean_field_solution') and model.mean_field_solution is not None:
        S_mf, E_mf, I_mf, R_mf = model.mean_field_solution.T
    
    # 绘制图像
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, infection_counts/node_num, '-b', linewidth=2, label='网络模拟')
    if hasattr(model, 'mean_field_solution') and model.mean_field_solution is not None:
        plt.plot(model.t_tilde, I_mf, '--r', linewidth=2, label='平均场近似')
    plt.title(f'{network_type} 网络 - 感病人数随时间的变化', fontsize=16, fontweight='bold')
    plt.xlabel('时间', fontsize=14)
    plt.ylabel('感病人数', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(f'infection_total_{network_type}.png', dpi=300)
    plt.close()
    
    # ---------------------将网络模拟和平均场数据导出为Excel---------------------------
    # 创建输出文件夹
    output_dir = "数据"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建文件夹: {output_dir}")
    
    # 1. 导出网络模拟数据
    print("导出网络模拟数据到Excel...")
    
    # 计算网络模拟的sigma*E数据（除以node_num以匹配平均场的比例形式）
    network_sigma_E = sigma * np.sum(first_sim_all_time_data[:, :, 1], axis=1) / node_num
    network_time = range(len(network_sigma_E))
    
    # 创建网络模拟数据的DataFrame
    network_df = pd.DataFrame({
        '时间点': network_time,
        '网络模拟sigma*E': network_sigma_E
    })
    
    # 导出到Excel
    network_excel_file = os.path.join(output_dir, "网络模拟.xlsx")
    network_df.to_excel(network_excel_file, index=False)
    print(f"网络模拟数据已保存到: {network_excel_file}")
    
    # 2. 导出平均场数据
    if hasattr(model, 'mean_field_solution') and model.mean_field_solution is not None:
        print("导出平均场数据到Excel...")
        
        S_mf, E_mf, I_mf, R_mf = model.mean_field_solution.T
        
        # 创建平均场数据的DataFrame
        # 计算平均场的sigma*E数据
        mf_sigma_E = sigma * E_mf
        
        mf_df = pd.DataFrame({
            '时间点': model.t_tilde,
            '平均场sigma*E': mf_sigma_E
        })
        
        # 导出到Excel
        mf_excel_file = os.path.join(output_dir, "平均场.xlsx")
        mf_df.to_excel(mf_excel_file, index=False)
        print(f"平均场数据已保存到: {mf_excel_file}")
    
    print(f"\n所有 {simulations_per_network} 次模拟完成，数据已保存到 {output_file}")
    print(f"第一次模拟的所有时间点数据已保存到 {first_sim_output_file}")

# ---------------------主函数---------------------------
def main():
    print("=" * 60)
    print("SEIR模型网络疾病传播模拟程序 (基于network_new.py修改)")
    print(f"固定参数: beta={beta}, sigma={sigma}, gamma={gamma}, mu={mu}")
    print(f"网络规模: {node_num}")
    print(f"每种网络模拟次数: {simulations_per_network}")
    print("=" * 60)
    
    # 对每种网络类型运行模拟
    for network_type in network_types:
        run_simulation(network_type)
    
    print("\n" + "=" * 60)
    print("所有网络类型的模拟已完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
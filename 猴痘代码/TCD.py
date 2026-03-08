import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import iqr
from joblib import Parallel, delayed
import os
import warnings
from tqdm import tqdm
import pandas as pd  # 导入pandas库用于excel导出
# 关闭所有类型的警告，特别是scikit-learn的收敛警告
warnings.filterwarnings("ignore")
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def outlieromit(predictions, threshold=1.5):
    """
    基于IQR方法去除异常值
    """
    q1 = np.percentile(predictions, 25)
    q3 = np.percentile(predictions, 75)
    iqr_val = q3 - q1
    lower_bound = q1 - threshold * iqr_val
    upper_bound = q3 + threshold * iqr_val

    filtered_preds = predictions[(predictions >= lower_bound) & (predictions <= upper_bound)]
    return filtered_preds if len(filtered_preds) > 0 else predictions


def myprediction_gp(X_train, y_train, X_test):
    """
    高斯过程预测函数
    """
    # 在每个并行进程中都添加警告过滤
    import warnings
    warnings.filterwarnings("ignore")
    
    # 设置高斯过程核函数 - 简化核函数以提高速度
    kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-1, 1e1))

    try:
        # 创建并拟合高斯过程模型 - 进一步减少优化重启次数
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, random_state=42)
        gp.fit(X_train.T, y_train)  # 注意：需要转置以匹配sklearn的输入格式

        # 预测
        y_pred, sigma = gp.predict(X_test.reshape(1, -1), return_std=True)
        return y_pred[0]
    except:
        # 出现错误时返回训练数据的均值作为备选
        return np.mean(y_train)


def main():
    # 1. 加载seir_network_simulation生成的BA网络数据
    pickle_file = 'last_simulation_data_ws.pkl'  # BA网络模拟数据文件
    
    try:
        with open(pickle_file, 'rb') as f:
            simulation_data = pickle.load(f)
        print(f"成功加载BA网络模拟数据：{pickle_file}")
    except FileNotFoundError:
        print(f"错误：找不到文件 {pickle_file}")
        print("请确保已运行seir_network_simulation.py生成BA网络数据")
        return

    # 2. 提取数据（根据BA网络数据结构调整）
    # 假设数据结构：simulation_data是一个包含节点状态的三维数组
    # 维度通常是 [时间步, 节点数, 状态数]
    # 状态通常包括：S=0, E=1, I=2, R=3, D=4
    
    # 检查数据结构
    if not isinstance(simulation_data, np.ndarray):
        print("警告：模拟数据不是预期的numpy数组格式")
        # 尝试从字典结构中提取数据
        if isinstance(simulation_data, dict):
            if 'node_states' in simulation_data:
                simulation_data = simulation_data['node_states']
            elif 'last_sim_all_time_data' in simulation_data:
                simulation_data = simulation_data['last_sim_all_time_data']
            else:
                print("错误：无法从字典中提取节点状态数据")
                return
        else:
            print(f"错误：模拟数据类型为 {type(simulation_data)}，无法处理")
            return

    print(f"数据形状：{simulation_data.shape}")
    print(f"时间步数：{simulation_data.shape[0]}")
    print(f"节点数：{simulation_data.shape[1]}")
    print(f"状态数：{simulation_data.shape[2]}")

    # 3. 数据预处理
    # 根据数据结构选择合适的数据范围
    # 假设我们需要I状态（感染状态）的数据
    state_index = 2  # I状态的索引
    max_time_steps = simulation_data.shape[0]  # 使用全部可用时间步
    node_num = simulation_data.shape[1]  # 节点总数
    
    # 提取指定时间步的I状态数据
    Y = simulation_data[0:max_time_steps, :, state_index]  # Y.shape = [时间步, 节点数]

    # 4. 计算Y1并绘图（所有节点在每个时间步的I状态总和）
    Y1 = np.sum(Y, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_time_steps + 1), Y1, '-o')
    plt.title('Sum of Y over dimension 2 (BA Network)')
    plt.xlabel('Time step')
    plt.ylabel('Sum value')
    plt.grid(True)
    plt.savefig('sum_plot_ba.png')
    plt.close()

    # 5. 设置参数
    noisestrength = 0 * 10 ** (-4)  # 外部噪声
    
    # 预测核心参数 - 调整参数以提高速度
    trainlength = 30  # 训练数据长度
    available_steps = max_time_steps - trainlength - 1  # 确保trainlength + step < max_time_steps
    maxstep = min(700, max(0, available_steps))  # 预测步数，不超过可用时间步，确保非负
    
    # 目标变量索引 - 确保不超过节点数量
    j = min(5, node_num - 1)  # 目标节点索引
    
    s = 100  # 进一步减少随机嵌入数量以提高速度
    L = 3  # 嵌入维度
    timelag = 0  # 时间偏移
    
    print(f"时间步总数：{max_time_steps}")
    print(f"节点总数：{node_num}")
    print(f"训练长度：{trainlength}")
    print(f"实际预测步数：{maxstep}")
    print(f"目标节点索引：{j}")

    # 6. 准备数据
    # Y的维度是[时间步, 节点数]，我们需要在时间点上对感病节点数据进行预测
    # 所以我们的输入是：使用前trainlength个时间步的数据来预测下一个时间步的数据
    xx = Y[timelag:, :]  # xx.shape = [时间步, 节点数]
    print(f"xx数组形状：{xx.shape}")

    # 7. 存储结果
    result = np.zeros((3, maxstep))  # 预测值、标准差、误差

    # 预测循环
    print("开始预测过程...")
    D = xx.shape[1]  # 系统变量数（节点数）
    total_predictions = maxstep * s
    
    # 使用tqdm创建进度条
    with tqdm(total=maxstep, desc="TCD预测进度", unit="步") as pbar:
        for step in range(maxstep):
            # 提取当前训练窗口数据
            # 使用前trainlength个时间步的数据来预测下一个时间步的数据
            traindata = xx[step:step + trainlength, :]  # traindata.shape = [trainlength, 节点数]
            real_value = xx[step + trainlength, j]  # 真实值：下一个时间步的目标节点值

            # 生成s个随机的L维组合
            B = np.zeros((s, L), dtype=int)
            for i in range(s):
                B[i, :] = np.random.permutation(D)[:L]

            # 并行预测函数 - 移除计数逻辑以提高效率
            def predict_for_combination(i):
                combo = B[i, :]
                return myprediction_gp(
                    traindata[:-1, combo],  # 输入特征（训练）：前trainlength-1个时间步的L个节点
                    traindata[1:, j],  # 目标输出（训练）：后trainlength-1个时间步的目标节点
                    traindata[-1, combo]  # 待预测输入：最后一个时间步的L个节点
                )

            # 使用joblib进行并行计算，增加verbose参数查看并行信息
            predictions = Parallel(n_jobs=-1, verbose=0)(
                delayed(predict_for_combination)(i) for i in range(s)
            )
            predictions = np.array(predictions)

            # 去除异常值
            pp = outlieromit(predictions)

            # 计算预测分布的期望和标准差
            if len(pp) > 0:
                prediction = np.mean(pp)
                std_val = np.std(pp)
            else:
                prediction = np.mean(predictions)
                std_val = np.std(predictions)

            # 存储结果
            result[0, step] = prediction  # 预测值
            result[1, step] = std_val  # 预测标准差
            result[2, step] = real_value - prediction  # 预测误差

            # 更新进度条
            pbar.update(1)

    print(f"\n预测完成！总预测次数: {total_predictions}")

    print("预测完成！")

    # 9. 计算阈值
    std_values = result[1, :]
    threshold = np.mean(std_values) + 2 * np.std(std_values)

    # 10. 识别变化点
    change_points = np.where(std_values > threshold)[0]
    print('识别的网络结构变化点位置（时间步）：')
    print(change_points + trainlength + 1)  # 调整为原始时间尺度

    # 添加：保存绘图数据为pkl和excel文件
    def save_plot_data():
        """保存绘图用的数据为pkl和excel文件"""
        # 创建分数图文件夹
        excel_dir = '北京数据'
        os.makedirs(excel_dir, exist_ok=True)
        
        # 准备要保存的数据
        data_to_save = {
            'time_range': time_range,
            'real_data': xx[:trainlength + maxstep, j],
            'predicted_data': result[0, :],
            'std_values': result[1, :],
            'threshold': threshold,
            'change_points': change_points,
            'prediction_errors': result[2, :],
            'trainlength': trainlength,
            'maxstep': maxstep
        }
        
        # 保存为pkl文件
        pkl_file = 'TCD_北京.pkl'
        with open(pkl_file, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"绘图数据已保存为pkl文件: {pkl_file}")
        
        # 保存为excel文件
        excel_file = os.path.join(excel_dir, 'TCD_北京.xlsx')
        
        # 创建时间步列表
        all_time_steps = np.arange(1, trainlength + maxstep + 1)
        
        # 创建预测相关的时间步列表（仅包含有预测值的时间步）
        pred_time_steps = np.arange(trainlength + 1, trainlength + maxstep + 1)
        
        # 创建完整的DataFrame结构
        df_data = {
            '时间步': all_time_steps,
            '预测标准差': np.full_like(all_time_steps, np.nan, dtype=float),
            '预测误差': np.full_like(all_time_steps, np.nan, dtype=float),
            '变化点标记': np.zeros_like(all_time_steps, dtype=int)
        }
        
        # 填充预测数据
        df_data['预测标准差'][trainlength:] = result[1, :]
        df_data['预测误差'][trainlength:] = result[2, :]
        
        # 标记变化点
        for cp in change_points:
            if trainlength + cp < len(df_data['变化点标记']):
                df_data['变化点标记'][trainlength + cp] = 1
        
        df = pd.DataFrame(df_data)
        
        # 保存到excel
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='TCD数据', index=False)
        
        print(f"绘图数据已保存为excel文件: {excel_file}")
    
    # 11. 绘制结果
    plt.figure(figsize=(12, 10), num='网络结构变化点检测结果')

    # 子图1：真实数据与预测数据
    plt.subplot(3, 1, 1)
    time_range = range(trainlength + 1, trainlength + maxstep + 1)
    plt.plot(range(1, trainlength + maxstep + 1), xx[:trainlength + maxstep, j], '-*', label='真实数据')
    plt.plot(time_range, result[0, :], 'ro', label='预测数据')
    
    # 调用保存函数（在time_range定义后）
    save_plot_data()
    plt.title('真实数据 vs 预测数据')
    plt.xlabel('时间步')
    plt.ylabel('变量值')
    plt.legend()
    plt.grid(True)

    # 子图2：预测标准差与阈值
    plt.subplot(3, 1, 2)
    plt.plot(range(1, maxstep + 1), std_values, 'b-', label='预测标准差')
    plt.plot([1, maxstep], [threshold, threshold], 'r--', label='阈值（μ+2σ）')

    # 标记变化点
    if len(change_points) > 0:
        plt.scatter(change_points + 1, std_values[change_points],
                    s=50, c='g', marker='o', edgecolors='k', label='变化点')

    plt.title('预测标准差与变化点')
    plt.xlabel('时间步')
    plt.ylabel('标准差')
    plt.legend()
    plt.grid(True)

    # 子图3：预测误差
    plt.subplot(3, 1, 3)
    plt.plot(range(1, maxstep + 1), result[2, :], 'k-')
    plt.title('预测误差')
    plt.xlabel('时间步')
    plt.ylabel('误差值')
    plt.grid(True)

    plt.tight_layout()

    # 12. 保存结果
    output_dir = r'D:\中大教学\研究生\三层网络\PIS-multilayer-main\猴痘'
    os.makedirs(output_dir, exist_ok=True)

    # 保存标准差数据
    std_file = os.path.join(output_dir, 'stderror.csv')
    np.savetxt(std_file, result[1, :maxstep], delimiter=',')
    print(f"标准差数据已保存至: {std_file}")

    # 保存图像
    img_file = 'tcd_1.png'
    plt.savefig(img_file, dpi=300)
    print(f"图像已保存至: {img_file}")

    plt.show()


if __name__ == "__main__":
    main()
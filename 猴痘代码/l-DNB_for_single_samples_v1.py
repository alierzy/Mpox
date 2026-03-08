import os
import scipy.stats as stat
import math
import numpy as np
import random
import time
import multiprocessing
import matplotlib.pyplot as plt  # 导入绘图库
import pandas as pd  # 导入pandas库用于excel导出
import pickle  # 导入pickle库用于数据保存

# 使用 Manager 创建一个全局共享字典
def ssn_score(deta, pcc, nn):
    if pcc == 1:
        pcc = 0.99999999
    if pcc == -1:
        pcc = -0.99999999
    z = deta / ((1 - pcc * pcc) / (nn - 1))
    return z


def parallel_procedure(stage, normal, disease, title, ref, sd_mean, j, refnum, pvalue, sample_scores):
    begin = time.strftime("%Y-%m-%d:%H-%M-%S", time.localtime(time.time()))
    print("Stage: ", stage, " Sample: ", j + 1)

    network = {}
    ssn = {}
    total_score = 0  # 用于存储当前样本的总得分
    count = 0  # 计算该样本的有效模块数目

    for p in ref.keys():
        t = p.split()
        r1 = ref[p]
        try:
            r2 = stat.pearsonr(normal[t[0]] + [disease[t[0]][j]], normal[t[1]] + [disease[t[1]][j]])[0]
        except KeyError as e:
            print(f"KeyError: {e} in sample {j + 1}")
            continue  # Skip this pair if there's an error
        r = r2 - r1
        z = ssn_score(r, r1, refnum)
        p_value = 1 - stat.norm.cdf(abs(z))
        if p_value < pvalue:
            r = r if r > 0 else -r
            ssn[p] = r
            ssn[t[1] + "\t" + t[0]] = r

            if t[0] not in network:
                network[t[0]] = []
            network[t[0]].append(t[1])

            if t[1] not in network:
                network[t[1]] = []
            network[t[1]].append(t[0])

    ci = {}
    for p in network.keys():
        if len(network[p]) < 3:
            continue

        sd = abs(disease[p][j] - sd_mean[p][1]) / sd_mean[p][0]
        pcc_in = 0
        pcc_out = 0
        count = 0
        for q in network[p]:
            sd += abs(disease[q][j] - sd_mean[q][1]) / sd_mean[q][0]
            pcc_in += ssn[p + "\t" + q]

            for m in network[q]:
                if m != p:
                    pcc_out += ssn[q + "\t" + m]
                    count += 1
        sd /= len(network[p]) + 1
        pcc_in /= len(network[p])
        if count == 0:
            continue
        pcc_out /= count
        if pcc_out == 0:
            continue
        ci[p] = [sd * pcc_in / pcc_out, sd, pcc_in, pcc_out]

    # 对 ci 按照评分的第一个值降序排序
    ci = sorted(ci.items(), key=lambda d: d[1][0], reverse=True)

    # 计算该样本的总得分
    cot = 0
    for k in range(len(ci)):
        total_score += ci[k][1][0]  # 取模块评分中的第一个得分
        cot += 1  # 增加模块计数
        if cot == 40:
            break  # 当 k 等于 20 时，终止循环

    if cot > 0:
        average_score = total_score / cot  # 计算该样本的平均模块评分
        sample_scores[j] = average_score  # 存储该样本的平均模块评分
        print(f"Sample {j + 1} has an average score of {average_score}")  # 添加调试输出

    # 使用 with 语句确保文件正确关闭，且只输出前20个模块
    with open(f"Max_score_module in {stage} for sample {j + 1}", "w") as fw:
        for k in range(min(40, len(ci))):  # 确保只写入前20个模块
            fw.write(ci[k][0] + "\t" + str(ci[k][1][0]) + "\t" + str(ci[k][1][1]) + "\t" + str(ci[k][1][2]) + "\t" + str(
                ci[k][1][3]) + "\n")

    print("Begin time is " + begin)
    print("End time is " + time.strftime("%Y-%m-%d:%H-%M-%S", time.localtime(time.time())))




if __name__ == "__main__":
    pvalue = 0.05  # p-value threshold is set
    refnum = 0
    normal = {}
    f = open("reference_samples_ws.txt")
    flag = 0
    for p in f:
        flag += 1
        if flag == 1:
            t = p.split()
            refnum = len(t) - 1
            continue
        t = p.split()
        normal[t[0]] = [float(t[i]) for i in range(1, len(t))]
    f.close()

    sd_mean = {}
    for key in normal.keys():
        sd_mean[key] = [np.std(normal[key]), np.mean(normal[key])]

    f = open("reference_network_Zambia.txt")
    network = {}
    ref = {}
    for p in f:
        t = p.split()
        ref[t[0] + "\t" + t[1]] = float(t[2])

    f.close()

    stages = ["stage_i_ws.txt"]

    # 使用 Manager 创建共享字典
    with multiprocessing.Manager() as manager:
        sample_scores = manager.dict()  # 共享字典，用于存储样本得分

        pool = multiprocessing.Pool(3)

        for stage in stages:
            file = stage
            f = open(file)
            disease = {}
            title = []
            flag = 0
            for p in f:
                flag += 1
                t = p.split()
                if flag == 1:
                    title = [str(k) for k in range(1, len(t))]
                    continue
                disease[t[0]] = [float(t[k]) for k in range(1, len(t))]

            f.close()

            for j in range(len(title)):
                pool.apply_async(parallel_procedure, (stage, normal, disease, title, ref, sd_mean, j, refnum, pvalue, sample_scores))

        pool.close()
        pool.join()

        # 在所有样本计算完成后，找出平均得分最高的样本
        if not sample_scores:
            print("No sample scores found.")
        else:
            max_sample = max(sample_scores, key=sample_scores.get)
            print(f"The sample with the highest average network module score is: Sample {max_sample + 1} with a score of {sample_scores[max_sample]}")


            # 绘制样本得分柱状图
            sample_scores_dict = dict(sample_scores)  # 将 Manager 的字典转为普通字典
            sample_ids = [f"{i + 1}" for i in sample_scores_dict.keys()]
            scores = list(sample_scores_dict.values())
            tim = np.arange(0, 49, 1)


            plt.figure(figsize=(10, 6))
            # 使用与scores长度匹配的x轴数据
            plt.plot(range(len(scores)), scores, color='skyblue')
            
            # 在得分最高的点添加标注
            if scores:
                max_score_idx = np.argmax(scores)
                max_score = scores[max_score_idx]
                # 添加红色标记点
                plt.plot(max_score_idx, max_score, 'ro', markersize=8, label=f'Highest Score: {max_score:.4f}')
                # 添加文本标签
                plt.text(max_score_idx, max_score, f'{max_score:.4f}', 
                        ha='center', va='bottom', fontsize=10, color='red')
                # 添加垂直虚线（表示最高分数所在的时间点）
                plt.axvline(x=max_score_idx, color='red', linestyle='--', alpha=0.7, label=f'Time:t = {max_score_idx}')
                # 添加水平虚线（表示最高分数值）
                plt.axhline(y=max_score, color='red', linestyle='--', alpha=0.7)
            
            plt.xlabel('time')
            plt.ylabel('Average Network Module Score')
            plt.title('Average Network Module Score for time')
            plt.xticks(rotation=45)
            plt.legend()  # 添加图例
            plt.tight_layout()
            plt.savefig('l-DNB_score.png', dpi=300)
            
            # 保存绘图数据为pkl和excel文件
            def save_plot_data(sample_scores_dict, filename_prefix='l-DNB_plot'):
                """保存绘图用的数据为pkl和excel文件
                
                参数:
                    sample_scores_dict: 样本得分字典
                    filename_prefix: 输出文件名前缀
                """
                # 创建分数图文件夹
                excel_dir = '分数图'
                os.makedirs(excel_dir, exist_ok=True)
                
                # 准备要保存的数据
                sample_ids = [i + 1 for i in sample_scores_dict.keys()]
                scores = list(sample_scores_dict.values())
                
                # 找到最高分数的时间点和值
                max_score_idx = np.argmax(scores)
                max_score = scores[max_score_idx]
                max_time_point = sample_ids[max_score_idx]
                
                data_to_save = {
                    'sample_ids': sample_ids,
                    'scores': scores,
                    'max_score': max_score,
                    'max_score_idx': max_score_idx,
                    'max_time_point': max_time_point
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
                    '时间点': sample_ids,
                    '平均网络模块得分': scores,
                    '最高分数标记': [1 if i == max_score_idx else 0 for i in range(len(sample_ids))]
                }
                
                df = pd.DataFrame(df_data)
                
                # 保存到excel
                with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='l-DNB数据', index=False)
                
                print(f"绘图数据已保存为excel文件: {excel_file}")
            
            # 调用保存数据函数
            save_plot_data(sample_scores_dict, filename_prefix='l-DNB_Zambia')

            plt.show()  # 显示图表

import os
import scipy.stats as stat
import math
import numpy as np
import random
import time
from tqdm import tqdm


begin=time.strftime("%Y-%m-%d:%H-%M-%S",time.localtime(time.time()))

normal={}
f=open("reference_samples_ws.txt")
flag = 0
for p in f:
    flag += 1
    if flag == 1:
        continue
    t = p.split()
    normal[t[0]] = [float(t[i]) for i in range(1, len(t))]
f.close()

network = {}
keys = list(normal.keys())
n = len(keys)
print(f"总节点数: {n}")
print(f"总节点对数: {n * (n - 1) // 2}")

fw = open("reference_network_Zambia.txt", "w")

# 使用tqdm显示进度条
total_pairs = n * (n - 1) // 2
current_pair = 0

# 创建tqdm进度条对象
with tqdm(total=total_pairs, desc="计算节点对相关系数") as pbar:
    for i in range(n - 1):
        for j in range(i + 1, n):
            r = stat.pearsonr(normal[keys[i]], normal[keys[j]])
            # The threshold of P-value need be set in here for Pearson Correlation Coefficient
            if r[1] < 0.01 / (20501 * 20501):
                fw.write(keys[i] + "\t" + keys[j] + "\t" + str(r[0]) + "\n")
            # 更新进度条
            pbar.update(1)

fw.close()

print("Begin time is " + begin)
print("End time is " + time.strftime("%Y-%m-%d:%H-%M-%S", time.localtime(time.time())))



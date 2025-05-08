# 训练验证模型
from sklearn.linear_model import LinearRegression
import numpy as np

# 计算x1和x2与y的相关系数
from scipy import stats

# 训练验证模型
from sklearn.linear_model import LinearRegression
import numpy as np
from joblib import dump

import os
import json
import re
import glob
from pathlib import Path


def train_and_evaluate_model(train_data, model_save_path=None, test_size=100, random_state=42, n_estimators=100):
    """
    训练并评估随机森林和线性回归模型
    
    参数:
        train_data: 包含(x1, x2, y)元组的列表，其中x1为context_length，x2为batch_size，y为目标变量
        model_save_path: 模型保存路径，如果为None则不保存
        test_size: 测试集大小
        random_state: 随机种子
        n_estimators: 随机森林中的树数量
        
    返回:
        dict: 包含训练好的模型和评估指标的字典
    """
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from scipy import stats
    import time
    from joblib import dump
    
    # 转换为numpy数组
    train_data_array = np.array(train_data)
    
    # 提取特征和目标变量
    x1_values = [x1 for x1, _, _ in train_data]
    x2_values = [x2 for _, x2, _ in train_data]
    y_values = [y for _, _, y in train_data]
    
    # 划分训练集和测试集
    train_idx, test_idx = train_test_split(np.arange(len(train_data)), test_size=test_size, random_state=random_state)
    
    
    # 构建训练集和测试集
    X_train = np.array([[x1, x2] for x1, x2, _ in train_data_array[train_idx]])
    y_train = np.array([y for _, _, y in train_data_array[train_idx]])
    X_test = np.array([[x1, x2] for x1, x2, _ in train_data_array[test_idx]])
    y_test = np.array([y for _, _, y in train_data_array[test_idx]])
    
        
    
    # 训练随机森林模型
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(X_train, y_train)
    
    # 评估随机森林模型在训练集上的表现
    y_train_pred = rf_model.predict(X_train)
    rf_train_r2 = r2_score(y_train, y_train_pred)
    
    # 评估随机森林模型在测试集上的表现
    begin_time = time.time()
    y_test_pred = rf_model.predict(X_test)
    rf_test_r2 = r2_score(y_test, y_test_pred)
    rf_inference_time = time.time() - begin_time
    
    # 获取特征重要性
    feature_importance = rf_model.feature_importances_
    
    # 计算相关系数
    corr_x1_y = stats.pearsonr(x1_values, y_values)
    corr_x2_y = stats.pearsonr(x2_values, y_values)
    
    # 训练线性回归模型
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # 评估线性回归模型
    lr_train_r2 = lr_model.score(X_train, y_train)
    begin_time = time.time()
    lr_test_r2 = lr_model.score(X_test, y_test)
    lr_inference_time = time.time() - begin_time
    
    # 保存模型
    if model_save_path:
        print("save model",model_save_path)
        dump(rf_model, f"{model_save_path}_RF.pkl")
        dump(lr_model, f"{model_save_path}_LR.pkl")
    
    # 输出评估结果
    print("\n随机森林模型评估结果:")
    print(f"训练集 R2 分数: {rf_train_r2:.4f}")
    print(f"测试集 R2 分数: {rf_test_r2:.4f}")
    print(f"推理时间: {rf_inference_time:.6f} 秒")
    
    # print("\n特征重要性:")
    # print(f"context_length重要性: {feature_importance[0]:.4f}")
    # print(f"batch_size重要性: {feature_importance[1]:.4f}")
    
    # print(f"\n相关系数分析 (样本数量: {len(train_data)}):")
    # print(f"x1(context_length)与y的相关系数: {corr_x1_y[0]:.4f}, p值: {corr_x1_y[1]:.4f}")
    # print(f"x2(batch_size)与y的相关系数: {corr_x2_y[0]:.4f}, p值: {corr_x2_y[1]:.4f}")
    
    print("\n线性回归模型评估结果:")
    print(f"系数: {lr_model.coef_}")
    print(f"截距: {lr_model.intercept_}")
    print(f"训练集 R2 分数: {lr_train_r2:.4f}")
    print(f"测试集 R2 分数: {lr_test_r2:.4f}")
    print(f"推理时间: {lr_inference_time:.6f} 秒")
    
    # 返回结果
    return {
        "rf_model": rf_model,
        "lr_model": lr_model,
        "rf_train_r2": rf_train_r2,
        "rf_test_r2": rf_test_r2,
        "lr_train_r2": lr_train_r2,
        "lr_test_r2": lr_test_r2,
        "feature_importance": feature_importance,
        "corr_x1_y": corr_x1_y,
        "corr_x2_y": corr_x2_y
    }

def read_nospec_json_files(patternstr):
    """
    Read all JSON files in the parent directory that start with "nospec_300_new1".
    
    Returns:
        dict: Dictionary with filenames as keys and JSON content as values
    """
    json_files = {}
    
    # Get the parent directory path
    parent_directory = Path(__file__).parent.parent
    
    # Define the regex pattern for files starting with "nospec_300_new1"
    pattern = re.compile(patternstr)
    
    # Get all JSON files in the directory
    for json_file in parent_directory.glob('*.json'):
        filename = json_file.name
        
        # Check if filename matches the pattern
        if pattern.match(filename):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_content = json.load(f)
                    json_files[filename] = json_content
                    print(f"Successfully read: {filename}")
            except Exception as e:
                print(f"Error reading {filename}: {str(e)}")
    
    return json_files


def read_json(patternstr):
    import json
    results = read_nospec_json_files(patternstr)
    action_time_historys = []
    for filename, action_time_history in results.items():
        # 将字符串key转换为int
        converted_history = {}
        for action_str in   action_time_history:
            action_int = int(action_str)
            converted_history[action_int] = {}
            for batch_str in action_time_history[action_str]:
                batch_int = int(batch_str)
                converted_history[action_int][batch_int] = action_time_history[action_str][batch_str]
                for i in range(2):
                    # 这样就会多出两个，一共四个，前两个是str 后两个是int
                    converted_history[action_int][batch_int][int(i)] = action_time_history[action_str][batch_str][str(i)]
        action_time_history = converted_history
        action_time_historys.append(action_time_history)
    return action_time_historys


action_time_history = {}
train_data_v = []
train_data_d = []

# Read data from deep_05b_300_new1 to deep_05b_300_new3
for gamma in range(3, 4):  # This will iterate through 1, 2, 3
    # Using f-string to create dynamic regex pattern
    pattern = fr"^deep_05b_300_new{gamma}.*\.json$"
    print(f"Reading files matching pattern: {pattern}")
    deep_action_time_historys = read_json(pattern)
    
    for i in range(len(deep_action_time_historys)):
        for batch in range(1, 300):
            datas = deep_action_time_historys[i][0][batch][1]
            for j in range(len(datas['proposal_time'])):
                y = datas['proposal_time'][j] / gamma
                x1 = datas['context_length'][j]
                x2 = batch
                train_data_d.append((x1, x2, y))
                
# Read data from deep_05b_300_new1 to deep_05b_300_new3
tt ={}
for gamma in range(1, 6):  # This will iterate through 1, 2, 3
    # Using f-string to create dynamic regex pattern
    pattern = fr"^deep_05b_300_new{gamma}.*\.json$"
    #print(f"Reading files matching pattern: {pattern}")
    deep_action_time_historys = read_json(pattern)
    tt[gamma] = deep_action_time_historys[0][0][30][1] # action 0 即 spec 的batch size 1 decode 阶段
    for i in range(len(deep_action_time_historys)):
        for batch in range(1, 300):
            datas = deep_action_time_historys[i][0][batch][1]
            for j in range(len(datas['proposal_time'])):
                x1 = datas['context_length'][j]
                x2 = batch
                #train_data_d.append((x1, x2, y))
                train_data_v.append((x1,x2*gamma,datas['scoring_time'][j]))

kk = {}
for gamma in range(1, 6):  # This will iterate through 1, 2, 3
    # Using f-string to create dynamic regex pattern
    pattern = fr"^nospec_300_new{gamma}.*\.json$"
    #print(f"Reading files matching pattern: {pattern}")
    nospec_action_time_historys = read_json(pattern)
    kk = nospec_action_time_historys[0][2][30][1]
    for i in range(len(nospec_action_time_historys)):
        for batch in range(1, 300):
            data = nospec_action_time_historys[i][2][batch][1]
            for j in range(len(data['scoring_time'])):
                y = data['scoring_time'][j]
                x1 = data['context_length'][j]
                x2 = batch
                train_data_v.append((x1,x2,y))
#print(tt[1]['proposal_time'][:10],tt[5]['proposal_time'][:10])
print(tt[1]['scoring_time'][:10],tt[5]['scoring_time'][:10])
#print(tt[1]['proposal_time'][:10],tt[4]['proposal_time'][:10])
print(kk['scoring_time'][:10])
print("len(train_data_d)",len(train_data_d))
print("len(train_data_v)",len(train_data_v))
# results_d = train_and_evaluate_model(train_data_d, model_save_path="./DeepSeek-R1-DRAFT-Qwen2.5-0.5B")
# results_v = train_and_evaluate_model(train_data_v, model_save_path="./DeepSeek-R1-Qwen2.5-0.5B-Verify")
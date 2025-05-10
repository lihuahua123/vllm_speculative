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
import matplotlib.pyplot as plt
import seaborn as sns



def train_and_evaluate_model(train_data, model_save_path=None, test_size=100, random_state=42, n_estimators=100):
    """
    训练并评估随机森林和线性回归模型以及其他快速替代模型
    
    参数:
        train_data: 包含(x1, x2, x3, y)元组的列表，其中x1为context_length，x2为batch_size，x3为gamma，y为目标变量
        model_save_path: 模型保存路径，如果为None则不保存
        test_size: 测试集大小
        random_state: 随机种子
        n_estimators: 随机森林中的树数量
        
    返回:
        dict: 包含训练好的模型和评估指标的字典
    """
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.svm import LinearSVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.model_selection import train_test_split
    from scipy import stats
    import time
    from joblib import dump
    import pandas as pd
    
    # 转换为numpy数组
    train_data_array = np.array(train_data)

    # 划分训练集和测试集
    train_idx, test_idx = train_test_split(np.arange(len(train_data)), test_size=test_size, random_state=random_state)
    
    if len(train_data_array[0]) > 3:
        # 构建训练集和测试集
        X_train = np.array([[x1, x2, x3] for x1, x2, x3, _ in train_data_array[train_idx]])
        y_train = np.array([y for _, _, _, y in train_data_array[train_idx]])
        X_test = np.array([[x1, x2, x3] for x1, x2, x3, _ in train_data_array[test_idx]])
        y_test = np.array([y for _, _, _, y in train_data_array[test_idx]])
        feature_names = ['context_length', 'batch_size', 'gamma']
    else:
        X_train = np.array([[x1, x2] for x1, x2, _ in train_data_array[train_idx]])
        y_train = np.array([y for _, _,  y in train_data_array[train_idx]])
        X_test = np.array([[x1, x2] for x1, x2, _ in train_data_array[test_idx]])
        y_test = np.array([y for _, _,  y in train_data_array[test_idx]])
        feature_names = ['context_length', 'batch_size']
    
    
    corr_results = {}
    for i, feature in enumerate(feature_names):
        corr = stats.pearsonr(X_train[:, i], y_train)
        corr_results[feature] = {'corr': corr[0], 'p_value': corr[1]}
    
    # 定义要评估的模型字典
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=random_state),
        'Lasso': Lasso(alpha=0.1, random_state=random_state),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state),
        'DecisionTree': DecisionTreeRegressor(max_depth=5, random_state=random_state),
        'RandomForest': RandomForestRegressor(n_estimators=n_estimators, random_state=random_state),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=random_state),
        'LinearSVR': LinearSVR(C=1.0, epsilon=0.2, random_state=random_state)
    }
    
    # 评估所有模型
    results = {}
    for name, model in models.items():
        # 训练模型
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # 训练集评估
        y_train_pred = model.predict(X_train)
        train_r2 = r2_score(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        
        # 测试集评估
        start_time = time.time()
        y_test_pred = model.predict(X_test)
        inference_time = time.time() - start_time
        
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # 保存结果
        results[name] = {
            'model': model,
            'train_r2': train_r2,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'train_time': train_time,
            'inference_time': inference_time
        }
        
        # # 保存模型
        # if model_save_path:
        #     dump(model, f"{model_save_path}_{name}.pkl")
    
    # 输出评估结果
    print("\n模型评估结果汇总:")
    df_results = pd.DataFrame({
        'Model': list(results.keys()),
        'Train R²': [results[model]['train_r2'] for model in results],
        'Test R²': [results[model]['test_r2'] for model in results],
        'Train RMSE': [results[model]['train_rmse'] for model in results],
        'Test RMSE': [results[model]['test_rmse'] for model in results],
        'Train MAE': [results[model]['train_mae'] for model in results],
        'Test MAE': [results[model]['test_mae'] for model in results],
        'Training Time (s)': [results[model]['train_time'] for model in results],
        'Inference Time (s)': [results[model]['inference_time'] for model in results]
    })
    print(df_results.sort_values('Test R²', ascending=False))
    
    # # 可视化模型比较
    # plt.figure(figsize=(14, 8))
    
    # # R²比较
    # plt.subplot(2, 2, 1)
    # models_names = list(results.keys())
    # train_r2_values = [results[model]['train_r2'] for model in results]
    # test_r2_values = [results[model]['test_r2'] for model in results]
    
    # x = np.arange(len(models_names))
    # width = 0.35
    
    # plt.bar(x - width/2, train_r2_values, width, label='Train R²')
    # plt.bar(x + width/2, test_r2_values, width, label='Test R²')
    # plt.xlabel('Models')
    # plt.ylabel('R²')
    # plt.title('R² Score Comparison')
    # plt.xticks(x, models_names, rotation=45, ha='right')
    # plt.legend()
    
    # # RMSE比较
    # plt.subplot(2, 2, 2)
    # train_rmse_values = [results[model]['train_rmse'] for model in results]
    # test_rmse_values = [results[model]['test_rmse'] for model in results]
    
    # plt.bar(x - width/2, train_rmse_values, width, label='Train RMSE')
    # plt.bar(x + width/2, test_rmse_values, width, label='Test RMSE')
    # plt.xlabel('Models')
    # plt.ylabel('RMSE')
    # plt.title('RMSE Comparison')
    # plt.xticks(x, models_names, rotation=45, ha='right')
    # plt.legend()
    
    # # 训练和推理时间比较
    # plt.subplot(2, 2, 3)
    # train_times = [results[model]['train_time'] for model in results]
    # inference_times = [results[model]['inference_time'] for model in results]
    
    # plt.bar(x - width/2, train_times, width, label='Training Time')
    # plt.bar(x + width/2, inference_times, width, label='Inference Time')
    # plt.xlabel('Models')
    # plt.ylabel('Time (s)')
    # plt.title('Training and Inference Time Comparison')
    # plt.xticks(x, models_names, rotation=45, ha='right')
    # plt.legend()
    
    # # 残差分布比较 (只展示前两个模型)
    # plt.subplot(2, 2, 4)
    # top_models = sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True)[:2]
    
    # for i, (name, result) in enumerate(top_models):
    #     model = result['model']
    #     y_pred = model.predict(X_test)
    #     residuals = y_test - y_pred
        
    #     sns.kdeplot(residuals, label=name)
    
    # plt.xlabel('Residuals')
    # plt.ylabel('Density')
    # plt.title('Residual Distribution of Top Models')
    # plt.legend()
    
    # plt.tight_layout()
    # plt.savefig('model_comparison.png')
    # plt.close()
    # print("模型比较图已保存为 'model_comparison.png'")
    if 'DecisionTree' in results:
        if model_save_path:
            dump(results['DecisionTree']['model'], f"{model_save_path}_dt.pkl")
        
    # # 输出线性回归模型的详细结果
    # if 'LinearRegression' in results:
    #     lr_model = results['LinearRegression']['model']
    #     print("\n线性回归模型详细结果:")
    #     print(f"系数: {lr_model.coef_}")
    #     print(f"截距: {lr_model.intercept_}")
        
    #     # 如果有三个特征，打印完整模型方程
    #     if len(feature_names) == 3:
    #         equation = f"y = {lr_model.intercept_:.4f}"
    #         for i, coef in enumerate(lr_model.coef_):
    #             equation += f" + {coef:.4f} * {feature_names[i]}"
    #         print(f"模型方程: {equation}")
    #     # 保存模型
    #     if model_save_path:
    #         dump(lr_model, f"{model_save_path}_lr.pkl")
    
    # 输出最佳模型及其结果
    best_model_name = df_results.sort_values('Test R²', ascending=False).iloc[0]['Model']
    best_model_result = results[best_model_name]
    
    print(f"\n最佳模型: {best_model_name}")
    print(f"测试集 R² 分数: {best_model_result['test_r2']:.4f}")
    print(f"测试集 RMSE: {best_model_result['test_rmse']:.4f}")
    print(f"测试集 MAE: {best_model_result['test_mae']:.4f}")
    print(f"推理时间: {best_model_result['inference_time']:.6f} 秒")
    
    return results


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
tt = {i:[] for i in range(10)}
for gamma in range(1, 6):  # This will iterate through 1, 2, 3
    # Using f-string to create dynamic regex pattern
    pattern = fr"^deep_05b_300_new{gamma}.*\.json$"
    print(f"Reading files matching pattern: {pattern}")
    deep_action_time_historys = read_json(pattern)
    
    for i in range(len(deep_action_time_historys)):
        for batch in range(1, 300):
            datas = deep_action_time_historys[i][0][batch][1]
            for j in range(len(datas['proposal_time'])):
                 # 0: draft, 1: scoring, 2: verification 3: batch size 4: num_accepted_tokens 5: context_length 6: stage 7: proposed_length
                tt[gamma].append((datas['accepted_tokens_length'][j],batch,datas['accepted_tokens_length'][j]/(batch * gamma)))


train_data = []
train_table = {i:{
     j : [] for j in range(1,300)
    } for i in range(1,6)}
train_table_avg =      {i:{
     j : -1 for j in range(1,300)
    } for i in range(1,6)}  
for key,value in tt.items():
    x = key
    y = []
    for k in value:
        train_data.append((key,k[1],k[0]+k[1]))
        train_table[key][k[1]].append(k[0]+k[1])
for key,value in train_table.items():
    for k,v in value.items():
        if len(v) > 0:
            train_table_avg[key][k] = np.mean(v)
# Save train_table_avg to a pickle file
import pickle
print(train_table_avg)
# Save the train_table_avg dictionary
with open('./train_table_avg.pkl', 'wb') as f:
    pickle.dump(train_table_avg, f)

# Load the train_table_avg dictionary (for future use)
# with open('train_table_avg.pkl', 'rb') as f:
#     train_table_avg = pickle.load(f)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create 3D figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Extract x, y, z coordinates from train_data
x = [data[0] for data in train_data]  # gamma
y = [data[1] for data in train_data]  # batch size
z = [data[2] for data in train_data]  # generated tokens

# Create scatter plot
scatter = ax.scatter(x, y, z)
# Plot the average points from train_table_avg
for gamma, batch_dict in train_table_avg.items():
    for batch_size, avg_tokens in batch_dict.items():
        if avg_tokens != -1:  # Only plot if we have valid data
            ax.scatter(gamma, batch_size, avg_tokens, color='red', s=10, marker='*')


# Add labels
ax.set_xlabel('Gamma')
ax.set_ylabel('Batch Size')
ax.set_zlabel('Generated Tokens')

# Add title
plt.title('3D Relationship between Gamma, Batch Size and Generated Tokens')

# Save plot
plt.savefig('./figs/3d_relationship.png')
plt.close()

train_and_evaluate_model(train_data,'./generated_data_num_predict_model')
# 画图1
# batch_size = 17             
# for key,value in tt.items():
#     x = key
#     y = []
#     for k in value:
#         if k[1] == batch_size:
#             y.append(k[0]+batch_size)
#     if len(y) > 0:  # Only plot if we have data
#         #tokens = (1 - 0.6 ** (x + 1)) / (1 - 0.6)
#         plt.boxplot(y, positions=[x])
#     # Plot theoretical tokens as points
#     tokens = batch_size * (1 - 0.6 ** (x + 1)) / (1 - 0.6)
#     plt.plot(x, tokens, 'ro', label='Theoretical tokens')  # 'ro' means red dots
# 画图2
# for key,value in tt.items():
#     x = key
#     y = []
#     for k in value:
#         y.append(k[2])
#     if len(y) > 0:  # Only plot if we have data
#         plt.boxplot(y, positions=[x])


# After the loop ends, add labels and show plot
plt.xlabel('Gamma')
plt.ylabel('# Generated tokens')
# plt.title('Acceptance Rate Distribution by Gamma')
plt.grid(True)
plt.savefig('./figs/acceptance_rate_distribution.png')
plt.close()


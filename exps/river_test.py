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
        # 'Ridge': Ridge(alpha=1.0, random_state=random_state),
        # 'Lasso': Lasso(alpha=0.1, random_state=random_state),
        # 'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state),
        # 'DecisionTree': DecisionTreeRegressor(max_depth=5, random_state=random_state),
        # 'RandomForest': RandomForestRegressor(n_estimators=n_estimators, random_state=random_state),
        # 'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state),
        # 'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=random_state),
        # 'LinearSVR': LinearSVR(C=1.0, epsilon=0.2, random_state=random_state)
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
        print(results[name])
        
        # 保存模型
        if model_save_path:
            dump(model, f"{model_save_path}_{name}.pkl")
    
    


def read_nospec_json_files(patternstr):
    """
    Read all JSON files in the parent directory that start with "nospec_300_new1".
    
    Returns:
        dict: Dictionary with filenames as keys and JSON content as values
    """
    json_files = {}
    
    # Get the parent directory path
    parent_directory = Path(__file__).parent.parent
    # Change to profile_log directory
    profile_log_dir = parent_directory / 'profile_log'
    if not profile_log_dir.exists():
        print(f"Warning: {profile_log_dir} does not exist")
        return {}
    parent_directory = profile_log_dir
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
smart_train_data_v = []
smart_train_data_d = []
file_name = "300_new_llama"
# Read data from deep_05b_300_new1 to deep_05b_300_new3
for gamma in range(1, 6):  # This will iterate through 1, 2, 3
    # Using f-string to create dynamic regex pattern
    pattern = fr"^{file_name}_{gamma}_.*_deep\.json$"
    print(f"Reading files matching pattern: {pattern}")
    deep_action_time_historys = read_json(pattern)
    
    for i in range(len(deep_action_time_historys)):
        for batch in range(1, 300):
            datas = deep_action_time_historys[i][0][batch][1]
            for j in range(len(datas['proposal_time'])):
                y = datas['proposal_time'][j]
                x1 = datas['context_length'][j]
                x2 = batch
                train_data_d.append((x1, x2,gamma, y))
                smart_train_data_d.append((x1,x2,y/(gamma)))
                
# Read data from deep_05b_300_new1 to deep_05b_300_new3
for gamma in range(1, 6):  # This will iterate through 1, 2, 3
    # Using f-string to create dynamic regex pattern
    pattern = fr"^{file_name}_{gamma}_.*_deep\.json$"
    #print(f"Reading files matching pattern: {pattern}")
    deep_action_time_historys = read_json(pattern)
    for i in range(len(deep_action_time_historys)):
        for batch in range(1, 300):
            datas = deep_action_time_historys[i][0][batch][1]
            for j in range(len(datas['proposal_time'])):
                x1 = datas['context_length'][j]
                x2 = batch
                #train_data_d.append((x1, x2, y))
                train_data_v.append((x1,x2*(gamma+1),1,datas['scoring_time'][j]))
                smart_train_data_v.append((x1,x2*(gamma+1),datas['scoring_time'][j]))

for gamma in range(1, 6):  # This will iterate through 1, 2, 3
    # Using f-string to create dynamic regex pattern
    pattern = fr"^{file_name}_{gamma}_.*_nospec\.json$"
    #print(f"Reading files matching pattern: {pattern}")
    nospec_action_time_historys = read_json(pattern)
    for i in range(len(nospec_action_time_historys)):
        for batch in range(1, 300):
            data = nospec_action_time_historys[i][2][batch][1]
            for j in range(len(data['scoring_time'])):
                y = data['scoring_time'][j]
                x1 = data['context_length'][j]
                x2 = batch
                train_data_v.append((x1,x2,1,y))
                smart_train_data_v.append((x1,x2,y))

print("len(train_data_d)",len(train_data_d))
print("len(train_data_v)",len(train_data_v))
def train_and_save_decision_tree(train_data, model_save_path=None, test_size=100, random_state=42, max_depth=5, min_samples_split=2, min_samples_leaf=1):
    """
    专门训练、评估和保存决策树模型
    
    参数:
        train_data: 包含(x1, x2, x3, y)元组的列表，其中x1为context_length，x2为batch_size，x3为gamma(可选)，y为目标变量
        model_save_path: 模型保存路径，如果为None则不保存
        test_size: 测试集大小
        random_state: 随机种子
        max_depth: 决策树最大深度
        min_samples_split: 内部节点再划分所需最小样本数
        min_samples_leaf: 叶子节点所需最小样本数
        
    返回:
        dict: 包含训练好的模型和评估指标的字典
    """
    import numpy as np
    from sklearn.tree import DecisionTreeRegressor, export_graphviz
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.model_selection import train_test_split
    import time
    from joblib import dump
    import pandas as pd
    import matplotlib.pyplot as plt

    # 转换为numpy数组
    train_data_array = np.array(train_data)

    # 划分训练集和测试集
    train_idx, test_idx = train_test_split(np.arange(len(train_data)), test_size=test_size, random_state=random_state)
    
    if len(train_data_array[0]) > 3:
        # 构建训练集和测试集 (包含gamma特征)
        X_train = np.array([[x1, x2, x3] for x1, x2, x3, _ in train_data_array[train_idx]])
        y_train = np.array([y for _, _, _, y in train_data_array[train_idx]])
        X_test = np.array([[x1, x2, x3] for x1, x2, x3, _ in train_data_array[test_idx]])
        y_test = np.array([y for _, _, _, y in train_data_array[test_idx]])
        feature_names = ['context_length', 'batch_size', 'gamma']
    else:
        # 构建训练集和测试集 (不包含gamma特征)
        X_train = np.array([[x1, x2] for x1, x2, _ in train_data_array[train_idx]])
        y_train = np.array([y for _, _,  y in train_data_array[train_idx]])
        X_test = np.array([[x1, x2] for x1, x2, _ in train_data_array[test_idx]])
        y_test = np.array([y for _, _,  y in train_data_array[test_idx]])
        feature_names = ['context_length', 'batch_size']
    
    # 创建决策树模型
    dt_model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    
    # 训练模型
    print("\n正在训练决策树模型...")
    start_time = time.time()
    dt_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"训练完成，耗时: {train_time:.4f} 秒")
    
    # 评估模型
    # 训练集评估
    y_train_pred = dt_model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    # 测试集评估
    start_time = time.time()
    y_test_pred = dt_model.predict(X_test)
    inference_time = time.time() - start_time
    
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
   
    
    # 保存模型
    if model_save_path:
        model_file = f"{model_save_path}_DecisionTree.pkl"
        dump(dt_model, model_file)
        print(f"\n决策树模型已保存到: {model_file}")
        
        # try:
        #     # 尝试直接生成PDF文件
        #     graph = graphviz.Source.from_file(dot_file)
        #     pdf_file = f"{model_save_path}_DecisionTree.pdf"
        #     graph.render(pdf_file.replace('.pdf', ''), format='pdf', cleanup=True)
        #     print(f"决策树PDF文件已保存到: {pdf_file}")
        # except Exception as e:
        #     print(f"无法自动生成PDF: {str(e)}")
    
    # 输出评估结果
    print("\n决策树模型评估结果:")
    print(f"训练集 R² 分数: {train_r2:.4f}")
    print(f"测试集 R² 分数: {test_r2:.4f}")
    print(f"训练集 RMSE: {train_rmse:.4f}")
    print(f"测试集 RMSE: {test_rmse:.4f}")
    print(f"训练集 MAE: {train_mae:.4f}")
    print(f"测试集 MAE: {test_mae:.4f}")
    print(f"推理时间: {inference_time:.6f} 秒")
# ... existing code ...

def train_river_online_decision_tree(train_data, model_save_path=None):
    """
    使用River库训练在线决策树模型
    
    参数:
        train_data: 包含(x1, x2, x3, y)元组的列表，其中x1为context_length，x2为batch_size，x3为gamma(可选)，y为目标变量
        model_save_path: 模型保存路径，如果为None则不保存
        
    返回:
        dict: 包含训练好的模型和评估指标的字典
    """
    from river import tree
    from river import metrics
    import time
    import pickle
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    print("\n=== 使用River在线决策树进行训练 ===")
    
    # 转换为numpy数组
    train_data_array = np.array(train_data)
    print("len(train_data)",len(train_data))
    # 划分训练集和测试集
    train_idx, test_idx = train_test_split(np.arange(len(train_data)), test_size=100, random_state=42)
    
    # 确定特征维度
    if len(train_data_array[0]) > 3:
        feature_names = ['context_length', 'batch_size', 'gamma']
        train_samples = [(dict(zip(feature_names, [x1, x2, x3])), y) for x1, x2, x3, y in train_data_array[train_idx]]
        test_samples = [(dict(zip(feature_names, [x1, x2, x3])), y) for x1, x2, x3, y in train_data_array[test_idx]]
    else:
        feature_names = ['context_length', 'batch_size']
        train_samples = [(dict(zip(feature_names, [x1, x2])), y) for x1, x2, y in train_data_array[train_idx]]
        test_samples = [(dict(zip(feature_names, [x1, x2])), y) for x1, x2, y in train_data_array[test_idx]]
    
    # 创建River在线决策树模型
    # HoeffdingTreeRegressor是River中的在线决策树回归器
    model = tree.HoeffdingTreeRegressor(
        grace_period=200,  # 在分裂前等待的样本数
        max_depth=10,      # 最大深度
        delta=1e-7,        # 置信度参数
        tau=0.05,          # 分裂阈值
        leaf_prediction='mean',  # 叶节点预测方法
    )
    
    # 初始化评估指标
    train_mae = metrics.MAE()
    train_rmse = metrics.RMSE()
    
    # 在线训练过程
    print("开始在线训练...")
    start_time = time.time()
    
    for i, (x, y) in enumerate(train_samples):
        # 预测（在学习之前）
        y_pred = model.predict_one(x)
        
        # 更新训练指标
        if y_pred is not None:  # 前几个样本可能无法预测
            train_mae.update(y, y_pred)
            train_rmse.update(y, y_pred)
        
        # 在线学习
        model.learn_one(x, y)
        
        # 每1000个样本打印一次进度
        if (i + 1) % 1000 == 0:
            print(f"已处理 {i + 1} 个训练样本, 当前训练MAE: {train_mae.get():.4f}, RMSE: {train_rmse.get():.4f}")
    
    train_time = time.time() - start_time
    print(f"训练完成，耗时: {train_time:.4f} 秒")
    
    # 测试集评估
    print("开始测试集评估...")
    test_mae = metrics.MAE()
    test_rmse = metrics.RMSE()
    
    start_time = time.time()
    predictions = []
    actuals = []
    
    for x, y in test_samples:
        y_pred = model.predict_one(x)
        if y_pred is not None:
            test_mae.update(y, y_pred)
            test_rmse.update(y, y_pred)
            predictions.append(y_pred)
            actuals.append(y)
    
    inference_time = time.time() - start_time
    
    # 计算R²分数
    if len(predictions) > 0:
        from sklearn.metrics import r2_score
        test_r2 = r2_score(actuals, predictions)
    else:
        test_r2 = 0.0
    
    # 保存模型
    if model_save_path:
        model_file = f"{model_save_path}_RiverDecisionTree.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"\nRiver在线决策树模型已保存到: {model_file}")
    
    # 输出评估结果
    print("\nRiver在线决策树模型评估结果:")
    print(f"训练集 MAE: {train_mae.get():.4f}")
    print(f"训练集 RMSE: {train_rmse.get():.4f}")
    print(f"测试集 MAE: {test_mae.get():.4f}")
    print(f"测试集 RMSE: {test_rmse.get():.4f}")
    print(f"测试集 R² 分数: {test_r2:.4f}")
    print(f"训练时间: {train_time:.4f} 秒")
    print(f"推理时间: {inference_time:.6f} 秒")
    print(f"模型大小 (节点数): {model.n_nodes}")
    print(f"模型深度: {model.height}")
    
    return {
        'model': model,
        'train_mae': train_mae.get(),
        'train_rmse': train_rmse.get(),
        'test_mae': test_mae.get(),
        'test_rmse': test_rmse.get(),
        'test_r2': test_r2,
        'train_time': train_time,
        'inference_time': inference_time,
        'n_nodes': model.n_nodes,
        'height': model.height
    }

def train_river_adaptive_random_forest(train_data, model_save_path=None):
    """
    使用River库训练自适应随机森林模型（适合概念漂移）
    
    参数:
        train_data: 包含(x1, x2, x3, y)元组的列表
        model_save_path: 模型保存路径
        
    返回:
        dict: 包含训练好的模型和评估指标的字典
    """
    from river import forest
    from river import metrics
    import time
    import pickle
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    print("\n=== 使用River自适应随机森林进行训练 ===")
    
    # 数据预处理（与上面相同）
    train_data_array = np.array(train_data)
    train_idx, test_idx = train_test_split(np.arange(len(train_data)), test_size=100, random_state=42)
    
    if len(train_data_array[0]) > 3:
        feature_names = ['context_length', 'batch_size', 'gamma']
        train_samples = [(dict(zip(feature_names, [x1, x2, x3])), y) for x1, x2, x3, y in train_data_array[train_idx]]
        test_samples = [(dict(zip(feature_names, [x1, x2, x3])), y) for x1, x2, x3, y in train_data_array[test_idx]]
    else:
        feature_names = ['context_length', 'batch_size']
        train_samples = [(dict(zip(feature_names, [x1, x2])), y) for x1, x2, y in train_data_array[train_idx]]
        test_samples = [(dict(zip(feature_names, [x1, x2])), y) for x1, x2, y in train_data_array[test_idx]]
    
    # 创建自适应随机森林模型
    model = forest.ARFRegressor(
        n_models=10,           # 森林中的树数量
        max_depth=10,          # 每棵树的最大深度
        delta=1e-7,            # 置信度参数
        grace_period=200,      # 分裂前等待的样本数
        lambda_value=6,        # 泊松分布参数

    )
    
    # 训练和评估过程（与上面类似）
    train_mae = metrics.MAE()
    train_rmse = metrics.RMSE()
    
    print("开始在线训练...")
    start_time = time.time()
    
    for i, (x, y) in enumerate(train_samples):
        y_pred = model.predict_one(x)
        
        if y_pred is not None:
            train_mae.update(y, y_pred)
            train_rmse.update(y, y_pred)
        
        model.learn_one(x, y)
        
        if (i + 1) % 1000 == 0:
            print(f"已处理 {i + 1} 个训练样本, 当前训练MAE: {train_mae.get():.4f}, RMSE: {train_rmse.get():.4f}")
    
    train_time = time.time() - start_time
    print(f"训练完成，耗时: {train_time:.4f} 秒")
    
    # 测试集评估
    test_mae = metrics.MAE()
    test_rmse = metrics.RMSE()
    
    start_time = time.time()
    predictions = []
    actuals = []
    
    for x, y in test_samples:
        y_pred = model.predict_one(x)
        if y_pred is not None:
            test_mae.update(y, y_pred)
            test_rmse.update(y, y_pred)
            predictions.append(y_pred)
            actuals.append(y)
    
    inference_time = time.time() - start_time
    
    if len(predictions) > 0:
        from sklearn.metrics import r2_score
        test_r2 = r2_score(actuals, predictions)
    else:
        test_r2 = 0.0
    
    # 保存模型
    if model_save_path:
        model_file = f"{model_save_path}_RiverARF.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"\nRiver自适应随机森林模型已保存到: {model_file}")
    
    print("\nRiver自适应随机森林模型评估结果:")
    print(f"训练集 MAE: {train_mae.get():.4f}")
    print(f"训练集 RMSE: {train_rmse.get():.4f}")
    print(f"测试集 MAE: {test_mae.get():.4f}")
    print(f"测试集 RMSE: {test_rmse.get():.4f}")
    print(f"测试集 R² 分数: {test_r2:.4f}")
    print(f"训练时间: {train_time:.4f} 秒")
    print(f"推理时间: {inference_time:.6f} 秒")
    print(f"森林中的模型数量: {len(model)}")
    
    return {
        'model': model,
        'train_mae': train_mae.get(),
        'train_rmse': train_rmse.get(),
        'test_mae': test_mae.get(),
        'test_rmse': test_rmse.get(),
        'test_r2': test_r2,
        'train_time': train_time,
        'inference_time': inference_time,
        'n_models': len(model)
    }

# 在现有代码的最后添加River模型训练
print("\n=== 使用River在线学习库进行训练 ===")

# 训练River在线决策树模型
print("训练草稿阶段的River在线决策树...")
river_dt_results_d = train_river_online_decision_tree(
    train_data_d, 
    model_save_path="./llama-eagle-online"
)

print("训练验证阶段的River在线决策树...")
river_dt_results_v = train_river_online_decision_tree(
    train_data_v, 
    model_save_path="./llama-Verify-online"
)

# # 训练River自适应随机森林模型
# print("训练草稿阶段的River自适应随机森林...")
# river_arf_results_d = train_river_adaptive_random_forest(
#     train_data_d, 
#     model_save_path="./llama-eagle_adaptive"
# )

# print("训练验证阶段的River自适应随机森林...")
# river_arf_results_v = train_river_adaptive_random_forest(
#     train_data_v, 
#     model_save_path="./llama-Verify_adaptive"
# )    
train_and_evaluate_model(smart_train_data_d,model_save_path="./llama-eagle",test_size=100,random_state=42,n_estimators=100)
train_and_evaluate_model(smart_train_data_v,model_save_path="./llama-Verify",test_size=100,random_state=42,n_estimators=100)

# 使用专门的决策树函数训练和保存模型
print("\n=== 使用专门的决策树模型进行训练 ===")
# 针对草稿阶段的数据训练决策树
dt_results_d = train_and_save_decision_tree(
    train_data_d, 
    model_save_path="./llama-eagle",
    max_depth=8,  # 可以根据需要调整参数
    min_samples_split=5,
    min_samples_leaf=2
)

# 针对验证阶段的数据训练决策树
dt_results_v = train_and_save_decision_tree(
    train_data_v, 
    model_save_path="./llama-Verify",
    max_depth=8,  # 可以根据需要调整参数
    min_samples_split=5,
    min_samples_leaf=2
)

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


def check_linear_assumptions(X, y, feature_names=None):
    """
    检查数据是否满足线性回归的假设
    
    参数:
        X: 特征数据
        y: 目标变量
        feature_names: 特征名称列表
        
    返回:
        dict: 包含检查结果的字典
    """
    import numpy as np
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    from scipy import stats
    import pandas as pd
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    
    if feature_names is None:
        feature_names = [f'特征{i+1}' for i in range(X.shape[1])]
        
    # 线性关系检查 - 计算相关系数
    correlations = {}
    for i, name in enumerate(feature_names):
        corr = stats.pearsonr(X[:, i], y)
        correlations[name] = {'相关系数': corr[0], 'p值': corr[1]}
    
    # 训练线性回归模型
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # 计算残差
    residuals = y - y_pred
    
    # 检查残差的正态性 (Shapiro-Wilk测试)
    # 如果样本量大，可能需要抽样
    if len(residuals) > 5000:
        sample_indices = np.random.choice(len(residuals), 5000, replace=False)
        residuals_sample = residuals[sample_indices]
    else:
        residuals_sample = residuals
        
    normality_test = stats.shapiro(residuals_sample)
    
    # 检查方差齐性 (残差与预测值的关系)
    # 使用Breusch-Pagan测试
    from statsmodels.stats.diagnostic import het_breuschpagan
    bp_test = het_breuschpagan(residuals, X)
    
    # 检查多重共线性
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif_data = pd.DataFrame()
    vif_data["特征"] = feature_names
    vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    
    # 保存结果
    results = {
        '相关系数': correlations,
        '残差正态性': {'统计量': normality_test[0], 'p值': normality_test[1], 
                   '结论': '残差呈正态分布' if normality_test[1] > 0.05 else '残差不呈正态分布'},
        '方差齐性': {'统计量': bp_test[0], 'p值': bp_test[1], 
                '结论': '方差齐性' if bp_test[1] > 0.05 else '方差不齐性'},
        '多重共线性': vif_data
    }
    
    print("\n线性回归假设检验结果:")
    print(f"1. 相关系数分析:")
    for feature, stats_values in correlations.items():
        print(f"   {feature}: 相关系数 = {stats_values['相关系数']:.4f}, p值 = {stats_values['p值']:.4f}")
        
    print(f"\n2. 残差正态性 (Shapiro-Wilk检验):")
    print(f"   统计量 = {normality_test[0]:.4f}, p值 = {normality_test[1]:.8f}")
    print(f"   结论: {results['残差正态性']['结论']}")
    
    print(f"\n3. 方差齐性 (Breusch-Pagan检验):")
    print(f"   统计量 = {bp_test[0]:.4f}, p值 = {bp_test[1]:.4f}")
    print(f"   结论: {results['方差齐性']['结论']}")
    
    print(f"\n4. 多重共线性检验 (VIF):")
    print(vif_data)
    
    # 绘制诊断图
    plt.figure(figsize=(12, 10))
    
    # 残差 vs 预测值
    plt.subplot(2, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('残差 vs 预测值')
    
    # 残差Q-Q图
    plt.subplot(2, 2, 2)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('残差Q-Q图')
    
    # 各特征与目标变量的关系
    plt.subplot(2, 2, 3)
    for i, name in enumerate(feature_names):
        if i == 0:  # 只显示第一个特征与目标变量的关系
            plt.scatter(X[:, i], y, alpha=0.5, label=name)
            plt.xlabel(name)
            plt.ylabel('目标变量')
            plt.title('特征与目标变量关系')
            break
    
    # 残差直方图
    plt.subplot(2, 2, 4)
    sns.histplot(residuals, kde=True)
    plt.xlabel('残差')
    plt.title('残差分布')
    
    plt.tight_layout()
    plt.savefig('linear_regression_diagnostics.png')
    plt.close()
    print("诊断图已保存为 'linear_regression_diagnostics.png'")
    
    return results


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
        
        # 保存模型
        if model_save_path:
            dump(model, f"{model_save_path}_{name}.pkl")
    
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
    
    # 可视化模型比较
    plt.figure(figsize=(14, 8))
    
    # R²比较
    plt.subplot(2, 2, 1)
    models_names = list(results.keys())
    train_r2_values = [results[model]['train_r2'] for model in results]
    test_r2_values = [results[model]['test_r2'] for model in results]
    
    x = np.arange(len(models_names))
    width = 0.35
    
    plt.bar(x - width/2, train_r2_values, width, label='Train R²')
    plt.bar(x + width/2, test_r2_values, width, label='Test R²')
    plt.xlabel('Models')
    plt.ylabel('R²')
    plt.title('R² Score Comparison')
    plt.xticks(x, models_names, rotation=45, ha='right')
    plt.legend()
    
    # RMSE比较
    plt.subplot(2, 2, 2)
    train_rmse_values = [results[model]['train_rmse'] for model in results]
    test_rmse_values = [results[model]['test_rmse'] for model in results]
    
    plt.bar(x - width/2, train_rmse_values, width, label='Train RMSE')
    plt.bar(x + width/2, test_rmse_values, width, label='Test RMSE')
    plt.xlabel('Models')
    plt.ylabel('RMSE')
    plt.title('RMSE Comparison')
    plt.xticks(x, models_names, rotation=45, ha='right')
    plt.legend()
    
    # 训练和推理时间比较
    plt.subplot(2, 2, 3)
    train_times = [results[model]['train_time'] for model in results]
    inference_times = [results[model]['inference_time'] for model in results]
    
    plt.bar(x - width/2, train_times, width, label='Training Time')
    plt.bar(x + width/2, inference_times, width, label='Inference Time')
    plt.xlabel('Models')
    plt.ylabel('Time (s)')
    plt.title('Training and Inference Time Comparison')
    plt.xticks(x, models_names, rotation=45, ha='right')
    plt.legend()
    
    # 残差分布比较 (只展示前两个模型)
    plt.subplot(2, 2, 4)
    top_models = sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True)[:2]
    
    for i, (name, result) in enumerate(top_models):
        model = result['model']
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        sns.kdeplot(residuals, label=name)
    
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title('Residual Distribution of Top Models')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
    print("模型比较图已保存为 'model_comparison.png'")
    
    # 输出线性回归模型的详细结果
    if 'LinearRegression' in results:
        lr_model = results['LinearRegression']['model']
        print("\n线性回归模型详细结果:")
        print(f"系数: {lr_model.coef_}")
        print(f"截距: {lr_model.intercept_}")
        
        # 如果有三个特征，打印完整模型方程
        if len(feature_names) == 3:
            equation = f"y = {lr_model.intercept_:.4f}"
            for i, coef in enumerate(lr_model.coef_):
                equation += f" + {coef:.4f} * {feature_names[i]}"
            print(f"模型方程: {equation}")
    
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
for gamma in range(1, 6):  # This will iterate through 1, 2, 3
    # Using f-string to create dynamic regex pattern
    pattern = fr"^deep_05b_300_new{gamma}.*\.json$"
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
                train_data_v.append((x1,x2*(gamma+1),1,datas['scoring_time'][j]))

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
                train_data_v.append((x1,x2,1,y))
#print(tt[1]['proposal_time'][:10],tt[5]['proposal_time'][:10])
print(tt[1]['scoring_time'][:10],tt[5]['scoring_time'][:10])
#print(tt[1]['proposal_time'][:10],tt[4]['proposal_time'][:10])
print(kk['scoring_time'][:10])
print("len(train_data_d)",len(train_data_d))
print("len(train_data_v)",len(train_data_v))
# results_d = train_and_evaluate_model(train_data_d, model_save_path="./DeepSeek-R1-DRAFT-Qwen2.5-0.5B")
# results_v = train_and_evaluate_model(train_data_v, model_save_path="./DeepSeek-R1-Qwen2.5-0.5B-Verify")

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
    import graphviz
    
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
    


# 使用专门的决策树函数训练和保存模型
print("\n=== 使用专门的决策树模型进行训练 ===")
# 针对草稿阶段的数据训练决策树
dt_results_d = train_and_save_decision_tree(
    train_data_d, 
    model_save_path="./DeepSeek-R1-DRAFT-Qwen2.5-0.5B",
    max_depth=8,  # 可以根据需要调整参数
    min_samples_split=5,
    min_samples_leaf=2
)

# 针对验证阶段的数据训练决策树
dt_results_v = train_and_save_decision_tree(
    train_data_v, 
    model_save_path="./DeepSeek-R1-Qwen2.5-0.5B-Verify",
    max_depth=8,  # 可以根据需要调整参数
    min_samples_split=5,
    min_samples_leaf=2
)

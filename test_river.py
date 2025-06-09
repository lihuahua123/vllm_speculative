#!/usr/bin/env python3
"""
测试River在线模型的加载和预测功能
"""

import pickle
import numpy as np
import time
from pathlib import Path

def load_and_test_online_models():
    """
    加载并测试River在线模型
    """
    # 模型文件路径
    verify_online_model_path = 'llama-Verify-online_RiverDecisionTree.pkl'
    draft_online_model_path = 'llama-eagle-online_RiverDecisionTree.pkl'
    
    print("=== 测试River在线模型 ===")
    
    # 检查文件是否存在
    if not Path(verify_online_model_path).exists():
        print(f"错误: {verify_online_model_path} 不存在")
        return False
        
    if not Path(draft_online_model_path).exists():
        print(f"错误: {draft_online_model_path} 不存在")
        return False
    
    try:
        # 加载模型
        print("正在加载验证模型...")
        with open(verify_online_model_path, 'rb') as f:
            verify_model = pickle.load(f)
        print(f"验证模型加载成功: {type(verify_model)}")
        
        print("正在加载草稿模型...")
        with open(draft_online_model_path, 'rb') as f:
            draft_model = pickle.load(f)
        print(f"草稿模型加载成功: {type(draft_model)}")
        
        # 测试预测功能
        print("\n=== 测试预测功能 ===")
        
        # 测试数据
        test_cases = [
            {'context_length': 100, 'batch_size': 1, 'gamma': 3},
            {'context_length': 500, 'batch_size': 2, 'gamma': 5},
            {'context_length': 1000, 'batch_size': 4, 'gamma': 2},
            {'context_length': 2000, 'batch_size': 8, 'gamma': 1},
        ]
        
        print("测试草稿模型预测:")
        for i, test_case in enumerate(test_cases):
            start_time = time.time()
            prediction = draft_model.predict_one(test_case)
            inference_time = time.time() - start_time
            
            print(f"  测试案例 {i+1}: {test_case}")
            print(f"    预测结果: {prediction}")
            print(f"    推理时间: {inference_time:.6f} 秒")
        
        print("\n测试验证模型预测:")
        for i, test_case in enumerate(test_cases):
            # 验证模型的输入格式可能不同
            verify_case = {
                'context_length': test_case['context_length'],
                'batch_size': test_case['batch_size'] * (test_case['gamma'] + 1),
                'gamma': 1
            }
            
            start_time = time.time()
            prediction = verify_model.predict_one(verify_case)
            inference_time = time.time() - start_time
            
            print(f"  测试案例 {i+1}: {verify_case}")
            print(f"    预测结果: {prediction}")
            print(f"    推理时间: {inference_time:.6f} 秒")
        
        # 测试在线学习功能
        print("\n=== 测试在线学习功能 ===")
        
        # 模拟新的训练数据
        new_training_data = [
            ({'context_length': 150, 'batch_size': 1, 'gamma': 2}, 0.05),
            ({'context_length': 300, 'batch_size': 2, 'gamma': 3}, 0.08),
            ({'context_length': 600, 'batch_size': 3, 'gamma': 4}, 0.12),
        ]
        
        print("在线学习前的预测:")
        test_input = {'context_length': 200, 'batch_size': 2, 'gamma': 3}
        pred_before = draft_model.predict_one(test_input)
        print(f"  输入: {test_input}")
        print(f"  预测: {pred_before}")
        
        print("\n进行在线学习...")
        for x, y in new_training_data:
            draft_model.learn_one(x, y)
            print(f"  学习: {x} -> {y}")
        
        print("\n在线学习后的预测:")
        pred_after = draft_model.predict_one(test_input)
        print(f"  输入: {test_input}")
        print(f"  预测: {pred_after}")
        print(f"  变化: {pred_after - pred_before if pred_before is not None and pred_after is not None else 'N/A'}")
        
        # 显示模型信息
        print("\n=== 模型信息 ===")
        if hasattr(draft_model, 'n_nodes'):
            print(f"草稿模型节点数: {draft_model.n_nodes}")
        if hasattr(draft_model, 'height'):
            print(f"草稿模型深度: {draft_model.height}")
        if hasattr(verify_model, 'n_nodes'):
            print(f"验证模型节点数: {verify_model.n_nodes}")
        if hasattr(verify_model, 'height'):
            print(f"验证模型深度: {verify_model.height}")
        
        print("\n✅ 在线模型测试成功!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def simulate_scheduler_usage():
    """
    模拟调度器中的使用场景
    """
    print("\n=== 模拟调度器使用场景 ===")
    
    try:
        # 加载模型
        with open('llama-Verify-online_RiverDecisionTree.pkl', 'rb') as f:
            verify_model = pickle.load(f)
        with open('llama-eagle-online_RiverDecisionTree.pkl', 'rb') as f:
            draft_model = pickle.load(f)
        
        # 模拟调度器中的预测场景
        context_length = 1000
        batch_size = 4
        proposed_length = 3
        
        print(f"模拟场景: context_length={context_length}, batch_size={batch_size}, proposed_length={proposed_length}")
        
        # 草稿阶段预测
        draft_features = {
            'context_length': context_length,
            'batch_size': batch_size,
            'gamma': proposed_length
        }
        draft_time = draft_model.predict_one(draft_features)
        
        # 验证阶段预测
        verify_features = {
            'context_length': context_length,
            'batch_size': batch_size * (proposed_length + 1),
            'gamma': 1
        }
        verify_time = verify_model.predict_one(verify_features)
        
        total_time = (draft_time or 0) + (verify_time or 0)
        
        print(f"草稿时间预测: {draft_time}")
        print(f"验证时间预测: {verify_time}")
        print(f"总时间预测: {total_time}")
        
        # 模拟在线学习更新
        print("\n模拟在线学习更新...")
        actual_draft_time = 0.045  # 模拟实际测量的时间
        actual_verify_time = 0.032
        
        draft_model.learn_one(draft_features, actual_draft_time)
        verify_model.learn_one(verify_features, actual_verify_time)
        
        print(f"使用实际时间更新模型: draft={actual_draft_time}, verify={actual_verify_time}")
        
        # 再次预测看是否有变化
        new_draft_time = draft_model.predict_one(draft_features)
        new_verify_time = verify_model.predict_one(verify_features)
        
        print(f"更新后的预测: draft={new_draft_time}, verify={new_verify_time}")
        
        return True
        
    except Exception as e:
        print(f"模拟失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("开始测试River在线模型...")
    
    # 测试基本功能
    success = load_and_test_online_models()
    
    if success:
        # 模拟调度器使用场景
        simulate_scheduler_usage()
    
    print("\n测试完成!")
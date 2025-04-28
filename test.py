import numpy as np
import re
import matplotlib.pyplot as plt
import requests
def send_speculative_action(host, port, action):
    """向服务器发送speculative_action请求"""
    url = f"http://{host}:{port}/speculative_action"
    data = {"action": action}
    
    # 创建一个会话对象，显式禁用所有代理
    session = requests.Session()
    session.trust_env = False  # 不使用环境变量中的代理设置
    
    try:
        print(f"正在向 {url} 发送请求，action={action}...")
        # 使用会话发送请求，并明确设置proxies为空字典
        response = session.post(url, json=data, proxies={})
        
        if response.status_code == 200:
            print(f"请求成功！响应状态码: {response.status_code}")
        else:
            print(f"请求失败。响应状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
    except Exception as e:
        print(f"发送请求时出错: {e}")
        return False
    
    return response.status_code == 200

send_speculative_action("localhost", 8000, 0)


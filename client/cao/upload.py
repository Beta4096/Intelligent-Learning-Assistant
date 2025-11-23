import base64
import json
import os
import shutil
import time
from tkinter import filedialog

import requests

SERVER_BASE_URL = os.getenv("SERVER_BASE_URL", "https://your-server.example.com")
TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "10.0"))


def select_source():
    """选择源文件"""
    file_path = filedialog.askopenfilename(
        title="选择要上传的文件",
        filetypes=[("所有文件", "*.*")]  #根据实现改
    )
    if file_path:
        return file_path
    return None

def upload_file(token,file_path:str):
    """上传文件给服务器并复制一份到textbook文件夹，带id，处理同名文件逻辑同浏览器"""
    if not os.path.exists(file_path):
        return {"success": False,"msg": "文件不存在"}

    url = f"{SERVER_BASE_URL}"
    with open(file_path, 'rb') as file:
        file_content = file.read()
        base64_data = base64.b64encode(file_content).decode('utf-8')

    target_dir = os.path.join(os.getcwd(), "textbook")
    os.makedirs(target_dir, exist_ok=True)

    # 1. 复制文件到 textbook 目录
    filename = os.path.basename(file_path)
    name, ext = os.path.splitext(filename)
    i = 1
    while os.path.exists(os.path.join(target_dir, filename)):
        filename = f"{name}({i}){ext}"
        i += 1
    # 复制文件
    dst_path = os.path.join(target_dir, filename)
    shutil.copy2(file_path, dst_path)

    # 生成唯一编号
    unique_id = f"{int(time.time()*1e6)}"

    # 写隐藏元数据文件
    meta_path = os.path.join(target_dir, f".{filename}.meta")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"id": unique_id, "original": filename, "timestamp": time.ctime()}, f)


    payload = {
        "type": "upload",
        "token":token,
        "textbook":{
            "name":filename,
            "file_id":unique_id,
            "data":base64_data
        }
    }

    try:
        resp = requests.post(url, json=payload, timeout=TIMEOUT)
    except requests.RequestException as e:
        return {"success": False,"msg": "发生错误，错误类型："+str(e)}

    if resp.status_code == 200:
        return {"success": True,"msg": "上传成功！"}
    else:
        return {"success": False,"msg": "上传失败！"}

def delete(token,file_path):
    if not os.path.exists(file_path):
        return {"success": False,"msg": "文件不存在"}
    filename = os.path.basename(file_path)
    meta_path = os.path.join(os.path.dirname(file_path), f".{filename}.meta")

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {"success": False,"msg": "发生错误，错误类型："+str(e)}

    file_id = data.get("id")
    payload = {
        "type": "delete",
        "token":token,
        "file_id":file_id
    }
    url = f"{SERVER_BASE_URL}"
    try:
        resp = requests.post(url, json=payload, timeout=TIMEOUT)
    except requests.RequestException as e:
        return {"success": False,"msg": "发生错误，错误类型："+str(e)}

    if resp.status_code == 200:
        os.remove(file_path)
        return {"success": True,"msg": "删除成功！"}
    else:
        return {"success": False,"msg": "删除失败！"}
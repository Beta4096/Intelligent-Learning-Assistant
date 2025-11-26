import os
import requests
from flask import jsonify
from dotenv import load_dotenv

load_dotenv()

NOTION_VERSION = "2022-06-28"
CLIENT_ID = os.getenv("NOTION_CLIENT_ID")
CLIENT_SECRET = os.getenv("NOTION_CLIENT_SECRET")
REDIRECT_URI = "http://localhost:8765/callback"   # 必须与客户端一致！


def notion_headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json"
    }


# ---------- 生成授权链接（逻辑函数） ----------
def get_auth_url():
    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "owner": "user",
        "redirect_uri": REDIRECT_URI,
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    auth_url = f"https://api.notion.com/v1/oauth/authorize?{query}"

    return jsonify({"auth_url": auth_url})


def find_learning_assistant_page(token):
    url = "https://api.notion.com/v1/search"
    payload = {"query": "Learning Assistant"}
    r = requests.post(url, json=payload, headers=notion_headers(token))
    r.raise_for_status()
    results = r.json().get("results", [])
    for page in results:
        if page.get("object") != "page":
            continue
        props = page.get("properties", {})
        title_prop = None
        # 找到一个 type 为 "title" 的属性（一般就是名字叫 "title" 的那个）
        for prop in props.values():
            if isinstance(prop, dict) and prop.get("type") == "title":
                title_prop = prop
                break
        if not title_prop:
            continue
        title_rich = title_prop.get("title", [])
        if not title_rich:
            continue
        # 尝试从 plain_text 或 text.content 拿字符串
        first = title_rich[0]
        title_text = first.get("plain_text") or first.get("text", {}).get("content", "")
        if title_text == "Learning Assistant":
            return page["id"]
    return None


def create_container_page(token):
    url = "https://api.notion.com/v1/pages"
    payload = {
        "parent": {"type": "workspace", "workspace": True},
        "properties": {
            "title": [{"type": "text", "text": {"content": "Learning Assistant"}}]
        }
    }
    r = requests.post(url, json=payload, headers=notion_headers(token))
    r.raise_for_status()
    return r.json()["id"]


def create_child_page(token, parent_id):
    url = "https://api.notion.com/v1/pages"
    payload = {
        "parent": {"page_id": parent_id},
        "properties": {
            "title": [{"type": "text", "text": {"content": "New Export"}}]
        }
    }
    r = requests.post(url, json=payload, headers=notion_headers(token))
    r.raise_for_status()
    return r.json()["id"]


def exchange_code_for_token(code):
    resp = requests.post(
        "https://api.notion.com/v1/oauth/token",
        json={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
        },
        auth=(CLIENT_ID, CLIENT_SECRET),
        timeout=30,
    )
    data = resp.json()
    if "access_token" not in data:
        raise RuntimeError(f"Token exchange failed: {data}")
    return data["access_token"]


# ---------- 主导出逻辑（给 app.py 调用） ----------
def handle_export(data):
    app_user_token = data.get("token")
    code = data.get("code")

    if not app_user_token or not code:
        return jsonify({"success": False, "msg": "missing token or code"}), 400

    # 1. server 用 secret 换 token（客户端不能做）
    access_token = exchange_code_for_token(code)

    # 2. 找容器页，没有就创建
    container_id = find_learning_assistant_page(access_token)
    if not container_id:
        container_id = create_container_page(access_token)

    # 3. 创建条目
    entry_id = create_child_page(access_token, container_id)

    return jsonify({
        "success": True,
        "container_id": container_id,
        "entry_id": entry_id
    })

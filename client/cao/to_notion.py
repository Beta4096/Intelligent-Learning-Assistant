import os
import threading
import webbrowser
from urllib.parse import urlencode

import requests
from flask import Flask, request

NOTION_VERSION = "2022-06-28"

app = Flask(__name__)
OAUTH_RESULT = {"access_token": None, "error": None}


def build_auth_url(client_id: str, redirect_uri: str) -> str:
    """
    Notion OAuth 授权链接
    """
    params = {
        "client_id": client_id,
        "response_type": "code",
        "owner": "user",
        "redirect_uri": redirect_uri,
    }
    return "https://api.notion.com/v1/oauth/authorize?" + urlencode(params)


def exchange_code_for_token(code: str, client_id: str, client_secret: str, redirect_uri: str) -> str:
    """
    用 code 换 access_token
    """
    resp = requests.post(
        "https://api.notion.com/v1/oauth/token",
        json={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
        },
        auth=(client_id, client_secret),
        timeout=30,
    )
    data = resp.json()
    if "access_token" not in data:
        raise RuntimeError(f"Token exchange failed: {data}")
    return data["access_token"]


def notion_headers(token: str):
    return {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def create_container_page_in_private(token: str, title="Learning Assistant") -> str:
    """
    在 workspace 私人空间创建一个顶层 private page（容器页）
    """
    url = "https://api.notion.com/v1/pages"
    payload = {
        "parent": {"type": "workspace", "workspace": True},
        "properties": {
            "title": [{"type": "text", "text": {"content": title}}]
        }
    }
    r = requests.post(url, json=payload, headers=notion_headers(token), timeout=30)
    r.raise_for_status()
    return r.json()["id"]


def create_child_page_and_write(token: str, parent_page_id: str, child_title="New Export") -> str:
    """
    在容器页下创建子页面，并写入 blocks
    """
    url = "https://api.notion.com/v1/pages"
    payload = {
        "parent": {"page_id": parent_page_id},
        "properties": {
            "title": [{"type": "text", "text": {"content": child_title}}]
        },
        "children": [
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": "Export Result"}}]
                }
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": "Hello from OAuth + Notion API!"}}]
                }
            }
        ]
    }
    r = requests.post(url, json=payload, headers=notion_headers(token), timeout=30)
    r.raise_for_status()
    return r.json()["id"]

def export_flow(token: str):
    """
    授权成功后执行：创建 Learning Assistant 容器页 + 写一条新信息
    """
    add_learning_assistant_entry(
        token,
        entry_title="Learning Log",
        entry_text="This is a new exported record."
    )


@app.route("/callback")
def callback():
    """
    Notion OAuth 回调：只接 code，不在这里换 token（换 token 在主线程做）
    """
    err = request.args.get("error")
    code = request.args.get("code")

    if err:
        OAUTH_RESULT["error"] = err
        return f"Authorization failed: {err}", 400

    if not code:
        OAUTH_RESULT["error"] = "No code in callback."
        return "No code provided.", 400

    # 把 code 存起来给主线程用
    OAUTH_RESULT["code"] = code
    return "Authorization succeeded. You can close this tab now."


def run_server(port: int):
    app.run(host="localhost", port=port, debug=False)


def notion_oauth_and_export(client_id: str, client_secret: str, redirect_uri: str):
    """
    你要的主入口：
      1) 打开网页授权
      2) 回调拿 code
      3) 换 token
      4) 写 Notion
    """
    # 从 redirect_uri 解析端口（简单处理：假设是 http://localhost:PORT/callback）
    try:
        port = int(redirect_uri.split(":")[2].split("/")[0])
    except Exception:
        raise RuntimeError("redirect_uri must look like http://localhost:PORT/callback")

    # 1) 启动本地回调 server
    server_thread = threading.Thread(target=run_server, args=(port,), daemon=True)
    server_thread.start()

    # 2) 打开授权页
    auth_url = build_auth_url(client_id, redirect_uri)
    print("Opening browser for Notion OAuth...")
    webbrowser.open(auth_url)

    # 3) 等回调拿 code
    print("Waiting for authorization callback...")
    while "code" not in OAUTH_RESULT and OAUTH_RESULT["error"] is None:
        pass

    if OAUTH_RESULT["error"]:
        raise RuntimeError(OAUTH_RESULT["error"])

    code = OAUTH_RESULT["code"]
    print("[OAuth] code obtained.")

    # 4) 换 token
    token = exchange_code_for_token(code, client_id, client_secret, redirect_uri)
    print("[OAuth] access_token obtained.")

    # 5) 执行导出写入
    export_flow(token)
def add_learning_assistant_entry(token: str, entry_title="New Entry", entry_text="Hello!"):
    """
    在私人空间创建 Learning Assistant 容器页
    然后在其下创建一个子页面写入一条信息
    """
    print("[LA] Creating 'Learning Assistant' container page in Private workspace...")
    container_id = create_container_page_in_private(token, title="Learning Assistant")

    print("[LA] Creating a child entry page...")
    child_id = create_child_page_and_write(
        token,
        container_id,
        child_title=entry_title
    )


    print("✅ Learning Assistant updated.")
    print("Container Page ID:", container_id)
    print("Entry Page ID:", child_id)
    return container_id, child_id


def main():
    client_id = os.getenv("NOTION_CLIENT_ID")
    client_secret = os.getenv("NOTION_CLIENT_SECRET")
    redirect_uri = "http://localhost:8765/callback"

    if "YOUR_" in client_id or "YOUR_" in client_secret:
        raise RuntimeError("Please fill client_id / client_secret / redirect_uri in main().")

    notion_oauth_and_export(client_id, client_secret, redirect_uri)


if __name__ == "__main__":
    main()

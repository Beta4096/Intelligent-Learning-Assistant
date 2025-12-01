import os
import webbrowser
import requests
import threading
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from dotenv import load_dotenv
load_dotenv()

SERVER_BASE_URL = os.getenv("SERVER_BASE_URL", "https://your-server.example.com")

class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        query = parse_qs(urlparse(self.path).query)
        code = query.get("code", [None])[0]

        if code:
            requests.post(f"{SERVER_BASE_URL}/notion/export", json={
                "code": code,
                "token": self.server.app_token,
            })
            msg = "授权成功！你可以关闭此窗口。"
            self.server.done = True  # 👈 设置退出标记
        else:
            msg = "未收到 code"

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(msg.encode())

    def log_message(self, *args):
        return



def start_callback_server(token, port=8765):
    server = ThreadingHTTPServer(("localhost", port), CallbackHandler)
    server.app_token = token

    print(f"[OK] 本地回调服务器已启动：http://localhost:{port}")

    # ⚡ 将 serve_forever() 放进 thread，并把 thread 返回
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()

    return server, thread


def export_cli(token):
    server, thread = start_callback_server(token)

    auth_url = requests.get(f"{SERVER_BASE_URL}/notion/auth_url").json()["auth_url"]
    webbrowser.open(auth_url)
    print("打开浏览器进行授权…")
    print("等待 Notion 回调……")

    # 👇 主线程轮询检查 done
    import time
    while not getattr(server, "done", False):
        time.sleep(0.1)

    # 收到回调 → 手动关闭服务器
    server.shutdown()
    server.server_close()

    # 等待线程真正结束
    #thread.join()
    print("授权流程结束，客户端程序退出。")



if __name__ == "__main__":
    export_cli("ODc9dBlSg0hvOGH1R8NKZekBrYmEMVJIpRYUwoH8pvo")

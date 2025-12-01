import React, { useEffect, useState } from "react";
import { apiRegister, apiLogin, apiQuestion, apiExport,apiUploadTextbook  } from "./api";

// ----- 把后端的 content 解析为前端消息格式 -----
function normalizeHistoryContent(content = []) {
  return content
    .map((item) => {
      const { timestamp, role, payload } = item;

      let text = "";
      if (Array.isArray(payload)) {
        for (const p of payload) {
          if (p.text) text += p.text;
        }
      } else if (payload?.text) {
        text = payload.text;
      }

      return {
        timestamp,
        role: role === "LLM" ? "assistant" : "user",
        text,
      };
    })
    .sort((a, b) => Date.parse(a.timestamp) - Date.parse(b.timestamp));
}

// ---------------- 登录页 ----------------
function LoginPage({ onLoginSuccess }) {
  const [mode, setMode] = useState("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  async function submit(e) {
    e.preventDefault();
    setErr("");
    setLoading(true);

    try {
      if (mode === "register") {
        const res = await apiRegister(username, password, confirm);
        if (!res.success) return setErr(res.msg);

        const token = res.msg.token;
        onLoginSuccess({ token, historyMessages: [] });
      } else {
        const res = await apiLogin(username, password);
        if (!res.success) return setErr(res.msg);

        const msg = res.msg;
        const token = msg.token;
        const content = msg.content || [];
        onLoginSuccess({
          token,
          historyMessages: normalizeHistoryContent(content),
        });
      }
    } catch (err) {
      setErr("网络错误");
    }

    setLoading(false);
  }

  return (
    <div style={{ width: 300, margin: "80px auto" }}>
      <h2>{mode === "login" ? "登录" : "注册"}</h2>

      <form onSubmit={submit}>
        <input
          placeholder="用户名"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          style={{ width: "100%", marginBottom: 8, padding: 6 }}
        />
        <input
          type="password"
          placeholder="密码"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          style={{ width: "100%", marginBottom: 8, padding: 6 }}
        />
        {mode === "register" && (
          <input
            type="password"
            placeholder="确认密码"
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
            style={{ width: "100%", marginBottom: 8, padding: 6 }}
          />
        )}

        {err && <div style={{ color: "red", marginBottom: 8 }}>{err}</div>}

        <button
          type="submit"
          style={{ width: "100%", padding: 8 }}
          disabled={loading}
        >
          {loading ? "处理中..." : mode === "login" ? "登录" : "注册并登录"}
        </button>
      </form>

      <div style={{ marginTop: 12 }}>
        {mode === "login" ? (
          <span>
            没账号？
            <button
              onClick={() => setMode("register")}
              style={{ color: "blue", border: "none", background: "none" }}
            >
              注册
            </button>
          </span>
        ) : (
          <span>
            已有账号？
            <button
              onClick={() => setMode("login")}
              style={{ color: "blue", border: "none", background: "none" }}
            >
              登录
            </button>
          </span>
        )}
      </div>
    </div>
  );
}

// ---------------- 聊天页（带文件上传） ----------------
function ChatPage({ token, initialMessages, logout }) {
  const [messages, setMessages] = useState(initialMessages);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [files, setFiles] = useState([]);  // 新增：存待上传的文件 Base64 列表
  // 新增：教材上传状态
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");

  async function handleUploadTextbook() {
    if (!uploadFile) return;
    setUploadStatus("上传中...");

    try {
      const res = await apiUploadTextbook(token, uploadFile);
      if (!res.success) {
        setUploadStatus("错误：" + (res.msg || "上传失败"));
      } else {
        setUploadStatus("上传成功，服务器已解析！");
      }
    } catch (e) {
      setUploadStatus("网络错误：" + e.message);
    }
  }
  // 将文件转为 Base64
  function fileToBase64(file) {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result.split(",")[1]); // 去掉 data:xxx;base64,
      reader.readAsDataURL(file);
    });
  }

  async function handleFileSelect(e) {
    const fileList = e.target.files;
    const newFiles = [];

    for (let file of fileList) {
      const base64 = await fileToBase64(file);
      newFiles.push({
        name: file.name,
        base64,
      });
    }

    setFiles(newFiles);
  }

  async function sendMessage() {
    if (!input.trim() && files.length === 0) return;

    const text = input.trim();
    setInput("");

    // 1) push 用户消息（携带是否上传文件）
    const userMsg = {
      role: "user",
      text: text || "(发送了文件)",
      timestamp: new Date().toISOString(),
    };
    setMessages((m) => [...m, userMsg]);

    setSending(true);

    // 2) 构造 images payload
    const imagesPayload = files.map((f) => ({
      image: f.base64,
    }));

    // 3) 清空文件缓存
    setFiles([]);

    // 4) 发送到后端
    const res = await apiQuestion(token, text, imagesPayload);

    setSending(false);

    if (!res.success) {
      setMessages((m) => [
        ...m,
        {
          role: "system",
          text: "错误：" + res.msg,
          timestamp: new Date().toISOString(),
        },
      ]);
      return;
    }

    // 5) push LLM 回复
    const replyMsg = {
      role: "assistant",
      text: res.msg,
      timestamp: new Date().toISOString(),
    };
    setMessages((m) => [...m, replyMsg]);
  }

return (
  <div style={{ maxWidth: 700, margin: "20px auto" }}>
    <button onClick={logout} style={{ marginBottom: 10 }}>
      退出登录
    </button>

    {/* 上传教材区域 */}
    <div
      style={{
        border: "1px solid #aaa",
        padding: 10,
        borderRadius: 6,
        marginBottom: 10,
      }}
    >
      <h3>上传教材（用于后续问答）</h3>
      <input
        type="file"
        accept=".pdf,.doc,.docx,.txt"
        onChange={(e) => setUploadFile(e.target.files[0])}
      />
      <button
        style={{ marginLeft: 8 }}
        disabled={!uploadFile}
        onClick={handleUploadTextbook}
      >
        上传
      </button>
      {uploadStatus && (
        <div style={{ marginTop: 6, fontSize: 14 }}>{uploadStatus}</div>
      )}
    </div>

    {/* 下面是原来的聊天区 + 输入区 */}
    <div
      style={{
        height: "60vh",
        overflowY: "scroll",
        border: "1px solid #ddd",
        padding: 10,
        marginBottom: 10,
      }}
    >
      {messages.map((msg, i) => (
        <div key={i} style={{ marginBottom: 8 }}>
          <b>{msg.role}:</b> {msg.text}
        </div>
      ))}
    </div>

    <textarea
      rows={3}
      value={input}
      onChange={(e) => setInput(e.target.value)}
      style={{ width: "100%", marginBottom: 8 }}
      placeholder="输入你的问题..."
    />
    <button onClick={sendMessage} disabled={sending}>
      {sending ? "发送中..." : "发送"}
    </button>
  </div>
);

}


// --------------- 主组件 ---------------
export default function App() {
  const [token, setToken] = useState(
    localStorage.getItem("ila_token") || null
  );
  const [history, setHistory] = useState(
    JSON.parse(localStorage.getItem("ila_history") || "[]")
  );

  function handleLogin({ token, historyMessages }) {
    setToken(token);
    setHistory(historyMessages);

    localStorage.setItem("ila_token", token);
    localStorage.setItem("ila_history", JSON.stringify(historyMessages));
  }

  function logout() {
    setToken(null);
    setHistory([]);

    localStorage.removeItem("ila_token");
    localStorage.removeItem("ila_history");
  }

  if (!token) {
    return <LoginPage onLoginSuccess={handleLogin} />;
  }

  return (
    <ChatPage
      token={token}
      initialMessages={history}
      logout={logout}
    />
  );
}

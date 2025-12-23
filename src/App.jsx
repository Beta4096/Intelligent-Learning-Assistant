// import React, { useEffect, useState } from "react";
// import { apiRegister, apiLogin, apiQuestion, apiExport,apiUploadTextbook  } from "./api";
//
// // ----- 把后端的 content 解析为前端消息格式 -----
// function normalizeHistoryContent(content = []) {
//   return content
//     .map((item) => {
//       const { timestamp, role, payload } = item;
//
//       let text = "";
//       if (Array.isArray(payload)) {
//         for (const p of payload) {
//           if (p.text) text += p.text;
//         }
//       } else if (payload?.text) {
//         text = payload.text;
//       }
//
//       return {
//         timestamp,
//         role: role === "LLM" ? "assistant" : "user",
//         text,
//       };
//     })
//     .sort((a, b) => Date.parse(a.timestamp) - Date.parse(b.timestamp));
// }
//
// // ---------------- 登录页 ----------------
// function LoginPage({ onLoginSuccess }) {
//   const [mode, setMode] = useState("login");
//   const [username, setUsername] = useState("");
//   const [password, setPassword] = useState("");
//   const [confirm, setConfirm] = useState("");
//   const [loading, setLoading] = useState(false);
//   const [err, setErr] = useState("");
//
//   async function submit(e) {
//     e.preventDefault();
//     setErr("");
//     setLoading(true);
//
//     try {
//       if (mode === "register") {
//         const res = await apiRegister(username, password, confirm);
//         if (!res.success) return setErr(res.msg);
//
//         const token = res.msg.token;
//         onLoginSuccess({ token, historyMessages: [] });
//       } else {
//         const res = await apiLogin(username, password);
//         if (!res.success) return setErr(res.msg);
//
//         const msg = res.msg;
//         const token = msg.token;
//         const content = msg.content || [];
//         onLoginSuccess({
//           token,
//           historyMessages: normalizeHistoryContent(content),
//         });
//       }
//     } catch (err) {
//       setErr("网络错误");
//     }
//
//     setLoading(false);
//   }
//
//   return (
//     <div style={{ width: 300, margin: "80px auto" }}>
//       <h2>{mode === "login" ? "登录" : "注册"}</h2>
//
//       <form onSubmit={submit}>
//         <input
//           placeholder="用户名"
//           value={username}
//           onChange={(e) => setUsername(e.target.value)}
//           style={{ width: "100%", marginBottom: 8, padding: 6 }}
//         />
//         <input
//           type="password"
//           placeholder="密码"
//           value={password}
//           onChange={(e) => setPassword(e.target.value)}
//           style={{ width: "100%", marginBottom: 8, padding: 6 }}
//         />
//         {mode === "register" && (
//           <input
//             type="password"
//             placeholder="确认密码"
//             value={confirm}
//             onChange={(e) => setConfirm(e.target.value)}
//             style={{ width: "100%", marginBottom: 8, padding: 6 }}
//           />
//         )}
//
//         {err && <div style={{ color: "red", marginBottom: 8 }}>{err}</div>}
//
//         <button
//           type="submit"
//           style={{ width: "100%", padding: 8 }}
//           disabled={loading}
//         >
//           {loading ? "处理中..." : mode === "login" ? "登录" : "注册并登录"}
//         </button>
//       </form>
//
//       <div style={{ marginTop: 12 }}>
//         {mode === "login" ? (
//           <span>
//             没账号？
//             <button
//               onClick={() => setMode("register")}
//               style={{ color: "blue", border: "none", background: "none" }}
//             >
//               注册
//             </button>
//           </span>
//         ) : (
//           <span>
//             已有账号？
//             <button
//               onClick={() => setMode("login")}
//               style={{ color: "blue", border: "none", background: "none" }}
//             >
//               登录
//             </button>
//           </span>
//         )}
//       </div>
//     </div>
//   );
// }
//
// // ---------------- 聊天页（带文件上传） ----------------
// function ChatPage({ token, initialMessages, logout }) {
//     const [exportStatus, setExportStatus] = useState("");
//   const [messages, setMessages] = useState(initialMessages);
//   const [input, setInput] = useState("");
//   const [sending, setSending] = useState(false);
//   const [files, setFiles] = useState([]);  // 新增：存待上传的文件 Base64 列表
//   // 新增：教材上传状态
//   const [uploadFile, setUploadFile] = useState(null);
//   const [uploadStatus, setUploadStatus] = useState("");
//
//   async function handleUploadTextbook() {
//     if (!uploadFile) return;
//     setUploadStatus("上传中...");
//
//     try {
//       const res = await apiUploadTextbook(token, uploadFile);
//       if (!res.success) {
//         setUploadStatus("错误：" + (res.msg || "上传失败"));
//       } else {
//         setUploadStatus("上传成功，服务器已解析！");
//       }
//     } catch (e) {
//       setUploadStatus("网络错误：" + e.message);
//     }
//   }
//   async function handleExport() {
//   setExportStatus("导出中...");
//
//   try {
//     const res = await apiExport(token);
//
//     if (!res.success) {
//       setExportStatus("导出失败：" + (res.msg || "未知错误"));
//     } else {
//       setExportStatus(
//         `导出成功！容器页ID: ${res.container_id}, 条目ID：${res.entry_id}`
//       );
//     }
//   } catch (e) {
//     setExportStatus("网络错误：" + e.message);
//   }
// }
//
//   // 将文件转为 Base64
//   function fileToBase64(file) {
//     return new Promise((resolve) => {
//       const reader = new FileReader();
//       reader.onload = () => resolve(reader.result.split(",")[1]); // 去掉 data:xxx;base64,
//       reader.readAsDataURL(file);
//     });
//   }
//
//   async function handleFileSelect(e) {
//     const fileList = e.target.files;
//     const newFiles = [];
//
//     for (let file of fileList) {
//       const base64 = await fileToBase64(file);
//       newFiles.push({
//         name: file.name,
//         base64,
//       });
//     }
//
//     setFiles(newFiles);
//   }
//
//   async function sendMessage() {
//     if (!input.trim() && files.length === 0) return;
//
//     const text = input.trim();
//     setInput("");
//
//     // 1) push 用户消息（携带是否上传文件）
//     const userMsg = {
//       role: "user",
//       text: text || "(发送了文件)",
//       timestamp: new Date().toISOString(),
//     };
//     setMessages((m) => [...m, userMsg]);
//
//     setSending(true);
//
//     // 2) 构造 images payload
//     const imagesPayload = files.map((f) => ({
//       image: f.base64,
//     }));
//
//     // 3) 清空文件缓存
//     setFiles([]);
//
//     // 4) 发送到后端
//     const res = await apiQuestion(token, text, imagesPayload);
//
//     setSending(false);
//
//     if (!res.success) {
//       setMessages((m) => [
//         ...m,
//         {
//           role: "system",
//           text: "错误：" + res.msg,
//           timestamp: new Date().toISOString(),
//         },
//       ]);
//       return;
//     }
//
//     // 5) push LLM 回复
//     const replyMsg = {
//       role: "assistant",
//       text: res.msg,
//       timestamp: new Date().toISOString(),
//     };
//     setMessages((m) => [...m, replyMsg]);
//   }
//
// return (
//   <div style={{ maxWidth: 700, margin: "20px auto" }}>
//     <button onClick={logout} style={{ marginBottom: 10 }}>
//       退出登录
//     </button>
// <div
//   style={{
//     border: "1px solid #aaa",
//     padding: 10,
//     borderRadius: 6,
//     marginBottom: 10,
//   }}
// >
//   <h3>导出聊天记录到 Notion</h3>
//   <button onClick={handleExport}>导出历史</button>
//   {exportStatus && (
//     <div style={{ marginTop: 6, fontSize: 14 }}>{exportStatus}</div>
//   )}
// </div>
//     {/* 上传教材区域 */}
//     <div
//       style={{
//         border: "1px solid #aaa",
//         padding: 10,
//         borderRadius: 6,
//         marginBottom: 10,
//       }}
//     >
//       <h3>上传教材（用于后续问答）</h3>
//       <input
//         type="file"
//         accept=".pdf,.doc,.docx,.txt"
//         onChange={(e) => setUploadFile(e.target.files[0])}
//       />
//       <button
//         style={{ marginLeft: 8 }}
//         disabled={!uploadFile}
//         onClick={handleUploadTextbook}
//       >
//         上传
//       </button>
//       {uploadStatus && (
//         <div style={{ marginTop: 6, fontSize: 14 }}>{uploadStatus}</div>
//       )}
//     </div>
//
//     {/* 下面是原来的聊天区 + 输入区 */}
//     <div
//       style={{
//         height: "60vh",
//         overflowY: "scroll",
//         border: "1px solid #ddd",
//         padding: 10,
//         marginBottom: 10,
//       }}
//     >
//       {messages.map((msg, i) => (
//         <div key={i} style={{ marginBottom: 8 }}>
//           <b>{msg.role}:</b> {msg.text}
//         </div>
//       ))}
//     </div>
//
//     <textarea
//       rows={3}
//       value={input}
//       onChange={(e) => setInput(e.target.value)}
//       style={{ width: "100%", marginBottom: 8 }}
//       placeholder="输入你的问题..."
//     />
//     <button onClick={sendMessage} disabled={sending}>
//       {sending ? "发送中..." : "发送"}
//     </button>
//   </div>
// );
//
// }
//
//
// // --------------- 主组件 ---------------
// export default function App() {
//   const [token, setToken] = useState(
//     localStorage.getItem("ila_token") || null
//   );
//   const [history, setHistory] = useState(
//     JSON.parse(localStorage.getItem("ila_history") || "[]")
//   );
//
//   function handleLogin({ token, historyMessages }) {
//     setToken(token);
//     setHistory(historyMessages);
//
//     localStorage.setItem("ila_token", token);
//     localStorage.setItem("ila_history", JSON.stringify(historyMessages));
//   }
//
//   function logout() {
//     setToken(null);
//     setHistory([]);
//
//     localStorage.removeItem("ila_token");
//     localStorage.removeItem("ila_history");
//   }
//
//   if (!token) {
//     return <LoginPage onLoginSuccess={handleLogin} />;
//   }
//
//   return (
//     <ChatPage
//       token={token}
//       initialMessages={history}
//       logout={logout}
//     />
//   );
// }
import React, { useEffect, useMemo, useState } from "react";
import { apiRegister, apiLogin, apiQuestion, apiExport, apiUploadTextbook } from "./api";

/**
 * 后端 history item 结构（按你 login_handler / question_handler 推断）：
 * {
 *   timestamp: "...",
 *   role: "USER" | "LLM" | ...,
 *   payload: [{text: "..."}] | {text:"..."} | ...
 *   session_id: number
 * }
 */
function extractTextFromPayload(payload) {
  if (!payload) return "";
  if (Array.isArray(payload)) {
    let s = "";
    for (const p of payload) if (p?.text) s += p.text;
    return s;
  }
  if (typeof payload === "object" && payload.text) return String(payload.text);
  if (typeof payload === "string") return payload;
  return "";
}

function normalizeHistory(history = []) {
  return (history || [])
    .map((item) => {
      const timestamp = item.timestamp || new Date().toISOString();
      const roleRaw = item.role || "USER";
      const role = roleRaw === "LLM" ? "assistant" : roleRaw === "USER" ? "user" : "system";
      const text = extractTextFromPayload(item.payload);
      const sessionId = Number(item.session_id ?? 1);
      return { timestamp, role, text, sessionId };
    })
    .sort((a, b) => Date.parse(a.timestamp) - Date.parse(b.timestamp));
}

function groupBySession(messages) {
  const map = new Map(); // sid -> msgs[]
  for (const m of messages) {
    const sid = m.sessionId;
    if (!map.has(sid)) map.set(sid, []);
    map.get(sid).push(m);
  }
  // sid desc (最新在上)
  const sessionIds = Array.from(map.keys()).sort((a, b) => b - a);
  return { map, sessionIds };
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const s = String(reader.result || "");
      const b64 = s.split(",")[1] || "";
      resolve(b64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
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
        if (!res?.success) {
          setErr(res?.msg || "注册失败");
          setLoading(false);
          return;
        }
        const token = res.msg?.token;
        onLoginSuccess({ token, history: [] });
      } else {
        const res = await apiLogin(username, password);
        if (!res?.success) {
          setErr(res?.msg || "登录失败");
          setLoading(false);
          return;
        }
        // submit_auth 的返回： {success:true, msg:data} 其中 data 是服务器 json :contentReference[oaicite:4]{index=4}
        const data = res.msg || {};
        const token = data.token;
        const history = data.history || []; // login_handler 返回 history :contentReference[oaicite:5]{index=5}
        onLoginSuccess({ token, history });
      }
    } catch (e) {
      setErr("网络错误");
    }

    setLoading(false);
  }

  return (
    <div style={{ width: 320, margin: "80px auto" }}>
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

        <button type="submit" style={{ width: "100%", padding: 8 }} disabled={loading}>
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

// ---------------- 聊天页（左侧会话 + 图片上传） ----------------
function ChatPage({ token, history, logout }) {
  const normalized = useMemo(() => normalizeHistory(history), [history]);
  const grouped = useMemo(() => groupBySession(normalized), [normalized]);

  // sessions: [{id:number, title:string}]
  const [sessions, setSessions] = useState(() =>
    grouped.sessionIds.map((sid) => ({
      id: sid,
      title: `Session ${sid}`,
    }))
  );

  const [activeSessionId, setActiveSessionId] = useState(() => grouped.sessionIds[0] ?? 1);

  // messagesBySession: Map(sid -> msgs[])
  const [messagesBySession, setMessagesBySession] = useState(() => {
    // 用 plain object 存，避免 Map 在 setState 里易踩坑
    const obj = {};
    for (const sid of grouped.sessionIds) obj[sid] = grouped.map.get(sid);
    // 如果没有历史，至少给一个默认 session
    if (grouped.sessionIds.length === 0) obj[1] = [];
    return obj;
  });

  // 保证 activeSessionId 对应存在
  useEffect(() => {
    if (messagesBySession[activeSessionId] == null) {
      const ids = Object.keys(messagesBySession).map(Number).sort((a, b) => b - a);
      setActiveSessionId(ids[0] ?? 1);
    }
  }, [activeSessionId, messagesBySession]);

  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);

  // 图片 base64 list
  const [images, setImages] = useState([]);

  // 教材上传
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");

  // 导出
  const [exportStatus, setExportStatus] = useState("");

  const activeMessages = messagesBySession[activeSessionId] || [];

  const nextSessionId = useMemo(() => {
    const ids = Object.keys(messagesBySession).map(Number);
    return (ids.length ? Math.max(...ids) : 0) + 1;
  }, [messagesBySession]);

  function createNewSession() {
    const sid = nextSessionId; // 递增 id（你要求前端持有并递增）
    setSessions((prev) => [{ id: sid, title: `Session ${sid}` }, ...prev]);
    setMessagesBySession((prev) => ({ ...prev, [sid]: [] }));
    setActiveSessionId(sid);
  }

  async function handleImageSelect(e) {
    const fileList = Array.from(e.target.files || []);
    const b64s = [];
    for (const f of fileList) b64s.push(await fileToBase64(f));
    setImages((prev) => [...prev, ...b64s]);
    e.target.value = "";
  }

  async function sendMessage() {
    if (!input.trim() && images.length === 0) return;

    const text = input.trim();
    setInput("");

    const now = new Date().toISOString();
    const userMsg = {
      role: "user",
      text: text || "(发送了图片)",
      timestamp: now,
      sessionId: activeSessionId,
    };

    // 先本地入队
    setMessagesBySession((prev) => ({
      ...prev,
      [activeSessionId]: [...(prev[activeSessionId] || []), userMsg],
    }));

    setSending(true);
    try {
      // 直接发 base64 list + session_id（api.js 已支持）:contentReference[oaicite:6]{index=6}
      const res = await apiQuestion(token, text, images, activeSessionId);
      setImages([]);

      if (!res?.success) {
        const errMsg = {
          role: "system",
          text: "错误：" + (res?.msg || "未知错误"),
          timestamp: new Date().toISOString(),
          sessionId: activeSessionId,
        };
        setMessagesBySession((prev) => ({
          ...prev,
          [activeSessionId]: [...(prev[activeSessionId] || []), errMsg],
        }));
        setSending(false);
        return;
      }

      // 你说“服务器返回值带 id”：
      // 我这里兼容两种：res.msg 为字符串；或 res.msg = {text, session_id}
      const payload = res.msg;
      const replyText = typeof payload === "string" ? payload : payload?.text || JSON.stringify(payload);
      const sidFromServer = typeof payload === "object" && payload?.session_id != null ? Number(payload.session_id) : activeSessionId;

      const replyMsg = {
        role: "assistant",
        text: replyText,
        timestamp: new Date().toISOString(),
        sessionId: sidFromServer,
      };

      setMessagesBySession((prev) => {
        const targetSid = sidFromServer;
        const existed = prev[targetSid] || [];
        const next = { ...prev, [targetSid]: [...existed, replyMsg] };
        return next;
      });

      // 如果服务端返回了一个新 session_id（理论上你说由前端递增，但这里也兼容）
      if (sidFromServer !== activeSessionId) {
        setActiveSessionId(sidFromServer);
        setSessions((prev) => {
          if (prev.some((s) => s.id === sidFromServer)) return prev;
          return [{ id: sidFromServer, title: `Session ${sidFromServer}` }, ...prev];
        });
      }
    } catch (e) {
      const errMsg = {
        role: "system",
        text: "网络错误：" + (e?.message || String(e)),
        timestamp: new Date().toISOString(),
        sessionId: activeSessionId,
      };
      setMessagesBySession((prev) => ({
        ...prev,
        [activeSessionId]: [...(prev[activeSessionId] || []), errMsg],
      }));
    }
    setSending(false);
  }

  async function handleUploadTextbook() {
    if (!uploadFile) return;
    setUploadStatus("上传中...");
    try {
      const res = await apiUploadTextbook(token, uploadFile);
      if (!res?.success) setUploadStatus("错误：" + (res?.msg || "上传失败"));
      else setUploadStatus("上传成功，服务器已解析！");
    } catch (e) {
      setUploadStatus("网络错误：" + (e?.message || String(e)));
    }
  }

  async function handleExport() {
    setExportStatus("导出中...");
    try {
      const res = await apiExport(token);
      if (!res?.success) setExportStatus("导出失败：" + (res?.msg || "未知错误"));
      else setExportStatus("导出成功");
    } catch (e) {
      setExportStatus("网络错误：" + (e?.message || String(e)));
    }
  }

  return (
    <div style={{ display: "flex", height: "100vh" }}>
      {/* 左侧会话栏 */}
      <div
        style={{
          width: 260,
          borderRight: "1px solid #ddd",
          padding: 12,
          boxSizing: "border-box",
          display: "flex",
          flexDirection: "column",
          gap: 10,
        }}
      >
        <div style={{ display: "flex", gap: 8 }}>
          <button onClick={createNewSession} style={{ flex: 1 }}>
            + 新建会话
          </button>
          <button onClick={logout} style={{ flex: 1 }}>
            退出
          </button>
        </div>

        <div style={{ fontWeight: 700, marginTop: 6 }}>会话列表</div>
        <div style={{ overflowY: "auto", flex: 1 }}>
          {sessions.map((s) => (
            <div key={s.id} style={{ marginBottom: 6 }}>
              <button
                onClick={() => setActiveSessionId(s.id)}
                style={{
                  width: "100%",
                  textAlign: "left",
                  padding: "8px 10px",
                  borderRadius: 8,
                  border: activeSessionId === s.id ? "2px solid #888" : "1px solid #ccc",
                  background: activeSessionId === s.id ? "#f2f2f2" : "transparent",
                  cursor: "pointer",
                }}
              >
                {s.title}
                <div style={{ fontSize: 12, opacity: 0.7 }}>
                  {((messagesBySession[s.id] || []).length / 2) | 0} 轮
                </div>
              </button>
            </div>
          ))}
        </div>

        {/* 导出 */}
        <div style={{ borderTop: "1px solid #eee", paddingTop: 10 }}>
          <div style={{ fontWeight: 700, marginBottom: 6 }}>Notion 导出</div>
          <button onClick={handleExport} style={{ width: "100%" }}>
            导出历史
          </button>
          {exportStatus && <div style={{ marginTop: 6, fontSize: 12 }}>{exportStatus}</div>}
        </div>
      </div>

      {/* 右侧聊天区 */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
        {/* 上传教材 */}
        <div style={{ borderBottom: "1px solid #ddd", padding: 10 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
            <div style={{ fontWeight: 700 }}>当前：Session {activeSessionId}</div>

            <div style={{ marginLeft: "auto", display: "flex", gap: 8, alignItems: "center" }}>
              <input
                type="file"
                accept=".pdf,.doc,.docx,.txt"
                onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
              />
              <button disabled={!uploadFile} onClick={handleUploadTextbook}>
                上传教材
              </button>
              {uploadStatus && <span style={{ fontSize: 12 }}>{uploadStatus}</span>}
            </div>
          </div>
        </div>

        {/* 消息区 */}
        <div style={{ flex: 1, overflowY: "auto", padding: 12 }}>
          {activeMessages.map((msg, i) => (
            <div key={i} style={{ marginBottom: 10 }}>
              <b>{msg.role}:</b> {msg.text}
              <div style={{ fontSize: 11, opacity: 0.6 }}>{msg.timestamp}</div>
            </div>
          ))}
        </div>

        {/* 输入区 + 图片 */}
        <div style={{ borderTop: "1px solid #ddd", padding: 12 }}>
          <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 8 }}>
            <input type="file" accept="image/*" multiple onChange={handleImageSelect} />
            {images.length > 0 && (
              <div style={{ fontSize: 12 }}>
                已选 {images.length} 张
                <button style={{ marginLeft: 8 }} onClick={() => setImages([])}>
                  清空
                </button>
              </div>
            )}
          </div>

          <textarea
            rows={3}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            style={{ width: "100%", marginBottom: 8 }}
            placeholder="输入你的问题..."
          />
          <button onClick={sendMessage} disabled={sending} style={{ width: 120 }}>
            {sending ? "发送中..." : "发送"}
          </button>
        </div>
      </div>
    </div>
  );
}

// --------------- 主组件 ---------------
export default function App() {
  const [token, setToken] = useState(localStorage.getItem("ila_token") || null);
  const [history, setHistory] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem("ila_history") || "[]");
    } catch {
      return [];
    }
  });

  function handleLogin({ token, history }) {
    setToken(token);
    setHistory(history || []);
    localStorage.setItem("ila_token", token);
    localStorage.setItem("ila_history", JSON.stringify(history || []));
  }

  function logout() {
    setToken(null);
    setHistory([]);
    localStorage.removeItem("ila_token");
    localStorage.removeItem("ila_history");
  }

  if (!token) return <LoginPage onLoginSuccess={handleLogin} />;

  return <ChatPage token={token} history={history} logout={logout} />;
}

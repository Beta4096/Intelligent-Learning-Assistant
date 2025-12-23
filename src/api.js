// src/api.js
const API_BASE = "http://localhost:9876/api"; // 或你的 Flask 客户端后端端口

// 通用 POST 封装
async function postJSON(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}
export async function apiUploadTextbook(token, file) {
  const formData = new FormData();
  formData.append("token", token);
  formData.append("file", file);

  const res = await fetch("http://localhost:9876/api/upload-textbook", {
    method: "POST",
    body: formData,
  });

  return res.json();
}

// 注册
export async function apiRegister(username, password, confirmPassword) {
  return postJSON("/auth", {
    type: "register",
    username,
    password,
    confirm_password: confirmPassword,
  });
}

// 登录
export async function apiLogin(username, password) {
  return postJSON("/auth", {
    type: "login",
    username,
    password,
    confirm_password: password,
  });
}

// 提问
//export async function apiQuestion(token, text) {
//  return postJSON("/question", {
//    token,
//    text,
//    images: [],
//  });
//}
// api.js
export async function apiQuestion(token, text, images = [], session_id = null) {
  return postJSON("/question", {
    token,
    text,
    images,       // 👈 直接带 base64 列表
    session_id,   // 👈 新增：多会话标识
  });
}

// 导出历史
export async function apiExport(token) {
  return postJSON("/export", { token });
}

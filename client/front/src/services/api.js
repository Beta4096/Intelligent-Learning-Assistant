// src/services/api.js
const API_BASE = "http://localhost:8080/api";

// -----------------------------
// 通用请求封装
// -----------------------------
export async function postJSON(url, body) {
  const res = await fetch(`${API_BASE}${url}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  return await res.json();
}

export async function postFormData(url, formData) {
  const res = await fetch(`${API_BASE}${url}`, {
    method: "POST",
    body: formData
  });
  return await res.json();
}


// -----------------------------
// 注册
// -----------------------------
export async function register(username, password, confirm_password) {
  return await postJSON("/auth", {
    type: "register",
    username,
    password,
    confirm_password
  });
}


// -----------------------------
// 登录
// -----------------------------
export async function login(username, password) {
  return await postJSON("/auth", {
    type: "login",
    username,
    password,
    confirm_password: password   // 兼容你服务器结构
  });
}


// -----------------------------
// 上传教材
// file: 前端 <input type="file"> 选出来的 File 对象
// -----------------------------
export async function uploadTextbook(token, file) {
  const form = new FormData();
  form.append("token", token);
  form.append("file", file);
  return await postFormData("/upload-textbook", form);
}


// -----------------------------
// 删除教材（不再使用 path）
// file_id 是客户端存 meta 里解析出的
// -----------------------------
export async function deleteTextbook(token, file_id) {
  return await postJSON("/delete-textbook", {
    token,
    file_id // 新版不再需要路径
  });
}


// -----------------------------
// 问题（支持图片）
// images: File[] 或 base64[] 都行，看你的前端处理方式
// -----------------------------
export async function askQuestion(token, text, images = []) {
  return await postJSON("/question", {
    token,
    text,
    images
  });
}


// -----------------------------
// 导出到 Notion
// -----------------------------
export async function exportToNotion(token) {
  return await postJSON("/export", {
    token
  });
}

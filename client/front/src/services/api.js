// ======================================================
//  最终稳定版 API（完美兼容 json-server）
//  前端可立即运行全部功能
// ======================================================

const API_BASE = "http://localhost:8080";

// ======================================================
// 工具函数：GET 请求（用于 json-server 固定返回的接口）
// ======================================================
async function getJSON(endpoint) {
  try {
    const res = await fetch(`${API_BASE}${endpoint}`);
    return await res.json();
  } catch (err) {
    return { success: false, msg: "网络错误：" + err.message };
  }
}

// ======================================================
// 工具函数：POST 模拟请求（用于 AI / 上传 / 删除）
// 注意：json-server 对 POST 是写入数据库操作，不适合作登录
// ======================================================
async function mockPost(endpoint, body) {
  // 模拟延迟让体验更真实
  await new Promise((r) => setTimeout(r, 200));

  // 返回你 db.json 中写好的内容
  return getJSON(endpoint);
}

// ======================================================
// 1️⃣ 用户登录（最佳 mock 方案）
// json-server 不支持 POST 登录逻辑，因此只能 GET /auth
// ======================================================
export async function loginUser(username, password) {
  const data = await getJSON("/auth");

  // db.json 中写好的固定结构
  if (data && data.type === "history") {
    return {
      success: true,
      token: data.token,
      history: data.content,
    };
  }

  return { success: false, msg: "登录失败（mock 数据格式错误）" };
}

// ======================================================
// 2️⃣ 用户注册（mock 永远成功）
// ======================================================
export async function registerUser(username, password, confirm_password) {
  // 未来接真实后端可改成 POST
  return {
    success: true,
    msg: "注册成功（模拟）",
    token: "mock-register-token",
  };
}

// ======================================================
// 3️⃣ 上传教材（模拟成功）
// ======================================================
export async function uploadTextbook(file, token) {
  return {
    success: true,
    msg: "模拟上传成功",
  };
}

// ======================================================
// 4️⃣ 删除教材（模拟成功）
// ======================================================
export async function deleteTextbook(path, token) {
  return {
    success: true,
    msg: "模拟删除成功",
  };
}

// ======================================================
// 5️⃣ 向 AI 提问（json-server 返回提前写好的回答）
// ======================================================
export async function askQuestion(text, images, token) {
  const data = await getJSON("/question");

  // 返回格式遵循 db.json
  return data;
}

// ======================================================
// 6️⃣ File → Base64 工具函数（用于图片发送）
// ======================================================
export function fileToBase64(file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result); // Base64 string
    reader.readAsDataURL(file);
  });
}

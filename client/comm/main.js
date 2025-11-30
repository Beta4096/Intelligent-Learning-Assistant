// main.js

// 1. 后端地址 —— 确认你的 Flask app.py 确实是跑在这个地址上
// 如果你是在别的机器 / 虚拟机里跑，用对应的 IP+端口改掉这里
const API_BASE = "http://127.0.0.1:8080/api";

// 2. 通用 POST JSON 封装，加上更详细的错误信息
async function postJSON(url, body) {
  let res;
  try {
    res = await fetch(`${API_BASE}${url}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      // 如果需要带 cookie / 认证头，可在这里加
      body: JSON.stringify(body),
    });
  } catch (err) {
    // 这里是“真正的 Failed to fetch”来源：网络错误 / 被浏览器拦截
    console.error("网络层错误:", err);
    throw new Error("网络请求失败（可能是端口/地址错误或被浏览器拦截）: " + err.message);
  }

  if (!res.ok) {
    // 能拿到 status 说明至少连上服务器了
    const text = await res.text().catch(() => "");
    console.error("HTTP 错误:", res.status, res.statusText, text);
    throw new Error(`HTTP ${res.status} ${res.statusText}\n${text}`);
  }

  // 正常解析 JSON
  return res.json();
}

// 3. 调用“上传教材”接口
// 注意：这里 body 的字段名我用的是 path，和你 api.js 示例保持一致
async function uploadTextbook(token, filePath) {
  return postJSON("/upload-textbook", {
    token: token,
    path: filePath,   // 关键：和后端、api.js 一致，不再用 file_path
  });
}

document.addEventListener("DOMContentLoaded", () => {
  const tokenInput = document.getElementById("token");
  const filePathInput = document.getElementById("filePath");
  const uploadBtn = document.getElementById("uploadBtn");
  const resultDiv = document.getElementById("result");

  uploadBtn.addEventListener("click", async () => {
    const token = tokenInput.value.trim();
    const filePath = filePathInput.value.trim();

    if (!token) {
      resultDiv.textContent = "请先填写 token";
      return;
    }
    if (!filePath) {
      resultDiv.textContent = "请先填写文件路径（服务器上的路径字符串）";
      return;
    }

    resultDiv.textContent = "正在上传，请稍候...";

    try {
      const res = await uploadTextbook(token, filePath);
      resultDiv.textContent = JSON.stringify(res, null, 2);
    } catch (err) {
      // 这里能看到更具体的错误信息
      resultDiv.textContent = "请求失败：\n" + err.message;
    }
  });
});

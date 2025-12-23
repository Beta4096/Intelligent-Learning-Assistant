


// const API_KEY = process.env.REACT_APP_OPENAI_API_KEY;
// const BASE_URL = process.env.REACT_APP_BASE_URL;

// /**
//  * @param {Array} messages 
//  * @param {string} modelName 
//  * @returns {Promise<String>} 
//  */
// export const getAIResponse = async (messages, modelName = 'gemini-2.5-pro-preview-06-05') => { 
//   try {
//     const response = await fetch(`${BASE_URL}/chat/completions`, {
//       method: 'POST',
//       headers: {
//         'Content-Type': 'application/json',
//         'Authorization': `Bearer ${API_KEY}`,
//       },
//       body: JSON.stringify({
//         model: modelName, 
//         messages: messages,
//         stream: false,
//       }),
//     });

//     if (!response.ok) {
//       const errorData = await response.json();
//       throw new Error(`API Error: ${response.status} ${response.statusText} - ${errorData.error.message}`);
//     }

//     const data = await response.json();
    
//     if (data.choices && data.choices.length > 0) {
//       return data.choices[0].message.content;
//     } else {
//       throw new Error("API response did not contain a valid choice.");
//     }

//   } catch (error) {
//     console.error("Failed to fetch AI response:", error);
//     return "抱歉，我现在遇到了一点麻烦，暂时无法回复您。";
//   }
// };


// export const getHistoryMessages = () => {
//   return new Promise((resolve) => {
//     setTimeout(() => {
//       resolve([
//         { id: 1, role: 'assistant', content: '您好！我是您的智能学习助手，有什么可以帮助您的吗？' },
//       ]);
//     }, 500);
//   });
// };

// export const uploadFile = (file) => {
//   return new Promise((resolve) => {
//     console.log(`模拟上传文件: ${file.name}`);
//     setTimeout(() => resolve({ success: true }), 1000);
//   });
// };


/**
 * API模块：用于与后端服务器进行所有通信。
 */

// 定义后端的根URL，所有请求都将发送到这里。




const API_BASE_URL = "http://localhost:8080/api";

/**
 * 一个通用的POST请求封装函数，用于简化代码。
 * @param {string} endpoint - 请求的API路径 (例如, "/auth")。
 * @param {object} body - 发送到服务器的JSON对象。
 * @returns {Promise<object>} - 从服务器返回的JSON响应。
 */
async function postJSON(endpoint, body) {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    // 如果服务器返回非2xx的状态码，则抛出错误。
    if (!response.ok) {
      // 尝试解析错误信息，以便更好地调试。
      const errorData = await response.json().catch(() => ({ msg: "无法解析的服务器错误" }));
      throw new Error(`服务器错误: ${response.status} - ${errorData.msg || response.statusText}`);
    }

    return response.json();
  } catch (error) {
    console.error(`请求失败: ${error.message}`);
    // 返回一个统一的错误格式，方便前端处理。
    return { success: false, msg: error.message };
  }
}

/**
 * 用户注册
 * @param {string} username - 用户名。
 * @param {string} password - 密码。
 * @param {string} confirm_password - 确认密码。
 * @returns {Promise<{success: boolean, msg: string, token?: string}>}
 *          - 成功时返回 { success: true, msg: "注册成功", token: "T-xyz789" }
 *          - 失败时返回 { success: false, msg: "错误信息" }
 */
export const registerUser = async (username, password, confirm_password) => {
  const response = await postJSON("/auth", {
    type: "register",
    username,
    password,
    confirm_password,
  });
  // 根据您提供的返回格式进行适配
  if (response.status === 200 && response.type === "reg_done") {
    return { success: true, msg: "注册成功", token: response.token };
  } else {
    return { success: false, msg: response.msg || "注册失败，请稍后再试。" };
  }
};

/**
 * 用户登录
 * @param {string} username - 用户名。
 * @param {string} password - 密码。
 * @returns {Promise<{success: boolean, msg: string, token?: string, history?: Array<object>}>}
 *          - 成功时返回 { success: true, msg: "登录成功", token: "T-abc123", history: [...] }
 *          - 失败时返回 { success: false, msg: "错误信息" }
 */
export const loginUser = async (username, password) => {
  // 注意：登录通常不需要confirm_password，这里根据您的示例添加
  const response = await postJSON("/auth", {
    type: "login",
    username,
    password,
    confirm_password: password 
  });
   // 根据您提供的返回格式进行适配
  if (response.status === 200 && response.type === "history") {
    return { success: true, msg: "登录成功", token: response.token, history: response.content };
  } else {
    return { success: false, msg: response.msg || "登录失败，请检查您的凭据。" };
  }
};

/**
 * 上传教材文件
 * 注意：前端无法直接发送本地文件路径，需要使用FormData来发送文件本身。
 * @param {File} file - 用户选择的文件对象。
 * @param {string} token - 用户的认证令牌。
 * @returns {Promise<{success: boolean, msg: string}>}
 */
export const uploadTextbook = async (file, token) => {
    // 文件上传通常使用 FormData
    const formData = new FormData();
    formData.append('file', file); // 'file' 是后端接收文件时使用的字段名
    formData.append('token', token);

    try {
        const response = await fetch(`${API_BASE_URL}/upload-textbook`, {
            method: "POST",
            body: formData, // 注意：使用FormData时不需要设置Content-Type头，浏览器会自动设置
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ msg: "上传失败" }));
            throw new Error(errorData.msg);
        }

        return await response.json(); // 假设后端返回 { success: true, msg: "上传成功" }
    } catch (error) {
        console.error(`文件上传失败: ${error.message}`);
        return { success: false, msg: error.message };
    }
};


/**
 * 删除已上传的教材
 * @param {string} filePath - 在服务器上唯一标识文件的路径或ID。
 * @param {string} token - 用户的认证令牌。
 * @returns {Promise<{success: boolean, msg: string}>}
 */
export const deleteTextbook = async (filePath, token) => {
  return await postJSON("/delete-textbook", {
    token: token,
    path: filePath, // 这个path应该是服务器能识别的唯一标识
  });
};

/**
 * 向AI提问
 * @param {string} text - 用户的文本问题。
 * @param {Array<string>} images - 图片的Base64编码字符串数组。
 * @param {string} token - 用户的认证令牌。
 * @returns {Promise<object>} - AI的回复。
 */
export const askQuestion = async (text, images, token) => {
  return await postJSON("/question", {
    token: token,
    text: text,
    images: images, //  images 数组应包含图片的Base64编码
  });
};
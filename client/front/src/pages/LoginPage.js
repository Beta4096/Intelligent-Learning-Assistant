// src/pages/LoginPage.js

import React, { useState } from "react";
import { Input, Button, message } from "antd";
import { Link, useNavigate } from "react-router-dom";
import { loginUser } from "../services/api";
import "./AuthPage.css";

const LoginPage = ({ onLogin }) => {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate();

  const handleLogin = async () => {
    if (!username || !password) {
      return message.error("请输入用户名和密码");
    }

    // 调用 json-server mock 登录
    const res = await loginUser(username, password);

    if (res.success) {
      message.success("登录成功");

      // 🌟 关键！把 token 和 history 传回给 App.js
      if (onLogin) {
        onLogin(res.token, res.history);
      }

      localStorage.setItem("token", res.token);

      navigate("/chat");
    } else {
      message.error(res.msg || "登录失败");
    }
  };

  return (
    <div className="auth-container">
      <div className="glass-card">
        <h1 className="auth-title">欢迎回来</h1>
        <p className="auth-subtitle">登录你的智能学习助手</p>

        <Input
          className="auth-input"
          size="large"
          placeholder="用户名（随便填）"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />

        <Input.Password
          className="auth-input"
          size="large"
          placeholder="密码（随便填）"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        <Button
          type="primary"
          className="auth-button"
          size="large"
          onClick={handleLogin}
        >
          登录
        </Button>

        <p className="auth-footer">
          还没有账号？ <Link to="/register">立即注册</Link>
        </p>
      </div>
    </div>
  );
};

export default LoginPage;

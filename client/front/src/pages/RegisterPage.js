import React, { useState } from "react";
import { Input, Button, message } from "antd";
import { Link, useNavigate } from "react-router-dom";
import { registerUser } from "../services/api";
import "./AuthPage.css";

const RegisterPage = () => {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const navigate = useNavigate();

  const handleRegister = async () => {
    if (!username || !password)
      return message.error("请填写完整信息");

    if (password !== confirmPassword)
      return message.error("两次密码不一致");

    const res = await registerUser(username, password, confirmPassword);
    if (res.success) {
      message.success("注册成功，请登录");
      navigate("/login");
    } else {
      message.error(res.msg || "注册失败");
    }
  };

  return (
    <div className="auth-container">
      <div className="glass-card">
        <h1 className="auth-title">注册账号</h1>
        <p className="auth-subtitle">加入智能学习助手</p>

        <Input
          className="auth-input"
          size="large"
          placeholder="设置用户名"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />

        <Input.Password
          className="auth-input"
          size="large"
          placeholder="设置密码"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        <Input.Password
          className="auth-input"
          size="large"
          placeholder="确认密码"
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
        />

        <Button
          type="primary"
          className="auth-button"
          size="large"
          onClick={handleRegister}
        >
          注册
        </Button>

        <p className="auth-footer">
          已有账号？ <Link to="/login">返回登录</Link>
        </p>
      </div>
    </div>
  );
};

export default RegisterPage;

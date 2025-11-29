// src/pages/RegisterPage.js

import React, { useState, useEffect, useRef } from "react";
import { Input, Button, message } from "antd";
import { Link, useNavigate } from "react-router-dom";
import { registerUser } from "../services/api";
import "./AuthPage.css";

const RegisterPage = () => {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const navigate = useNavigate();

  // 🌌 四个 Canvas 引用
  const starCanvas = useRef(null);
  const meteorCanvas = useRef(null);
  const nebulaCanvas = useRef(null);
  const particleCanvas = useRef(null);

  /* ----------------------------------------------------------
     🌌 星空 / 流星 / 星云 / 粒子动画（完整拷贝自 HomePage）
  ---------------------------------------------------------- */
  useEffect(() => {
    const starCtx = starCanvas.current.getContext("2d");
    const meteorCtx = meteorCanvas.current.getContext("2d");
    const nebulaCtx = nebulaCanvas.current.getContext("2d");
    const particleCtx = particleCanvas.current.getContext("2d");

    let w = window.innerWidth;
    let h = window.innerHeight;

    // 设置尺寸
    [starCanvas, meteorCanvas, nebulaCanvas, particleCanvas].forEach((ref) => {
      ref.current.width = w;
      ref.current.height = h;
    });

    /* 🌟 1. 星空 */
    const stars = Array.from({ length: 350 }).map(() => ({
      x: Math.random() * w,
      y: Math.random() * h,
      r: Math.random() * 1.2 + 0.2,
      speed: Math.random() * 0.2 + 0.05,
    }));

    function drawStars() {
      starCtx.clearRect(0, 0, w, h);
      starCtx.fillStyle = "rgba(255,255,255,0.9)";
      stars.forEach((s) => {
        starCtx.beginPath();
        starCtx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        starCtx.fill();
        s.y += s.speed;
        if (s.y > h) {
          s.y = 0;
          s.x = Math.random() * w;
        }
      });
    }

    /* ☄️ 2. 流星 */
    const meteors = [];
    function spawnMeteor() {
      meteors.push({
        x: Math.random() * w,
        y: -20,
        length: Math.random() * 230 + 120,
        speed: Math.random() * 6 + 4,
        opacity: Math.random() * 0.4 + 0.3,
      });
    }

    function drawMeteors() {
      meteorCtx.clearRect(0, 0, w, h);
      meteors.forEach((m, i) => {
        meteorCtx.strokeStyle = `rgba(180,180,255,${m.opacity})`;
        meteorCtx.lineWidth = 2.2;
        meteorCtx.beginPath();
        meteorCtx.moveTo(m.x, m.y);
        meteorCtx.lineTo(m.x - m.length, m.y + m.length * 0.4);
        meteorCtx.stroke();

        m.x -= m.speed;
        m.y += m.speed * 0.4;
        if (m.y > h || m.x < -200) meteors.splice(i, 1);
      });

      if (Math.random() < 0.01) spawnMeteor();
    }

    /* 🌈 3. 星云 */
    function drawNebula() {
      nebulaCtx.clearRect(0, 0, w, h);
      const g = nebulaCtx.createRadialGradient(
        w * 0.65, h * 0.35, 0,
        w * 0.65, h * 0.35, w * 0.8
      );
      g.addColorStop(0, "rgba(120,80,255,0.6)");
      g.addColorStop(0.4, "rgba(80,40,200,0.3)");
      g.addColorStop(1, "rgba(0,0,0,0)");
      nebulaCtx.fillStyle = g;
      nebulaCtx.fillRect(0, 0, w, h);
    }

    /* ✨ 4. 粒子光点 */
    const particles = Array.from({ length: 60 }).map(() => ({
      x: Math.random() * w,
      y: Math.random() * h,
      r: Math.random() * 3 + 1,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      alpha: Math.random() * 0.5 + 0.3,
    }));

    function drawParticles() {
      particleCtx.clearRect(0, 0, w, h);
      particles.forEach((p) => {
        particleCtx.fillStyle = `rgba(180,170,255,${p.alpha})`;
        particleCtx.beginPath();
        particleCtx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        particleCtx.fill();

        p.x += p.vx;
        p.y += p.vy;

        if (p.x < 0 || p.x > w) p.vx *= -1;
        if (p.y < 0 || p.y > h) p.vy *= -1;
      });
    }

    /* 🎞 主循环 */
    function animate() {
      drawStars();
      drawNebula();
      drawParticles();
      drawMeteors();
      requestAnimationFrame(animate);
    }
    animate();

    /* 📐 窗口尺寸变化 */
    window.addEventListener("resize", () => {
      w = window.innerWidth;
      h = window.innerHeight;
      [starCanvas, meteorCanvas, nebulaCanvas, particleCanvas].forEach((ref) => {
        ref.current.width = w;
        ref.current.height = h;
      });
      drawNebula();
    });
  }, []);

  /* ----------------------------------------------------------
     注册逻辑
  ---------------------------------------------------------- */
  const handleRegister = async () => {
    if (!username || !password || !confirmPassword)
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
    <>
      {/* ⭐ 四层 Canvas 星空背景 */}
      <canvas ref={starCanvas} id="auth-stars"></canvas>
      <canvas ref={meteorCanvas} id="auth-meteors"></canvas>
      <canvas ref={nebulaCanvas} id="auth-nebula"></canvas>
      <canvas ref={particleCanvas} id="auth-particles"></canvas>

      {/* ⭐ 注册卡片 */}
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
    </>
  );
};

export default RegisterPage;

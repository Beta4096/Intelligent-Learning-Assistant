// src/components/AppHeader.js

import React from "react";
import { Avatar, Switch } from "antd";
import { UserOutlined } from "@ant-design/icons";
import "./AppHeader.css";

const AppHeader = ({ onLogout, isDarkMode, onToggleTheme }) => {
  return (
    <header className={`app-header ${isDarkMode ? "dark" : ""}`}>
      <div className="left-section">
        <img
          src="/logo192.png"
          alt="logo"
          className="app-logo"
        />
        <h1 className="app-title">智能学习助手</h1>
      </div>

      <div className="right-section">
        <Switch
          checked={isDarkMode}
          onChange={onToggleTheme}
          checkedChildren="🌙"
          unCheckedChildren="☀️"
          className="theme-switch"
        />

        <Avatar
          size={40}
          icon={<UserOutlined />}
          className="user-avatar"
        />

        <button className="logout-btn" onClick={onLogout}>
          登出
        </button>
      </div>
    </header>
  );
};

export default AppHeader;

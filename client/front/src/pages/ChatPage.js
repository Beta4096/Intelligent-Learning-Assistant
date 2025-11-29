// src/pages/ChatPage.js

import React, { useState, useEffect } from "react";
import { Layout, message } from "antd";

import AppHeader from "../components/AppHeader";          // 🌟 新 Header
import Sidebar from "../components/Sidebar";
import MessageList from "../components/MessageList";
import MessageInput from "../components/MessageInput";

import { askQuestion, uploadTextbook } from "../services/api";
import useTypingEffect from "../hooks/useTypingEffect";

const { Content, Sider } = Layout;

const ChatPage = ({ token, history, onLogout }) => {
  const [messages, setMessages] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);

  // AI 打字动画的完整文本
  const [aiFullReply, setAiFullReply] = useState("");

  // 逐字打印效果
  const typingText = useTypingEffect(aiFullReply, 20);

  // 🌙 主题切换状态
  const [isDarkMode, setIsDarkMode] = useState(false);
  const handleToggleTheme = () => {
    setIsDarkMode((prev) => !prev);
  };

  /** 转换历史记录 */
  const formatHistory = (h) => {
    if (!Array.isArray(h)) return [];

    return h.map((msg) => {
      let text = "";
      let image = null;

      if (Array.isArray(msg.payload)) {
        const t = msg.payload.find((p) => p.text);
        const i = msg.payload.find((p) => p.image);
        text = t?.text || "";
        image = i?.image || null;
      } else {
        text = msg.payload?.text || "";
        image = msg.payload?.image || null;
      }

      return {
        id: msg.timestamp,
        sender: msg.role === "user" ? "user" : "ai",
        text,
        image,
      };
    });
  };

  /** 首次加载历史 */
  useEffect(() => {
    setMessages(formatHistory(history));
  }, [history]);

  /** 打字期间更新最后一条消息 */
  useEffect(() => {
    if (!isTyping || typingText === "") return;

    setMessages((prev) => {
      const updated = [...prev];
      if (updated.length > 0) {
        updated[updated.length - 1].text = typingText;
      }
      return updated;
    });
  }, [typingText, isTyping]);

  /** 发送消息 */
  const handleSendMessage = async (text, images = []) => {
    if (!text.trim()) return;

    // 显示用户消息
    const userMsg = {
      id: Date.now(),
      sender: "user",
      text,
      image: images[0] || null,
    };

    setMessages((prev) => [...prev, userMsg]);
    setIsTyping(true);

    // 请求 AI 回复
    const res = await askQuestion(text, images, token);

    if (!res.success) {
      setIsTyping(false);
      return message.error(res.msg || "AI 回复失败");
    }

    // AI 回复文本
    const aiText = res.content?.text || "";

    // 插入一条空 AI 消息，打字动画再填充内容
    const aiMsg = {
      id: Date.now() + 1,
      sender: "ai",
      text: "",
      image: res.content?.image || null,
    };

    setMessages((prev) => [...prev, aiMsg]);

    // 开始逐字打印
    setAiFullReply(aiText);
  };

  /** 上传教材 */
  const handleFileUpload = async (file) => {
    const res = await uploadTextbook(file, token);
    if (res.success) {
      message.success("上传成功");
      setUploadedFiles((prev) => [
        ...prev,
        { uid: Date.now(), name: file.name, status: "done" },
      ]);
    } else {
      message.error(res.msg || "上传失败");
    }
  };

  /** 删除教材 */
  const handleDeleteFile = (item) => {
    setUploadedFiles((prev) => prev.filter((f) => f.uid !== item.uid));
  };

  return (
    <Layout style={{ height: "100vh" }}>
      {/* 🌟 美化后的 Header */}
      <AppHeader
        onLogout={onLogout}
        isDarkMode={isDarkMode}
        onToggleTheme={handleToggleTheme}
      />

      <Layout>
        <Sider width={280} theme="light" style={{ padding: "16px" }}>
          <Sidebar
            uploadedFiles={uploadedFiles}
            onFileUpload={handleFileUpload}
            onDeleteFile={handleDeleteFile}
          />
        </Sider>

        {/* 主体内容区域 */}
        <Layout style={{ display: "flex", flexDirection: "column" }}>
          <Content
            style={{
              padding: "24px",
              overflowY: "auto",
              backgroundColor: isDarkMode ? "#1f1f1f" : "#f0f2f5",
              flexGrow: 1,
            }}
          >
            <MessageList
              messages={messages}
              isTyping={isTyping}
              typingText={typingText}
            />
          </Content>

          <div style={{ padding: "16px", backgroundColor: "white" }}>
            <MessageInput
              onSendMessage={handleSendMessage}
              disabled={isTyping}
            />
          </div>
        </Layout>
      </Layout>
    </Layout>
  );
};

export default ChatPage;

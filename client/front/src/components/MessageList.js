// src/components/MessageList.js

import React, { useEffect, useRef } from "react";
import { Avatar } from "antd";
import { UserOutlined, RobotOutlined } from "@ant-design/icons";
import "./MessageList.css";

const MessageList = (props) => {
  const {
    messages = [],
    isTyping = false,
    typingText = "",
  } = props || {};

  const endRef = useRef(null);

  const scrollToBottom = () => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages, isTyping, typingText]);

  return (
    <div className="chat-container">
      {messages.map((msg) => (
        <div
          key={msg.id}
          className={`chat-message ${msg.sender === "user" ? "user" : "ai"}`}
        >
          <Avatar
            className="chat-avatar"
            icon={msg.sender === "user" ? <UserOutlined /> : <RobotOutlined />}
            style={{
              backgroundColor:
                msg.sender === "user" ? "#1677ff" : "#10b981",
            }}
          />
          <div className="chat-bubble">{msg.text}</div>
        </div>
      ))}

      {isTyping && (
        <div className="chat-message ai">
          <Avatar
            className="chat-avatar"
            icon={<RobotOutlined />}
            style={{ backgroundColor: "#10b981" }}
          />
          <div className="chat-bubble typing-effect">{typingText}</div>
        </div>
      )}

      <div ref={endRef}></div>
    </div>
  );
};

export default MessageList;

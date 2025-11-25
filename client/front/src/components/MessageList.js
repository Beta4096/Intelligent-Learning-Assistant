// src/components/MessageList.js

import React, { useEffect, useRef } from 'react';
import { List, Spin } from 'antd'; // 引入 Spin
import './MessageList.css';

const MessageList = ({ messages, isTyping }) => { // 接收 isTyping prop
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages, isTyping]); // 当 isTyping 变化时也滚动

  return (
    <>
      <List
        dataSource={messages}
        rowKey="id"
        renderItem={item => (
          <List.Item className={item.sender === 'user' ? 'user-message' : 'ai-message'}>
            <div className="message-bubble">{item.text}</div>
          </List.Item>
        )}
      />
      {/* 如果 AI 正在输入，则显示一个加载提示 */}
      {isTyping && (
        <List.Item className="ai-message">
          <div className="message-bubble">
            <Spin size="small" />
          </div>
        </List.Item>
      )}
      <div ref={messagesEndRef} />
    </>
  );
};

export default MessageList;
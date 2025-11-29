// src/components/MessageInput.js

import React, { useState } from "react";
import { Input, Button } from "antd";
import { SendOutlined } from "@ant-design/icons";

const MessageInput = ({ onSendMessage, disabled }) => {
  const [inputValue, setInputValue] = useState("");

  const handleSend = () => {
    if (!inputValue.trim()) return;
    onSendMessage(inputValue);
    setInputValue("");
  };

  return (
    <div style={{ display: "flex" }}>
      <Input
        size="large"
        placeholder="在这里输入您的问题..."
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        onPressEnter={handleSend}
        disabled={disabled}
      />
      <Button
        type="primary"
        size="large"
        icon={<SendOutlined />}
        style={{ marginLeft: "8px" }}
        onClick={handleSend}
        disabled={disabled}
        loading={disabled}
      >
        发送
      </Button>
    </div>
  );
};

export default MessageInput;

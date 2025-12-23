
import React, { useState } from 'react';
import { Input, Button } from 'antd';
import { SendOutlined } from '@ant-design/icons';

const MessageInput = ({ onSendMessage, disabled }) => { // 接收 disabled prop
  const [inputValue, setInputValue] = useState('');

  const handleSend = () => {
    onSendMessage(inputValue);
    setInputValue('');
  };

  return (
    <div style={{ display: 'flex' }}>
      <Input
        size="large"
        placeholder="在这里输入您的问题..."
        value={inputValue}
        onChange={e => setInputValue(e.target.value)}
        onPressEnter={handleSend}
        disabled={disabled} // 在 AI 回复时禁用输入框
      />
      <Button
        type="primary"
        size="large"
        icon={<SendOutlined />}
        style={{ marginLeft: '8px' }}
        onClick={handleSend}
        disabled={disabled} // 在 AI 回复时禁用按钮
        loading={disabled} // 让按钮显示加载状态
      >
        发送
      </Button>
    </div>
  );
};

export default MessageInput;
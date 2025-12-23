import React, { useState, useEffect } from 'react';
import { Layout, message, Spin, Select, Typography, Button } from 'antd';
import Sidebar from '../components/Sidebar';
import MessageList from '../components/MessageList';
import MessageInput from '../components/MessageInput';
import { getHistoryMessages, uploadFile, getAIResponse } from '../services/api';

const { Header, Content, Sider } = Layout;


// AI模型
const modelOptions = [
  { value: 'gemini-2.5-pro-preview-06-05', label: 'gemini-2.5-pro-preview-06-05' },
  { value: 'gpt-5-2025-08-07', label: 'gpt-5-2025-08-07' },
  { value: 'o4-mini-2025-04-16', label: 'o4-mini-2025-04-16' },

  
 
];

const ChatPage = ({ onLogout }) => {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [isTyping, setIsTyping] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([
    { uid: '-1', name: 'React 基础教程.pdf', status: 'done' },
  ]);

  // 存储当前选择的模型 
  const [currentModel, setCurrentModel] = useState('gpt-5-2025-08-07'); // 默认模型

  useEffect(() => {
    setLoading(true);
    getHistoryMessages()
      .then(res => {
        const formattedMessages = res.map(msg => ({
          ...msg,
          sender: msg.role === 'user' ? 'user' : 'ai',
          text: msg.content,
        }));
        setMessages(formattedMessages);
      })
      .catch(() => message.error('历史消息加载失败!'))
      .finally(() => setLoading(false));
  }, []);

  const handleModelChange = (value) => {
    message.success(`模型已切换至: ${value}`);
    setCurrentModel(value);
  };

  const handleSendMessage = async (text) => {
    if (!text.trim() || isTyping) return;

    const userMessage = {
      id: Date.now(),
      sender: 'user',
      role: 'user',
      content: text,
      text: text,
    };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setIsTyping(true);

    const apiMessages = newMessages.map(({ role, content }) => ({ role, content }));
    
    const aiResponseText = await getAIResponse(apiMessages, currentModel);

    const aiMessage = {
      id: Date.now() + 1,
      sender: 'ai',
      role: 'assistant',
      content: aiResponseText,
      text: aiResponseText,
    };
    setMessages(prevMessages => [...prevMessages, aiMessage]);
    setIsTyping(false);
  };

  const handleFileUpload = async (file) => { /* */ };

  return (
    <Layout style={{ height: '100vh' }}>
      {/* */}
      <Header style={{ color: 'white', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h1>智能学习助手</h1>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <Typography.Text style={{ color: 'white', marginRight: '10px', fontSize: '14px' }}>当前模型:</Typography.Text>
            <Select
              defaultValue={currentModel}
              style={{ width: 150 }}
              onChange={handleModelChange}
              options={modelOptions}
            />
          </div>
          <Button type="primary" danger onClick={onLogout}>登出</Button>
        </div>
      </Header>
      <Layout>
        <Sider width={280} theme="light" style={{ padding: '16px' }}>
          <Sidebar uploadedFiles={uploadedFiles} onFileUpload={handleFileUpload} />
        </Sider>
        <Layout style={{ display: 'flex', flexDirection: 'column' }}>
          <Content style={{ padding: '24px', overflowY: 'auto', backgroundColor: '#f0f2f5', flexGrow: 1 }}>
            {loading ? (
              <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                <Spin size="large" />
              </div>
            ) : (
              <MessageList messages={messages} isTyping={isTyping} />
            )}
          </Content>
          <div style={{ padding: '16px', backgroundColor: 'white' }}>
            <MessageInput onSendMessage={handleSendMessage} disabled={isTyping} />
          </div>
        </Layout>
      </Layout>
    </Layout>
  );
};

export default ChatPage;
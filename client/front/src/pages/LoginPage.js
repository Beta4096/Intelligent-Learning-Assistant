import React from 'react';
import { Button, Card, Form, Input } from 'antd';
import { useNavigate, Link } from 'react-router-dom';

const LoginPage = ({ onLogin }) => {
  const navigate = useNavigate();

  const onFinish = (values) => {
    console.log('登录信息:', values);
    
    onLogin();
    
    navigate('/chat'); 
  };

  return (
    <div style={{ 
      display: 'flex', 
      justifyContent: 'center', 
      alignItems: 'center', 
      height: '100vh', 
      backgroundImage: `url('/assets/background.png')`,
      backgroundSize: 'cover',
      backgroundPosition: 'center' 
    }}>
      <Card title="欢迎登录" style={{ width: 400 }}>
        {}
        <Form
          name="login"
          onFinish={onFinish}
          autoComplete="off"
          layout="vertical"
        >
          <Form.Item
            label="用户名"
            name="username"
            rules={[{ required: true, message: '请输入用户名!' }]}
          >
            <Input placeholder="请输入你的用户名" />
          </Form.Item>

          <Form.Item
            label="密码"
            name="password"
            rules={[{ required: true, message: '请输入密码!' }]}
          >
            <Input.Password placeholder="请输入你的密码" />
          </Form.Item>
          
          <Form.Item>
            <Button type="primary" htmlType="submit" style={{ width: '100%' }}>
              登录
            </Button>
          </Form.Item>

          <Form.Item style={{ textAlign: 'center', marginBottom: 0 }}>
            <Link to="/register">还没有账户？立即注册</Link>
          </Form.Item>
        </Form>
      </Card>
    </div>
  );
};

export default LoginPage;
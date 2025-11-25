import React from 'react';
import { Button, Card, Form, Input, message } from 'antd';
import { Link, useNavigate } from 'react-router-dom';

const RegisterPage = () => {
  const navigate = useNavigate();

  const onFinish = (values) => {
    console.log('收到的注册信息:', values);
    message.success('恭喜你，注册成功！即将跳转到登录页...');
    setTimeout(() => {
      navigate('/login');
    }, 1500);
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
      
      {/*  */}
      <Card title="创建一个新账户" style={{ width: 400 }}>
        <Form
          name="register"
          onFinish={onFinish}
          autoComplete="off"
          layout="vertical"
        >
          {/* 用户名输入项 */}
          <Form.Item
            label="用户名"
            name="username"
            rules={[{ required: true, message: '用户名是必填项!' }]}
          >
            <Input placeholder="请输入你的用户名" />
          </Form.Item>

          {/* 密码输入项 */}
          <Form.Item
            label="密码"
            name="password"
            rules={[{ required: true, message: '密码是必填项!' }]}
            hasFeedback
          >
            <Input.Password placeholder="请输入你的密码" />
          </Form.Item>

          {/* 确认密码输入项 */}
          <Form.Item
            label="确认密码"
            name="confirm"
            dependencies={['password']}
            hasFeedback
            rules={[
              { required: true, message: '请再次输入你的密码!' },
              ({ getFieldValue }) => ({
                validator(_, value) {
                  if (!value || getFieldValue('password') === value) {
                    return Promise.resolve();
                  }
                  return Promise.reject(new Error('两次输入的密码不一致!'));
                },
              }),
            ]}
          >
            <Input.Password placeholder="请确认你的密码" />
          </Form.Item>

          {/* 注册按钮 */}
          <Form.Item>
            <Button type="primary" htmlType="submit" style={{ width: '100%' }}>
              立即注册
            </Button>
          </Form.Item>

          {/* 跳转到登录页的链接 */}
          <Form.Item style={{ textAlign: 'center', marginBottom: 0 }}>
            <Link to="/login">已经有账户了？返回登录</Link>
          </Form.Item>
        </Form>
      </Card>
    </div>
  );
};

export default RegisterPage;
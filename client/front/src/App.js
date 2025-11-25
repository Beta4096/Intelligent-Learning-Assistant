import React, { useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';

import HomePage from './pages/HomePage'; 
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import ChatPage from './pages/ChatPage';
import ProtectedRoute from './components/ProtectedRoute'; // 路由守卫

import 'antd/dist/reset.css'; 
import './App.css'; 

function App() {
  //false -> 未登录
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  const handleLogin = () => {
    setIsLoggedIn(true);
  };

  const handleLogout = () => {
    // 登出后，更新状态为 false
    setIsLoggedIn(false);
  };

  return (
    <BrowserRouter>
      {/* Routes 组件用于包裹所有的路由规则 */}
      <Routes>
        {/* 公共路由 (无论是否登录都可以访问) */}
        
        {/* 根路径，指向主页 */}
        <Route path="/" element={<HomePage />} />

        {/* 登录页 */}
        <Route path="/login" element={<LoginPage onLogin={handleLogin} />} />

        {/* 注册页 */}
        <Route path="/register" element={<RegisterPage />} /> 

        {/* --- 受保护的路由 (只有登录后才能访问) */}

        {/* 聊天页 */}
        <Route
          path="/chat" 
          element={
            <ProtectedRoute isLoggedIn={isLoggedIn}>
              {/* 只有当 isLoggedIn 为 true 时，ProtectedRoute 才会渲染下面的 ChatPage */}
              <ChatPage onLogout={handleLogout} />
            </ProtectedRoute>
          }
        />
        
        {/* 可以添加更多受保护的路由*/}
        {/* <Route path="/profile" element={<ProtectedRoute isLoggedIn={isLoggedIn}><ProfilePage /></ProtectedRoute>} /> */}

      </Routes>
    </BrowserRouter>
  );
}

export default App;
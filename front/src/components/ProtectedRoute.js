import React from 'react';
// 引入 Navigate 组件，它专门用于重定向
import { Navigate } from 'react-router-dom';

// isLoggedIn: 代表用户是否登录的布尔值
// children: 代表被这个组件包裹的子组件
const ProtectedRoute = ({ isLoggedIn, children }) => {
  
  
  // 如果用户没有登录 (isLoggedIn 为 false)
  if (!isLoggedIn) {
    // 就使用 Navigate 组件把他重定向到 /login 页面
    return <Navigate to="/login" replace />;
  }

  // 如果用户已经登录了，就正常显示被包裹的子组件
  return children;
};

export default ProtectedRoute;
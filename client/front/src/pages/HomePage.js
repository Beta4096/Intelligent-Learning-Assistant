import React from 'react';
import { Layout, Button, Typography, Row, Col, Card } from 'antd';
import { useNavigate } from 'react-router-dom';
import './HomePage.css'; // 1. 导入新建的 CSS 文件

const { Header, Content } = Layout;
const { Title, Paragraph } = Typography;

const HomePage = () => {
  const navigate = useNavigate();

  const handleLogin = () => navigate('/login');
  const handleRegister = () => navigate('/register');

  return (
    <Layout className="home-page-layout">
      <Header className="home-page-header">
        <div>
          <Button type="primary" onClick={handleLogin} style={{ marginRight: '10px' }}>
            登录
          </Button>
          <Button onClick={handleRegister}>
            注册
          </Button>
        </div>
      </Header>

      <Content className="home-page-content">
        <div className="title-section">
          <Title level={1} className="main-title">
            智能学习助手系统
          </Title>
          <Paragraph className="subtitle">
            从通用大模型到教材级私教，实现知识的精准匹配与个性化辅导。
          </Paragraph>
        </div>

        <Row gutter={24} className="cards-section">
          <Col span={8}>
            <Card title="行业痛点" bordered={false} className="info-card">
              <Paragraph>
                通用大模型在学科深度上存在不足，无法满足学生在作业和自学场景中的精准需求。
              </Paragraph>
            </Card>
          </Col>
          <Col span={8}>
            <Card title="项目愿景" bordered={false} className="info-card">
              <Paragraph>
                打造一款智能学习助手，将教材知识融入模型，实现逐步辅导而非直接给出答案。
              </Paragraph>
            </Card>
          </Col>
          <Col span={8}>
            <Card title="核心价值" bordered={false} className="info-card">
              <Paragraph>
                让模型拥有指定教材的知识边界与教学策略，实现知识的精准匹配与个性化辅导。
              </Paragraph>
            </Card>
          </Col>
        </Row>
      </Content>
    </Layout>
  );
};

export default HomePage;
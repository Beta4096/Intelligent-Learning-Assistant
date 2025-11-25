import React from 'react';
import { Upload, Button, List, Typography, Divider } from 'antd';
import { UploadOutlined, FilePdfOutlined, FileTextOutlined } from '@ant-design/icons';

const { Title } = Typography;

const Sidebar = ({ uploadedFiles, onFileUpload }) => {
  const props = {
    beforeUpload: file => {
      onFileUpload(file);
      return false;
    },
    showUploadList: false,
  };

  return (
    <div>
      <Title level={4}>我的教材库</Title>
      <Upload {...props}>
        <Button icon={<UploadOutlined />} style={{ width: '100%' }}>
          上传新教材
        </Button>
      </Upload>
      <Divider />
      <List
        header={<div>已上传文件</div>}
        dataSource={uploadedFiles}
        renderItem={item => (
          <List.Item>
            <List.Item.Meta
              avatar={item.name.endsWith('.pdf') ? <FilePdfOutlined /> : <FileTextOutlined />}
              // 将 <a> 标签修改为 <span>
              title={<span>{item.name}</span>}
              description={item.status === 'uploading' ? '上传中...' : item.status === 'error' ? '上传失败' : ''}
            />
          </List.Item>
        )}
      />
    </div>
  );
};

export default Sidebar;
import React from "react";
import { Upload, Button, List, Typography, Popconfirm } from "antd";
import {
  UploadOutlined,
  FilePdfOutlined,
  FileImageOutlined,
  DeleteOutlined,
} from "@ant-design/icons";

import "./Sidebar.css";

const { Text } = Typography;

const Sidebar = ({ uploadedFiles = [], onFileUpload, onDeleteFile, t }) => {
  const uploadProps = {
    beforeUpload: (file) => {
      onFileUpload(file);
      return false; // 阻止自动上传，让我们自己处理
    },
  };

  const getFileIcon = (name) => {
    const ext = name.split(".").pop().toLowerCase();
    if (["png", "jpg", "jpeg", "gif"].includes(ext))
      return <FileImageOutlined className="file-icon" />;
    if (["pdf"].includes(ext))
      return <FilePdfOutlined className="file-icon red" />;
    return <FilePdfOutlined className="file-icon" />;
  };

  return (
    <div className="sidebar-container">
      {/* 标题 */}
      <h2 className="sidebar-title">📚 {t.uploadTextbook}</h2>

      {/* 上传按钮 */}
      <Upload {...uploadProps} showUploadList={false}>
        <Button className="upload-btn" icon={<UploadOutlined />}>
          {t.uploadTextbook}
        </Button>
      </Upload>

      {/* 文件列表 */}
      <List
        className="file-list"
        dataSource={uploadedFiles}
        locale={{ emptyText: t.noFiles || "No files" }}  // 国际化"暂无上传文件"
        renderItem={(item) => (
          <List.Item
            className="file-item"
            actions={[
              <Popconfirm
                title={t.confirmDelete || "Confirm delete this file?"}
                onConfirm={() => onDeleteFile && onDeleteFile(item)}
                okText={t.delete || "Delete"}
                cancelText={t.cancel || "Cancel"}
              >
                <DeleteOutlined className="delete-btn" />
              </Popconfirm>,
            ]}
          >
            <List.Item.Meta
              avatar={getFileIcon(item.name)}
              title={<Text className="file-name">{item.name}</Text>}
            />
          </List.Item>
        )}
      />
    </div>
  );
};

export default Sidebar;

# AniGen ComfyUI 插件 - 快速开始指南

## 概述

我们已经成功将 AniGen 项目转换为 ComfyUI 插件！这个插件允许你直接在 ComfyUI 中使用 AniGen 从单张图像生成可动画的 3D 资产。

## 插件位置

插件位于：`/workspace/comfyui_plugin/`

## 插件内容

插件包含以下文件：

1. **`__init__.py`** - 插件初始化文件，导出节点映射
2. **`nodes.py`** - 包含两个 ComfyUI 节点：
   - `AniGenImageTo3D` - 主要的图像到 3D 生成节点
   - `AniGenCleanupTemp` - 清理临时文件的节点
3. **`install.py`** - 安装脚本，帮助将插件安装到 ComfyUI
4. **`README.md`** - 详细的使用文档
5. **`.gitignore`** - Git 忽略文件配置

## 安装步骤

### 1. 确保 AniGen 环境已正确配置

首先确保你已经按照 AniGen 项目根目录的 README.md 完成了环境配置。

### 2. 安装插件

#### 方法一：使用安装脚本

```bash
cd /workspace/comfyui_plugin
python install.py --comfyui-dir /path/to/your/ComfyUI --link
```

#### 方法二：手动安装

将 `comfyui_plugin` 目录复制或链接到 ComfyUI 的 `custom_nodes` 目录：

```bash
ln -s /workspace/comfyui_plugin /path/to/ComfyUI/custom_nodes/AniGen
```

### 3. 下载模型（如果尚未下载）

```bash
cd /workspace
./setup.sh
```

### 4. 重启 ComfyUI

重启 ComfyUI 以加载插件。

## 使用方法

1. 打开 ComfyUI
2. 在节点菜单中找到 "AniGen" 类别
3. 添加 "AniGen: Image to 3D" 节点
4. 连接图像输入
5. 调整参数（如果需要）
6. 运行工作流
7. 获取生成的 GLB 文件路径

## 节点功能

### AniGen: Image to 3D

- **输入**：图像和各种生成参数
- **输出**：
  - 网格文件路径（GLB 格式）
  - 骨架文件路径（GLB 格式）
  - 处理后的图像

### AniGen: Cleanup Temp Files

- 清理插件生成的临时文件

## 注意事项

- 需要 CUDA 支持
- 至少需要 18GB GPU 内存
- 首次运行会自动检查模型文件
- 生成的 GLB 文件可以直接导入 Blender、Unity、Unreal 等 3D 软件

## 故障排除

请参考 `/workspace/comfyui_plugin/README.md` 中的故障排除部分。

# AniGen ComfyUI Plugin

这是一个将 AniGen 集成到 ComfyUI 的插件，允许你直接在 ComfyUI 中使用 AniGen 从单张图像生成可动画的 3D 资产。

## 目录结构

```
comfyui_plugin/
├── __init__.py        # 插件初始化文件
├── nodes.py           # ComfyUI 节点实现
├── install.py         # 安装脚本
├── README.md          # 本文件
└── .gitignore         # Git 忽略文件
```

## 安装

### 方法一：使用安装脚本

1. 首先确保你已经按照 AniGen 的安装指南完成了环境配置（参见项目根目录的 README.md）。
2. 使用安装脚本：
   ```bash
   python install.py --comfyui-dir /path/to/ComfyUI [--link]
   ```
   - `--comfyui-dir`: 指向你的 ComfyUI 安装目录
   - `--link`: 可选，使用符号链接而不是复制文件
3. 确保模型文件正确放置在 AniGen 的 `ckpts/anigen/` 目录下（你可以使用 `setup.sh` 脚本下载模型）。
4. 重启 ComfyUI。

### 方法二：手动安装

1. 首先确保你已经按照 AniGen 的安装指南完成了环境配置（参见项目根目录的 README.md）。
2. 将此 `comfyui_plugin` 目录复制或链接到 ComfyUI 的 `custom_nodes` 目录：
   ```bash
   # 假设 ComfyUI 安装在 ~/ComfyUI
   ln -s /path/to/AniGen/comfyui_plugin ~/ComfyUI/custom_nodes/AniGen
   # 或者复制
   cp -r /path/to/AniGen/comfyui_plugin ~/ComfyUI/custom_nodes/AniGen
   ```
3. 确保模型文件正确放置在 AniGen 的 `ckpts/anigen/` 目录下（你可以使用 `setup.sh` 脚本下载模型）。
4. 重启 ComfyUI。

## 使用

在 ComfyUI 中，你可以找到以下节点，它们位于 "AniGen" 类别下：

### 1. AniGen: Image to 3D

主要的图像到 3D 生成节点。

#### 参数

- **image**: 输入图像
- **seed**: 随机种子
- **ss_model**: Sparse Structure 模型选择
  - `ss_flow_duet`: 推荐，详细骨架（完整微调几何）
  - `ss_flow_solo`: 精确几何（冻结几何）
  - `ss_flow_epic`: 几何和骨架平衡（LoRA 微调几何）
- **slat_model**: Structured Latent 模型选择
  - `slat_flow_auto`: 自动确定关节数量
  - `slat_flow_control`: 可控制关节密度
- **ss_guidance_strength**: SS 模型引导强度（0.0-15.0）
- **ss_sampling_steps**: SS 模型采样步数（1-100）
- **slat_guidance_strength**: SLAT 模型引导强度（0.0-10.0）
- **slat_sampling_steps**: SLAT 模型采样步数（1-100）
- **joints_density**: 关节密度（0-4，仅在 slat_flow_control 模型下有效）
- **texture_size**: 纹理分辨率（256-2048）
- **simplify_ratio**: 网格简化比例（0.0-1.0）
- **fill_holes**: 是否填充网格中的空洞
- **smooth_skin_weights**: 是否平滑蒙皮权重
- **filter_skin_weights**: 是否过滤蒙皮权重

#### 输出

- **mesh_path**: 生成的 3D 网格文件路径（GLB 格式）
- **skeleton_path**: 生成的骨架文件路径（GLB 格式）
- **processed_image**: 处理后的图像

### 2. AniGen: Cleanup Temp Files

清理插件生成的临时文件。

#### 参数

- **confirm**: 确认清理（必须设为 True 才能执行）

#### 输出

- **status**: 清理操作的状态信息

## 注意事项

1. 此插件需要 CUDA 支持和足够的 GPU 内存（至少 18GB）。
2. 首次运行会自动检查并确保模型文件正确下载。
3. 生成过程可能需要一些时间，具体取决于图像内容和硬件配置。
4. 生成的 GLB 文件可以直接导入到 Blender、Unity、Unreal Engine 等 3D 软件中使用。
5. 临时文件保存在插件目录下的 `temp` 文件夹中，可以使用清理节点定期清理。

## 故障排除

### 问题：找不到模块

确保插件目录正确放置在 ComfyUI 的 `custom_nodes` 目录下，并且 AniGen 的所有依赖都已正确安装。

### 问题：模型加载失败

确保模型文件已正确下载并放置在 AniGen 的 `ckpts/anigen/` 目录下。你可以运行 AniGen 根目录下的 `setup.sh` 脚本下载模型。

### 问题：GPU 内存不足

尝试减小采样步数或纹理大小，或者使用更小的模型。

## 许可证

与 AniGen 项目相同，遵循 MIT 许可证。

# AniGen ComfyUI Plugin

将 AniGen（Image-to-3D Animatable Asset Generation）拆分为独立 ComfyUI 节点，通过 ComfyUI 工作流串联实现完整的 3D 资产生成流程。

## 节点架构

原 `AnigenImageTo3DPipeline` 被拆分为以下 8 个独立节点，每个节点职责单一：

```
LoadImage (ComfyUI原生)
    │
    ▼
AniGenModelLoader ──────────────────────────────────────────────┐
    │                                                            │
    ▼                                                            │
AniGenPreprocessImage ◄── IMAGE (ComfyUI原生)                    │
    │                                                            │
    ├─ processed_image ──┐                                       │
    └─ normal_image ─────┤                                       │
                         ▼                                       │
              AniGenEncodeCondition ◄── models ──────────────────┤
                  │                                              │
                  ├─ cond_ss ──► AniGenSampleSS ◄── models ─────┤
                  │                    │                         │
                  │                    └─ ss_result ──┐          │
                  │                                   ▼          │
                  └─ cond_slat ──► AniGenSampleSLat ◄── models ─┤
                                         │                      │
                                         └─ slat_result ──┐     │
                                                            ▼    │
                                              AniGenDecodeSLat ◄──┘
                                                    │
                                                    └─ mesh_result ──► AniGenPostprocess
                                                                              │
                                                                              └─ post_result ──► AniGenExportGLB
```

## 节点说明

| 节点 | 功能 | 输入 | 输出 |
|------|------|------|------|
| **AniGenModelLoader** | 加载所有模型（DINOv2, DSINE, SS Flow, SS Decoder, SLat Flow, SLat Decoder） | ss_model, slat_model | models |
| **AniGenPreprocessImage** | 图像预处理：去背景 + 法线估计 | models, image | processed_image, normal_image |
| **AniGenEncodeCondition** | 用 DINOv2 编码图像条件 | models, processed_image, normal_image | cond_ss, cond_slat |
| **AniGenSampleSS** | 稀疏结构采样 + 解码 | models, cond_ss, seed, guidance, steps | ss_result |
| **AniGenSampleSLat** | 结构化隐变量采样 | models, cond_slat, ss_result, seed, guidance, steps, joints_density | slat_result |
| **AniGenDecodeSLat** | SLat 解码为网格和骨架 | models, slat_result | mesh_result |
| **AniGenPostprocess** | 网格简化、蒙皮权重处理、UV参数化、纹理烘焙 | mesh_result, 各种参数 | post_result |
| **AniGenExportGLB** | 导出 GLB 文件 | post_result, filename_prefix | mesh_path, skeleton_path |

## 自定义数据类型

| 类型名 | 说明 |
|--------|------|
| `ANIGEN_MODELS` | 所有已加载模型的字典 |
| `ANIGEN_COND_SS` | SS 模型条件（cond, neg_cond, normal） |
| `ANIGEN_COND_SLAT` | SLat 模型条件（cond, neg_cond, normal） |
| `ANIGEN_SS_RESULT` | SS 采样结果（coords, coords_skl） |
| `ANIGEN_SLAT_RESULT` | SLat 采样结果（slat, slat_skl SparseTensor） |
| `ANIGEN_MESH_RESULT` | 解码后的网格和骨架数据 |
| `ANIGEN_POST_RESULT` | 后处理完成的完整数据 |

## 安装

```bash
# 方法一：符号链接（推荐，方便开发）
ln -s /path/to/AniGen/comfyui_plugin /path/to/ComfyUI/custom_nodes/AniGen

# 方法二：复制
cp -r /path/to/AniGen/comfyui_plugin /path/to/ComfyUI/custom_nodes/AniGen
```

确保 AniGen 的 `ckpts/` 目录已下载模型权重，然后重启 ComfyUI。

## 注意事项

- 需要 NVIDIA GPU（≥18GB 显存）
- 使用 ComfyUI 原生 `LoadImage` 节点加载输入图像
- `AniGenExportGLB` 是输出节点（OUTPUT_NODE），使用 ComfyUI 的输出目录
- 各节点之间通过自定义类型传递数据，确保节点按顺序连接

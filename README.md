# Local Edit Detector — 本地部署 Demo

AI 修图检测与定位(冻结 DINOv2 + 29,699 参数混合监督头)。
输入一张图片,输出:**检测分数**(是否被 AI 局部编辑)+ **编辑区域热图**。

## 快速开始

```bash
# 1. 依赖 (Python 3.10+, PyTorch 2.x)
pip install fastapi uvicorn python-multipart pillow numpy torch matplotlib

# 2. 启动服务 (首次运行会自动下载 DINOv2 权重 ~90MB)
cd app
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000

# 3. 打开浏览器
#    http://127.0.0.1:8000   前端 Demo (拖拽图片)
#    接口文档 http://127.0.0.1:8000/docs
```

> ⚠️ 首次运行会自动下载 DINOv2 骨干权重 (~90MB, 来自 facebookresearch/dinov2)。
> 若下载慢或失败(国内网络),可先手动下载放到缓存目录:
> ```
> # Windows: C:\Users\<你>\.cache\torch\hub\checkpoints\dinov2_vits14_pretrain.pth
> # Linux:   ~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth
> # 下载地址: https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
> ```

## API

| 端点 | 方法 | 说明 |
|---|---|---|
| `/predict` | POST | multipart 上传 `file`,返回 `{score, heatmap_b64, mask_b64}` |
| `/health` | GET | 健康检查 |
| `/` | GET | 前端 Demo 页面 |

```python
import requests
r = requests.post("http://127.0.0.1:8000/predict",
                  files={"file": open("img.jpg", "rb")})
d = r.json()
print(d["score"])                    # 0~1,>0.5 视为被编辑
# d["heatmap_b64"] / d["mask_b64"]   # base64 PNG 热图,可直接 <img src="data:image/png;base64,...">
```

## 测试图片 (`examples/`)

- `fake_1~10.jpg`:AI 编辑过的图片(INP-X 修复式编辑,检测分数接近 1.0,热图清晰)
- `real_1~10.jpg`:对应的原始图片(未编辑,检测分数 ~0.001-0.003)
- `hard/fake_1~3.jpg`、`hard/real_1~3.jpg`:SDXL-inpaint 现代修图器编辑的"困难样本"——现代修图痕迹弱,检测分数可能偏低,属正常现象(论文 §4.4)

## 说明

- 模型:冻结 DINOv2 ViT-S/14(自动下载)+ 高通残差 + 29,699 参数 MLP 头(`app/weak_sup_v4_head.pt`,source-disjoint 严格协议下训练)
- 输入自动 resize 到 518×518,输出 37×37 热图
- 学术工作,详见主仓库 wuzhang114/inpx-edit-detection

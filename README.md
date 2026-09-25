# 基于特征工程的动漫人脸检测与关键点回归

本仓库用于完成数字图像处理实验一：使用 11 通道手工特征、Depth-2 AdaBoost Cascade 和多级形状回归，实现动漫人脸检测与 28 点关键点定位。

当前进度和下一步任务见 [TODO.md](TODO.md)。

## 最终 Demo

使用已导出的 `models/` 配置与权重，对本地图像执行检测和 28 点回归：

```powershell
.venv\Scripts\python.exe demo.py --image test.jpg --model-dir models --output results/test_vis.jpg
```

同时生成 `results/test_vis.json`，内容为按末级 Cascade 分数降序排列的 `bbox`（原图像素、右下角半开）、`score` 和固定顺序的 28 个 `[x,y]` 点。Python 调用方式：

```python
import cv2
from src.detector import AnimeFaceDetector

detector = AnimeFaceDetector("models")
results = detector.detect(cv2.imread("test.jpg"))
```

导出清单见 `models/config.json`，通道定义见 `models/feature_definition.json`，关键点编号见 `models/landmark28_schema.json`。当前 Demo 在 Cascade 后使用 train 页面训练的 HOG 候选复核器，按固定阈值筛选，再做 28 点回归。八个未用于调参的 test 漫画作品上，IoU≥0.5 的 Precision 为 0.097、Recall 为 0.188、F1 为 0.128；仍有漏检和误报，不能视为可靠的整页人脸识别。效果、对照和限制见 `实验记录/2026-09-25_步骤八最终Demo与模型导出.md`。

已完成步骤的规范核对、验收证据、报告可用图表及局限见 [审计与报告素材索引](实验记录/2026-09-24_已完成步骤审计与报告素材索引.md)。原创统计图位于 `results/report_evidence/`，可用 `python scripts/plot_report_evidence.py` 从已保存的 JSON 重建。

## 实验目标

- 从灰度图计算 11 个 `uint8` 手工特征通道；
- 使用像素差构建 Depth-2 弱分类器，并训练至少 3 个 Stage 的 Cascade；
- 在真实动漫/漫画场景中采样负样本并进行困难负样本挖掘；
- 完成多尺度滑窗搜索、候选框合并和 28 点形状回归；
- 报告 Precision、Recall、F1、IoU、NME 和运行效率；
- 提供统一的 `AnimeFaceDetector.detect()` 接口和命令行 Demo。

完整要求见根目录中的实验说明 Markdown 和 PDF。PDF 的页面渲染与文本提取属于临时材料，不纳入版本控制。

## 环境

推荐使用 Python 3.12 和 [uv](https://docs.astral.sh/uv/)（关键点预标注依赖要求 Python 3.12+）。在 PowerShell 中一键创建环境、安装依赖并执行验收：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_environment.ps1
```

脚本会安装 Python 3.12（若本机尚无）、创建或修正 `.venv`、安装 `requirements.txt`，最后运行无网络、无 GUI 的依赖冒烟测试。成功后可按需激活环境：

```powershell
.venv\Scripts\Activate.ps1
python scripts/verify_environment.py
```

## 当前目录

```text
experiment1/
├── datasets/
│   ├── downloads/       # 原始压缩包，不提交
│   ├── raw/             # 解压后的原始数据，不提交
│   ├── manifests/       # 固定划分、数据清单和统一标注
│   ├── annotations/
│   │   ├── auto/        # 自动预标注，不提交
│   │   └── corrected/   # 人工修正的 28 点标注，可提交
│   ├── derived/         # 可再生成的裁剪与负样本，不提交
│   └── tools/           # 数据整理脚本
├── 实验记录/            # 阶段性实验记录
├── requirements.txt
└── README.md
```

后续实现按实验说明逐步加入 `src/`、`scripts/`、`configs/`、`models/`、`results/` 和 `reports/`。最终模型、配置、演示结果与报告是实验交付物，不应整体加入 `.gitignore`；训练过程中的 checkpoint、运行日志和服务缓存会被忽略。

## 数据准备

当前数据入口包含 LAP、Manga109 和 anime256。原始数据总量较大且 Manga109 有再分发限制，因此只保留在本地。数据用途、许可注意事项、统一标注格式和固定划分方法见 [`datasets/README.md`](datasets/README.md)。

重新生成数据清单：

```powershell
python datasets/tools/build_manifests.py
```

对 anime256 裁剪池做一小批 28 点关键点预标注并测量本机吞吐量：

```powershell
python scripts/preannotate_landmarks.py --limit 64 --split all
```

每次运行会在 `datasets/annotations/auto/<run-name>/` 下生成不可变的 JSONL 标注和包含模型校验值、环境、耗时及全量外推的元数据。anime256 已是一脸一图的裁剪池，因此脚本跳过人脸检测，直接运行 HRNetV2 关键点模型；自动标注的 `visibility` 保持为空，等待人工确认。

当前人工复核入口是 320 张固定预标注（train 240、validation 32、test 48），清单和校验值在 `datasets/annotations/review_sets/landmark320/`。源图保留在本地忽略目录 `datasets/raw/anime256/`。启动本地复核工具：

```powershell
.venv\Scripts\python.exe scripts\review_landmarks.py
```

操作和交接规则见 [`datasets/annotations/LANDMARK_REVIEW_GUIDE.md`](datasets/annotations/LANDMARK_REVIEW_GUIDE.md)。

320 张现已完成 V/H/U 人工粗复核并接受辅助标注，但未人工精调坐标。正式记录在 `datasets/annotations/corrected/landmark28_review320.jsonl`，来源与限制见 `datasets/annotations/corrected/landmark28_review320_acceptance.json` 及阶段记录；不要将其称为精确人工关键点真值。

固定划分种子为 `experiment1-anime-face-v1`，按来源组进行 Train 75% / Validation 10% / Test 15% 划分，避免同源图片泄漏到不同集合。

## 11 通道特征复现与观察

核心实现位于 `src/channels11.py`，使用 `int16` 中间量和显式 NumPy 切片复现原图、两级平滑及八个方向差分通道。运行公式、类型、形状和边界验收：

```powershell
python -m unittest discover -s tests -v
```

生成步骤二要求的 6 组“原图 + 11 通道”图（两张真实人脸会在首次运行时从 `face_recognition` 示例仓库下载；其余样本读取本地 anime256 和 Manga109）：

```powershell
python scripts/visualize_channels.py
```

图片与带 SHA-256 的样本清单输出到 `results/step2_channels/`。完整公式、样本选择、观察结果及局限见 `实验记录/2026-09-23_11通道特征复现与观察.md`。

## 步骤三：检测训练样本

使用固定来源划分，从 Manga109 人脸框和漫画页面生成 24×24 正负样本、训练集水平翻转增强、元数据和人工复核总览图：

```powershell
.venv\Scripts\python.exe scripts/prepare_detection_dataset.py
.venv\Scripts\python.exe scripts/cache_detection_channels.py
```

输出在 `datasets/derived/step3_detection_v1/`。人工挑图的准确目录、ID 回填格式和复核命令见 [`datasets/derived/STEP3_REVIEW.md`](datasets/derived/STEP3_REVIEW.md)。

已将人工提供的错误样本清单 `datasets/annotations/corrected/revise_reason.csv` 转为完整复核表，并生成 `usable_samples.jsonl` 与 `usable_channels11_index.csv`。后续训练使用可用样本清单和对应通道索引。

## 步骤四：像素差特征与弱树

`src/weak_tree.py` 提供 `PixelDifferenceFeature`、`TreeNode` 和 `Depth2WeakTree`。输入为单个 `(11, 24, 24)` 或一批 `(N, 11, 24, 24)` 的 `uint8` 通道图。像素差按 `C[c, y1, x1] - C[c, y2, x2]` 计算为有符号 `int16`；节点在差值小于等于阈值时走左分支。四个叶子分数依次对应根左/子左、根左/子右、根右/子左、根右/子右。

`src/weak_tree_training.py` 实现单棵弱树的随机候选采样和加权贪心搜索；`src/cascade.py` 实现加权弱树组合、3 个 Stage 的早拒绝、召回阈值校准、逐级统计及 JSON 模型读写。训练只读取人工确认保留的样本和 `usable_channels11_index.csv` 对应的通道缓存行。

复现初版模型、扫描训练漫画页、使用已复核的困难负样本回训：

```powershell
.venv\Scripts\python.exe scripts/train_cascade.py --trees-per-stage 18 --root-candidates 32 --child-candidates 24
.venv\Scripts\python.exe scripts/mine_hard_negatives.py --output datasets/derived/step4_hard_negatives_v2
.venv\Scripts\python.exe scripts/train_cascade.py --trees-per-stage 18 --root-candidates 32 --child-candidates 24 --hard-negative-dir datasets/derived/step4_hard_negatives_v2 --hard-review-csv datasets/annotations/corrected/step4_hard_negative_review.csv --model models/step4_cascade.json --report results/step4_stage_stats.json
```

初版与回训模型保存在 `models/`，各 split 的 Stage 通过统计保存在 `results/`。扫描候选与编号总览图保存在本地 `datasets/derived/step4_hard_negatives_v2/`；其余未复核候选不会加入训练。方法和局限见 [弱树候选搜索记录](实验记录/2026-09-23_Depth-2弱树与候选搜索.md)及 [Cascade 训练记录](实验记录/2026-09-23_Cascade训练与困难负样本.md)。

## 步骤五：多尺度搜索与候选框合并

`src/multiscale.py` 在原图上构建逐层缩小的金字塔，对每层运行 24×24 Cascade 滑窗，将半开区间候选框映射回输入图坐标，再按末级分数做 IoU NMS。默认参数保存在 `models/step5_search_config.json`：尺度因子 1.2、步长 1、NMS IoU 0.3。`detect_multiscale(gray, cascade)` 返回原图坐标框、分数和逐层尺寸/窗口数/候选数/耗时。

复现 1.1/1.2/1.3 与 step 1/2 对照：

```powershell
.venv\Scripts\python.exe scripts/compare_multiscale.py
```

该脚本从两个不同作品的 validation 页面各选一张 320×320 人脸周边裁剪，完成 3 种尺度因子与 2 种步长的对照，逐层数据写入 `results/step5_comparison.json`。随后对第一张原生分辨率整页用默认尺度因子比较 step 1/2，原页坐标框和逐层数据写入 `results/step5_full_page.json`。预览图位于本地忽略目录 `datasets/derived/step5_comparison/`，红框为最高分的 20 个 NMS 检测，绿框为裁剪的锚定标注。方法、数值和当前检测误报情况见 [步骤五实验记录](实验记录/2026-09-23_多尺度搜索与候选框合并.md)。

## 步骤七：固定页面评价

运行 `.venv\Scripts\python.exe scripts/evaluate_detection.py` 可复现两张 validation 整页的 IoU ≥ 0.5 一对一匹配指标，并记录 step 1/2 的窗口数和含 NMS 耗时；运行 `.venv\Scripts\python.exe scripts/compare_integer_float.py` 比较固定 100 个窗口的整数与等价舍入浮点通道及 Cascade 输出。结果见 `results/step7_detection_evaluation.json`、`results/step7_integer_float.json` 和 `实验记录/2026-09-25_步骤七评价指标.md`。当前模型整页误检较多，指标仅对应固定已接触的 validation 页面。

## 计划中的工程结构

```text
src/                       # 11 通道、金字塔、Cascade、回归与指标
scripts/                   # 数据准备、训练、挖掘、评估脚本
configs/                   # 可复现实验配置和固定随机种子
models/                    # 最终模型及导出配置
results/                   # 11 通道图、检测/关键点 Demo 与指标
reports/                   # 项目报告
demo.py                    # 最终命令行入口
```

## 最终交付检查

- Python 源码、依赖文件和运行说明；
- 人脸 28 点关键点数据集（图片与人工确认标注）；
- 模型文件与完整推理配置；
- 原图加 11 通道可视化、困难负样本和最终检测效果图；
- 检测、关键点与效率指标；
- 项目报告和可运行 Demo。

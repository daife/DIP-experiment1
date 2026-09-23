# 步骤三：检测样本人工复核

运行 `python scripts/prepare_detection_dataset.py` 后，检查 `datasets/derived/step3_detection_v1/review_sheets/`。文件名含 `train`、`validation` 或 `test`，以及 `positive` 或 `negative`；每张总览图最多显示 100 个裁剪，每格下方就是 `sample_id`。实际 24×24 图在同级目录的 `train/positive/` 等目录。

## 要挑什么

1. **正样本**：逐张检查是否有完整动漫/漫画脸，五官是否足够清楚，裁剪是否保留脸部。剔除错误框、只有局部五官、极糊或无法辨认的图。保留不同脸型、眼型、发型和画风，特别注意轻微旋转、头发遮挡。可用 `rotation`、`occlusion` 标签标出；无需凭 24×24 小图猜测脸型类别。
2. **负样本**：逐张检查是否混入完整人脸，有就剔除。优先保留头发、衣服、手、文字框、拟声词、建筑、普通背景或局部似脸的区域。可用 `hair`、`clothing`、`hand`、`text_box`、`sound_effect`、`building`、`background`、`face_like` 标签。纯白/纯黑、极糊、无可识别结构的区域应剔除。
3. 对存疑样本，在 `samples.jsonl` 中按 `sample_id` 找 `parent_image_path` 和 `crop_bbox_xyxy`；原图路径以 `datasets/` 为根，例如 `datasets/raw/manga109/images/...`。不要移动或修改原图。

## 把哪些 ID 交给脚本

编辑 **`datasets/annotations/corrected/step3_crop_review.csv`**，每行一个基本样本 ID。列为 `sample_id,decision,tags,note`，`decision` 填 `keep` 或 `reject`；多个标签以英文分号隔开。例如：

```csv
sample_id,decision,tags,note
trp00042,keep,rotation;occlusion,额发遮住左眼
trn00107,reject,face,漏标完整人脸
ten00012,keep,text_box,
```

生成的 `datasets/derived/step3_detection_v1/review_template.csv` 已列出全部基本样本 ID。可以把它复制到上述 `corrected` 路径，在表格软件里批量填 `keep`，再将问题样本改为 `reject` 并补标签。留空的样本会被标记为 `unreviewed`，不会误算成人工通过。训练增强图的 ID 末尾带 `f`，只需复核基本样本；增强图自动继承决定。

完成或阶段性录入后运行：

```powershell
.venv\Scripts\python.exe scripts/apply_detection_review.py
```

脚本生成 `datasets/derived/step3_detection_v1/usable_samples.jsonl` 和 `review_summary.json`，剔除所有 `reject` 样本及其增强图，并保留未复核状态供后续训练选择。`fully_reviewed` 为 `true` 才表示所有基本样本都经过人工复核。请把填好的 CSV 留在上述 `corrected` 路径，我就能接续清洗和检查类别覆盖。

## 数据规则

- 每类基本样本默认 Train 1800、Validation 240、Test 360；训练集每张另有水平翻转增强图。固定种子为 `experiment1-step3-v1`。
- 正样本来自 Manga109 的已标人脸框，过滤最短边小于 24 原图像素的框，再加约 15% 正方形边界并缩放。
- 负样本来自同一数据源的真实漫画页面，避开所有已标注人脸及其周边，并过滤空白或几乎纯黑区域。漏标人脸仍需人工剔除。
- `samples.jsonl` 保存父图 ID、原图路径、原始框、裁剪框、来源、split、增强方式和复核状态；`split_assignment.csv` 固定记录父图划分。同一作品的所有页面和裁剪继承同一 split。
- `channels11.npy` 按 `channels11_index.csv` 的行号存储每张 24×24 图的 11 个 `uint8` 通道，形状为 `(样本数, 11, 24, 24)`；运行 `python scripts/cache_detection_channels.py` 可重算。
- 总览图和裁剪由脚本再生成，不提交仓库。人工复核 CSV 可以提交。Manga109 图像使用遵守原数据集许可。

# Repository Instructions

## Source of truth and scope

- The original project specification, `项目一 基于特征工程的动漫人脸检测与关键点回归 (1).pdf`, and its companion `项目一_基于特征工程的动漫人脸检测与关键点回归_详细说明 (1).md` are the authoritative requirements and acceptance criteria. Consult the relevant sections before planning, implementing, or declaring a task complete. The PDF's extracted text and page images under `tmp/pdfs/requirements/` may help with inspection; they are temporary derivatives, not a replacement for the original PDF.
- `TODO.md` is a working step-by-step plan based on the specification and the repository's current state. It may contain inaccuracies, omissions, or unnecessary scope. Check disputed items against the authoritative requirements, correct the plan when warranted, and avoid treating a TODO item alone as a mandatory requirement.
- Keep the work focused on the experiment's required deliverables. Treat optional investigations and enhancements as optional unless the user explicitly requests them. If the PDF and Markdown appear to conflict, inspect the original PDF and explain any unresolved ambiguity to the user before making a decision that depends on it.
- Preserve traceability: record substantive implementation and experiment decisions, validation results, and any justified changes to the plan in the appropriate repository documentation. Do not mark a task complete without evidence that its applicable acceptance criteria are met.

## Execution and collaboration

- Work autonomously by default. Inspect the repository, make reasonable implementation decisions, run appropriate checks, and finish authorized work without asking for routine confirmation at each step. Give concise progress updates when work is prolonged or a meaningful finding changes the approach.
- Inspect a small or manageable set of images directly with available tools such as `view_image`. Make the visual assessment yourself and document the basis for consequential judgments; do not automatically delegate ordinary image review to the user.
- If a large image set, a specialized subjective judgment, or another task genuinely requires human review, pause the affected work and tell the user exactly what to do. Identify the files or review sheet, give simple numbered actions and explicit accept/reject or correction criteria, state where to save the results, and explain how work will resume. Prepare and narrow the review material first so the user has as little manual work as possible. Do not claim that unreviewed material was reviewed.
- If a necessary fact, preference, or decision cannot be inferred reliably from the specification, repository, or existing conversation, ask the user a focused question and pause the dependent work. Continue independent work when useful. State the available options and a sensible default when that helps the user answer quickly; do not silently invent a requirement.
- Keep raw or licensed datasets and disposable generated files in their existing local/ignored locations. Preserve fixed data splits, provenance, random seeds, and human-corrected annotations needed for reproducibility. Do not overwrite source data or recorded manual judgments without a clear reason.

## 实验工作记录与报告素材

- 每完成一个可验收的步骤，同步在 `实验记录/` 写阶段记录；记录对应的 PDF 步骤、实际命令和参数、输入数据及 split、随机种子、模型/清单版本或 SHA-256、输出路径、关键数值、测试结果、人工观察、限制和下一步。历史记录中的测试数量应标明当时的运行时间，后续新增测试不必改写历史。
- 将报告可用的图、表及其原始数值文件登记在阶段记录或素材索引中。每张图写明来源、图中颜色/符号的含义、使用的样本或 split、生成命令、适用结论与限制；复核图片本身可打开且内容与说明相符。由统计 JSON 生成的图应保存可重跑脚本。不要把裁剪分类结果写成整页 IoU 检测指标，也不要把 validation/test 上探索过的结果称为盲测。
- 保留有价值的过程证据，包括样本筛选前后计数、人工复核 CSV、困难样本编号及原因、各 Stage 通过数、逐层搜索数据、典型成功和失败案例。人工判断只按实际复核范围描述，未检查的图像保持未复核状态。
- 大型或授权受限图片继续留在本地忽略目录，但在记录中提供稳定的相对路径、来源 ID、生成方式和复核状态；提交报告前确认当地文件仍在，并遵守数据源展示/署名要求。可再分发的原创统计图和必要的小型报告素材放入 `results/` 以便版本管理。不要为凑报告图片复制原始授权数据到可提交目录。
- 每次修改模型、数据划分或推理配置后，核对旧图表是否仍对应当前版本；若不对应，重跑并更新记录，或清楚标为历史对照。步骤完成的勾选以规格验收和可追溯证据为依据，报告素材不足则记录明确的补充任务。

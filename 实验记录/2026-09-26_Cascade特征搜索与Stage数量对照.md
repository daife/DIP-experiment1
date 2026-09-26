# 2026-09-26 Cascade特征搜索与Stage数量对照

对应规格步骤四随机特征搜索/AdaBoost/Stage、步骤五候选合并、步骤七整页指标及用户要求的最终效果改进。未部署。

```powershell
.venv/Scripts/python.exe scripts/train_cascade.py --trees-per-stage 18 --root-candidates 128 --child-candidates 48 --hard-negative-dir datasets/derived/step4_hard_negatives_v2 --hard-review-csv datasets/annotations/corrected/step4_hard_negative_review.csv --model datasets/derived/step8_cascade_search128/model.json --report results/step8_cascade_search128_stage_stats.json
.venv/Scripts/python.exe scripts/train_cascade.py --stages 6 --trees-per-stage 18 --root-candidates 128 --child-candidates 48 --hard-negative-dir datasets/derived/step4_hard_negatives_v2 --hard-review-csv datasets/annotations/corrected/step4_hard_negative_review.csv --model datasets/derived/step8_cascade_search128_stage6/model.json --report results/step8_cascade_search128_stage6_stats.json
.venv/Scripts/python.exe scripts/evaluate_candidate_verifier.py --cascade datasets/derived/step8_cascade_search128/model.json --verifier models/candidate_hog_rbf_step3_v1.joblib --output results/step8_cascade_search128_validation.json --cache-dir datasets/derived/step8_cascade_search128/validation_cache
.venv/Scripts/python.exe scripts/evaluate_candidate_verifier.py --cascade datasets/derived/step8_cascade_search128_stage6/model.json --verifier models/candidate_hog_rbf_step3_v1.joblib --output results/step8_cascade_search128_stage6_validation.json --cache-dir datasets/derived/step8_cascade_search128_stage6/validation_cache
.venv/Scripts/python.exe scripts/audit_cascade_training_counts.py --model models/step4_cascade.json --model datasets/derived/step8_cascade_search128/model.json --model datasets/derived/step8_cascade_search128_stage6/model.json --output results/step8_cascade_search_training_audit.json
.venv/Scripts/python.exe -m py_compile scripts/audit_cascade_training_counts.py
git diff --check
```

固定train已复核裁剪6996加24个旧已复核困难负例，共7020（3442正/3578负）。新73个负例没有加入，避免混杂。seed20260923、每Stage18棵树、目标validation正样本通过率0.99；旧根/子候选32/24，新128/48。第二对照仅Stage数量3→6。已实际断言三个前缀Stage的参数、树、阈值完全相同。随机候选数量变化也改变随机池定义，不保证旧候选是新候选的子集。

训练脚本原报告的train只统计原裁剪、不包含24个追加负例；本次独立审计重建实际7020训练输入，报告为 `results/step8_cascade_search_training_audit.json`。逐Stage通过数如下（正/负）：

| 模型 | 各Stage累计通过正样本 | 各Stage累计通过负样本 |
|---|---|---|
| models\step4_cascade.json | 3408,3255,3213 | 1789,891,676 |
| datasets\derived\step8_cascade_search128\model.json | 3421,3370,3335 | 1640,1127,934 |
| datasets\derived\step8_cascade_search128_stage6\model.json | 3421,3370,3335,3255,3241,3226 | 1640,1127,934,682,609,540 |

validation裁剪：旧三阶段保留220/226正、31/236负；扩大搜索三阶段220正/47负；六阶段214正/30负。阈值由这批validation正脸校准，不能把该裁剪召回作为独立泛化证明。训练脚本沿用既有行为自动输出test裁剪统计，test已被历史探索；此次选择和判断以validation整页为依据，不称盲测，不用test裁剪数优化参数。

固定八个已探索validation作品73脸，金字塔1.2/步长2/NMS0.3/HOG旧RBF/边长36/IoU≥0.5；未修框或追加NMS。

| 模型/HOG阈值 | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| 旧三阶段/1.5 | 16 | 102 | 57 | 0.167539 |
| 搜索128三阶段/rbf1.5_side36 | 21 | 103 | 52 | 0.213198 |
| 搜索128三阶段/rbf1.75_side36 | 21 | 60 | 52 | 0.272727 |
| 搜索128六阶段/rbf1.5_side36 | 13 | 89 | 60 | 0.148571 |
| 搜索128六阶段/rbf1.75_side36 | 9 | 43 | 64 | 0.144000 |

完整逐页/阈值结果见上述JSON；两个新模型及候选缓存留在本地忽略目录。三阶段扩大搜索有收益，六阶段在较好背景拒绝数下反而降低整页表现。多Stage改变了最后Stage分数，当前NMS按该分数排序，所以两种模型最后NMS选择不同；不能将全部变化解释为简单拒绝了人脸。下一步应在相同通过候选集合上比较最后Stage、累计分数等排序，分离背景拒绝和NMS选框因素。

耗时含缓存建立/所有阈值评估；三阶段评价与六阶段训练曾并行，机器负载不同，不能据这些值宣称精确速度比。本次未新增视觉复核，数字提升不代表图像全部验收。语法/diff检查通过，前缀参数一致性检查通过。

## 版本

- models\step4_cascade.json model_sha256: `595983a793aa40ba1dfcaec704f24b15ac65ba7069887cdc9bc7d69630554194`
- models\step4_cascade.json train_manifest_sha256: `318fc00bbbc272adc279cbc42da1141b44734da67d856c05455065b19792ef8d`
- models\step4_cascade.json hard_cache_sha256: `5f6dc126f8db78f26c06d632f8e9a6bb592fe0fcf95a872f99cbb141e31d164e`
- datasets\derived\step8_cascade_search128\model.json model_sha256: `dccac3e24b030e5c3ebef4bf8fbe6cf65d06292d5d7b2744e3484fc941931ca8`
- datasets\derived\step8_cascade_search128\model.json train_manifest_sha256: `318fc00bbbc272adc279cbc42da1141b44734da67d856c05455065b19792ef8d`
- datasets\derived\step8_cascade_search128\model.json hard_cache_sha256: `5f6dc126f8db78f26c06d632f8e9a6bb592fe0fcf95a872f99cbb141e31d164e`
- datasets\derived\step8_cascade_search128_stage6\model.json model_sha256: `0b29510560a8871dd03df9a753eb873effa59c1724087ec1195410dbba596708`
- datasets\derived\step8_cascade_search128_stage6\model.json train_manifest_sha256: `318fc00bbbc272adc279cbc42da1141b44734da67d856c05455065b19792ef8d`
- datasets\derived\step8_cascade_search128_stage6\model.json hard_cache_sha256: `5f6dc126f8db78f26c06d632f8e9a6bb592fe0fcf95a872f99cbb141e31d164e`
- 三阶段页面清单SHA `f35b78e7a1717a88b4d5f5bdd39c8e50c095880e5113e6e9b90be72f3e23fce3`；HOG SHA `83ca4e51eafbc0279ec6d922d65b2ed16ba62bdf612c45c2b75bf523de29ec39`
- 六阶段页面清单SHA `f35b78e7a1717a88b4d5f5bdd39c8e50c095880e5113e6e9b90be72f3e23fce3`；HOG SHA `83ca4e51eafbc0279ec6d922d65b2ed16ba62bdf612c45c2b75bf523de29ec39`

最终效果仍未优秀，模型未锁定、未进行锁定整页test评价、未更新Demo；目标继续。

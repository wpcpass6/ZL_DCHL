# ZL_DCHL

`ZL_DCHL` 是一个面向下一兴趣点推荐任务的实验项目，核心方法是基于高阶异构超图与节点掩码自监督学习的 H-DCHL-B 模型。项目围绕 TSMC2014 数据集展开，包含完整的数据预处理、异构图构建、模型训练评估以及专利示例导出流程。

## 项目目标

- 基于用户历史签到序列预测下一个访问的 POI
- 联合建模用户协同关系、区域语义关系、类别语义关系和兴趣点转移关系
- 在区域节点和类别节点上引入随机掩码机制，通过解码重构进行自监督学习
- 输出常规排序指标，并支持导出专利展示所需的 Top-1 推荐结果样本表

## 目录说明

- `preprocess.py`：将原始 TSMC2014 数据预处理为训练可直接读取的 `pkl` 文件
- `dataset.py`：读取样本与图结构相关文件，构建训练/测试数据集和四类图分支
- `model.py`：H-DCHL-B 模型定义，包含异构超图卷积、转移分支和节点掩码重构损失
- `train.py`：训练、测试、保存模型、导出专利结果表
- `metrics.py`：`Rec@K` 与 `NDCG@K` 指标计算
- `utils.py`：预处理、构图、稀疏矩阵转换、GeoHash 编码等通用函数
- `zl_seq_sam.py`：按历史序列长度分层抽样专利示例
- `zl_global_seq_sam.py`：直接从结果总表中全局随机抽样专利示例
- `view.py`：简单查看 `pkl` 内容的辅助脚本

## 数据流程

### 1. 预处理

原始输入默认为：

- `datasets/dataset_TSMC2014_TKY.txt`

预处理主要步骤：

- 按用户聚合签到记录并按时间排序
- 过滤低频 POI
- 按时间间隔切分 session
- 按用户时间顺序划分训练 session 和测试 session
- 训练集采用滑窗方式构造 `prefix -> next POI`
- 测试集采用留一方式构造最后一步预测样本
- 建立用户、POI、类别、区域的整数索引映射

预处理后会生成 `datasets/TKY` 下的一组 `pkl` 文件，例如：

- `train_samples.pkl`
- `test_samples.pkl`
- `train_user_sessions.pkl`
- `test_user_sessions.pkl`
- `poi_coos.pkl`
- `poi_category.pkl`
- `poi_region.pkl`
- `idx2user.pkl`
- `idx2poi.pkl`
- `meta.pkl`

### 2. 数据集与构图

`dataset.py` 在加载样本的同时，会基于训练 session 构建四类结构：

- 协同分支：`POI-User`
- 区域分支：`POI-Region`
- 类别分支：`POI-Category`
- 转移分支：session 内前序到后序的有向 `POI-POI` 转移图

## 模型概述

模型由四个分支组成：

- 协同分支：建模用户长期协同行为
- 区域分支：建模 POI 的空间语义关系
- 类别分支：建模 POI 的类别语义关系
- 转移分支：建模 session 内的有向转移关系

其中：

- 协同、区域、类别分支采用“兴趣点到异构节点聚合、异构节点特征融合、异构节点回传到兴趣点”的异构超图卷积传播方式
- 转移分支采用基于有向转移图的传播方式
- 区域节点和类别节点支持随机掩码，并通过解码器重构形成辅助损失

最终模型输出：
- 所有候选 POI 的预测分数
- 训练时的辅助掩码重构损失

## 训练与评估

当前 `train.py` 默认参数包括：

- `seed=2026`
- `num_epochs=30`
- `batch_size=1024`
- `emb_dim=128`
- `lr=1e-3`
- `decay=5e-4`
- `dropout=0.3`
- `region_precision=5`

评估指标：

- `Rec@1/5/10/20`
- `NDCG@1/5/10/20`

## 常用命令

### 1. 预处理数据

```bash
python preprocess.py --raw_path datasets/dataset_TSMC2014_TKY.txt --output_dir datasets/TKY --min_poi_users 5 --min_session_len 3 --min_user_sessions 3 --session_gap_hours 24 --train_ratio 0.8 --geohash_precision 6
```

### 2. 训练并导出专利结果表

```bash
python train.py --data_dir datasets/TKY --meta_path datasets/TKY/meta.pkl --region_precision 5 --seed 2026 --num_epochs 30 --batch_size 1024 --emb_dim 128 --lr 0.001 --decay 0.0005 --dropout 0.3 --keep_rate 1 --keep_rate_poi 1 --num_col_layers 2 --num_reg_layers 2 --num_cat_layers 1 --num_trans_layers 4 --lr_scheduler_factor 0.1 --mask_rate_cat 0.2 --lambda_cat 0.05 --mask_rate_reg 0.2 --lambda_reg 0.05 --mask_alpha 2.0 --save_dir logs_ex --patent_export
```

训练输出目录示例：

- `logs_ex/seed2026_YYYYMMDD_HHMMSS/`

其中常见文件有：

- `log_training.txt`
- `result.txt`
- `args.json`
- `H_DCHL_B.pt`
- `patent_table_final.csv`

### 3. 按序列长度分层抽样

```bash
python zl_seq_sam.py --input "E:\code\lwCode\ZL_DCHL\logs_ex3\seed2026_20260422_211551\patent_table_final.csv" --base-dir "E:\code\lwCode\ZL_DCHL\logs_ex3\seed2026_20260422_211551" --sample-per-group 2 --seed 2026
```

### 4. 全局随机抽样

```bash
python zl_global_seq_sam.py --input "E:\code\lwCode\ZL_DCHL\logs_ex3\seed2026_20260422_211551\patent_table_final.csv" --output "E:\code\lwCode\ZL_DCHL\logs_ex3\seed2026_20260422_211551\patent_table_selected_examples_global.csv" --sample-count 6 --seed 2026
```

## 当前专利导出表说明

`train.py` 在开启 `--patent_export` 后，会导出：

- `patent_table_final.csv`

当前字段为：

- `sample_idx`
- `user_id`
- `prefix_poi_seq`
- `recommended_poi`
- `top1_score`
- `normalized_top1_score`

其中：

- `recommended_poi`：模型对该测试样本推荐的 Top-1 POI
- `top1_score`：Top-1 的原始分数
- `normalized_top1_score`：对 `top1_score` 进行 `sigmoid` 映射后的展示分数

## 说明

- 当前项目中存在部分专利辅助脚本，主要用于结果导出和样本筛选，不影响模型训练主逻辑
- `view.py` 为临时查看脚本，可能包含历史路径，使用前建议自行确认
- 若后续统一改为动态 Region 构图，则 `poi_region.pkl` 可不再作为训练主流程必需文件

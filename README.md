# ann-vector-index-go

一个可在本地运行的近似最近邻（ANN）向量检索库，用可接受的召回损失换取
大规模向量查询开销的下降。纯 Go 实现，无外部依赖。

## 特性

- **确定性**：图是「存活向量集合 + 随机种子」的纯函数。插入顺序、批次划分、
  重复插入同一向量都不改变最终检索结果；相同数据集与相同种子下重复构建得到
  一致结果。
- **召回/延迟可调**：构建参数 `M` 与检索参数 `efSearch` 可配置，`annbench`
  可观测召回率与访问规模的权衡。
- **并发安全**：`sync.RWMutex` 保护全部状态，查询不会读到半更新状态；
  `go test -race` 覆盖并发读写。
- **删除与空间回收**：被删向量不再出现在任何结果中；`Compact` 重建紧凑存储，
  回收全部已删向量占用的空间。
- **持久化与崩溃恢复**：原子写（临时文件 + fsync + rename + 目录 fsync），
  CRC32 校验覆盖全部载荷；截断、坏 CRC、坏 magic、尾部字节等损坏被识别为
  `ErrCorrupt`，绝不静默返回错误结果。
- **精确对照**：`ExactSearch` 暴力计算精确最近邻，`Evaluate` 据此评估召回率。

## 安装

```bash
go get github.com/guoyyit/ann-vector-index-go
```

## 快速开始

```go
cfg := ann.DefaultConfig(128)          // 128 维，默认 L2 距离
cfg.M = 16                             // 邻居预算
cfg.Seed = 42                          // 确定性种子
idx, _ := ann.New(cfg)

// 批量构建或增量插入，顺序任意。
for i, v := range vectors {
    idx.Insert(uint64(i+1), v)
}

// 检索：k=10，ef=64（越大召回越高、访问越多）。
results, _ := idx.Search(query, 10, 64)

// 精确对照评估召回率。
report, _ := idx.Evaluate(queries, 10, 64)
fmt.Printf("recall=%.4f visited=%.1f\n", report.Recall, report.AvgVisited)

// 删除与空间回收。
idx.Delete(id)
idx.Compact()

// 持久化与恢复。
idx.Save("/path/to/index.ann")
loaded, _ := ann.Load("/path/to/index.ann")
```

## 参数语义

| 参数 | 位置 | 语义 | 影响 |
|------|------|------|------|
| `Dim` | Config | 向量维度 | 必须与插入/查询向量一致 |
| `Metric` | Config | 距离度量：`MetricL2`（平方欧氏）、`MetricCosine`（余弦，插入时归一化）、`MetricIP`（内积） | 决定最近邻定义 |
| `M` | Config | 每节点邻居预算；第 0 层用 `2*M`，上层用 `M` | 越大召回越高、构建与检索成本越高 |
| `Seed` | Config | 确定性种子，与 id 一起决定每节点的层分配 | 同种子 → 同图层 |
| `MaxLevel` | Config | 层分配上限（默认 16） | 安全边界 |
| `k` | Search | 返回的最近邻个数 | — |
| `ef` | Search | 检索广度，`ef >= k` | 越大召回越高、访问节点越多 |

层分配由 `splitmix64(id ^ seed)` 确定性派生，与插入顺序无关。

## 一致性保证

- **确定性**：图的构建是数据集的纯函数。`Insert` 仅存储向量并标记 dirty；
  下次检索（或 `Rebuild`/`Compact`/`Save`）时重建为规范图。因此无论以何种顺序、
  何种批次插入，最终检索结果一致。
- **并发**：所有变更（插入、删除、重建）在写锁下执行；检索在写锁下重建并搜索。
  查询期间到达的写入要么在查询前完成、要么在查询后完成，查询永远看到完整状态。
- **删除**：删除立即生效，被删向量不再出现在结果中。反复插入/删除后调用
  `Compact` 回收空间；不 compact 时存储随历史插入增长（有界于累计插入量）。

## 持久化

文件格式：`ANNIDX01` magic + 版本 + 配置头 + 按 id 排序的向量与层出边表 +
CRC32 校验。写入采用临时文件 + fsync + rename，目标文件永远不会处于半写状态。

`Load` 时校验：magic、版本、长度、CRC、id 严格递增、邻居引用存在且在层内、
无自环、无尾部字节。任何不满足都返回 `ErrCorrupt`，不返回部分索引。

## 复杂度

- 构建：O(n² · M)（启发式选择），n 为存活向量数。
- 检索：O(ef · log n) 期望访问规模（实际访问节点数见 `Evaluate.AvgVisited`）。
- 精确检索：O(n · d)，用于对照。

## 基准工具

```bash
go run ./cmd/annbench -n 5000 -dim 32 -clusters 20 -queries 100
```

输出不同 `ef` 下的召回率、平均访问节点数与每查询延迟，可直接观察参数对
召回与访问规模的影响。

## 局限

- 检索在写锁下执行，并发检索会串行化；这是为保证无半更新状态的取舍。
- 构建为 O(n²)，适合本地中等规模（数万向量以内）；更大规模可考虑分片或
  更激进的近似策略。

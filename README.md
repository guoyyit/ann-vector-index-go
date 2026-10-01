# ann-vector-index-go

一个可在本地直接运行的**近似最近邻（ANN）向量检索库**。用可控的召回损失换取
大规模向量查询时访问规模（距离计算次数、探测桶数）的显著下降。

- **算法**：随机超平面 LSH（Random Hyperplane / Sign-Random-Projection LSH），
  多 band 分桶 + 汉明半径多探针（multi-probe）+ 精确距离重排。
- **度量**：余弦距离 `1 - cos(a,b)`（向量入库即归一化）。
- **零第三方依赖**，仅用 Go 标准库；确定性随机数（SplitMix64 + Box-Muller），
  不依赖 `math/rand`，跨 Go 版本/平台结果一致。

---

## 快速开始

```go
import ann "github.com/guoyyit/ann-vector-index-go"

cfg := ann.DefaultConfig(128)      // 128 维；Bits=128, BandWidth=8, Seed=1
cfg.Seed = 42                      // 相同种子 ⇒ 相同超平面 ⇒ 相同结果
ix, _ := ann.New(cfg)

_ = ix.InsertMany(ids, vecs)       // 批量构建

res, stats, _ := ix.Search(query, 10, ann.SearchConfig{
    Radius:        2,  // 每 band 探测汉明半径 0..2 的桶
    MaxCandidates: 0,  // 0 = 不限制候选数
})
// res 按 (距离, ID) 升序，完全确定
// stats 给出本次的 BandsProbed / BucketsProbed / Candidates / Distances ...
```

精确对照与召回评估：

```go
gt, _    := ann.GroundTruth(ix, queries, 10)          // 暴力精确 top-k
report, _:= ann.EvaluateRecall(ix, queries, gt, 10,
    ann.SearchConfig{Radius: 2})
report.MeanRecall  // 平均 Recall@10
report.MinRecall   // 最差查询召回，用于回归判定
// report.Queries[i].Stats 可逐条观察访问规模
```

---

## 参数语义

### 构建参数 `Config`

| 参数 | 含义 | 调节影响 |
|---|---|---|
| `Dim` | 向量维度 | 必须与插入向量一致 |
| `Bits` | 随机超平面（签名比特）数 | 越多签名越精细；默认 128 |
| `BandWidth` | 每个 band 的比特宽度，需整除 `Bits`（≤64） | 越小桶越多越小、命中越严格；默认 8 |
| `Seed` | 超平面随机种子 | **只取决于种子**，同种子结果可复现 |
| `Metric` | 距离度量 | 当前仅 `MetricCosine` |

band 数 `NumBands = Bits / BandWidth`。一个向量在每个 band 内落入且仅落入一个桶
（其签名片段对应的桶）。默认 128/8 ⇒ 16 个 band，每桶期望容纳约 1/256 的数据。

- **调桶宽**：`BandWidth` 增大 → 桶更少更大 → 单次精确桶召回更高但访问更多；
  减小 → 桶更小更多 → 访问更少、召回更依赖多探针。
- **调 band 数**：`Bits/BandWidth` 越大，向量在越多独立表里各有一个桶，
  Radius=0 时取并集，召回提升但访问规模线性上升。

### 查询参数 `SearchConfig`

| 参数 | 含义 | 调节影响 |
|---|---|---|
| `Radius` | 每 band 内探测的汉明半径，范围 `[0, BandWidth]` | **单调**：越大召回越高、`BucketsProbed`/`Candidates` 越多 |
| `MaxCandidates` | 重排候选数上限（按 band 顺序收满即停），`<=0` 不限 | 越小延迟越低，可能牺牲召回；命中时 `Stats.Truncated=true` |

每 band 探测桶数 = `Σ_{r=0}^{Radius} C(BandWidth, r)`，按 (汉明重量, 掩码值)
确定顺序展开，因此访问集合与顺序完全可复现。

### 可观测性 `SearchStats`

`Radius / BandsProbed / BucketsProbed / BucketsHit / Candidates / Distances /
Returned / Truncated / DurationMicros` —— 直接回答“为了这次近似结果，到底看了
多少桶、算了多少距离”，配合 `EvaluateRecall` 即可画出召回—访问规模曲线做调参。

---

## 一致性保证

1. **与插入顺序/批次/重复插入无关**
   桶成员关系只由「向量 × 种子决定的超平面」决定，与到达顺序无关；倒排表按 ID
   排序维护。批量 `InsertMany`、逐条 `Insert`、乱序、任意分批、重复插入同一向量
   （upsert，按 ID 幂等）最终检索结果逐位一致，重复插入不产生重复倒排项。
   同数据集 + 同 `Seed` 重新构建结果一致。

2. **并发安全，读不到半更新**
   所有写操作持写锁，所有读操作持读锁；一次 `Search` 的候选收集与距离重排全部在
   **同一把读锁区间**内完成，看到的是严格一致的单一状态快照。查询期间到达的写入
   在锁外等待，不会读到“向量已删但倒排还在”之类的中间态。已通过 `-race` 长压测。

3. **结果确定性**
   候选按 ID 去重排序后再算距离；top-k 在距离相同时按 ID 决胜并按 (距离, ID)
   排序。因此即便底层使用 map，输出仍与遍历顺序无关、可复现。

4. **删除与空间回收**
   `Delete` 就地压缩倒排切片并删除空桶与主记录，**不保留墓碑**；被删 ID 不会再
   出现在 `Search` 或 `BruteForce` 中。反复插入/删除不会使索引无界膨胀。
   `Rebuild()` 重建全部倒排结构（结果与等价数据全新构建一致），在启用持久化时
   同时做快照并压缩日志。

---

## 持久化与恢复

```go
ix, _ := ann.New(cfg, ann.WithPersistence("index.snap", "index.wal", ann.CorruptionFail))
// 每次 Insert/Delete 都 fsync 进预写日志（WAL）
_ = ix.Save() // 原子快照 + 压缩 WAL
ix2, _ := ann.Load(ann.PersistenceOptions{
    SnapshotPath: "index.snap",
    JournalPath:  "index.wal",
    Policy:       ann.CorruptionFail,
})
```

- **快照**：配置（含种子）+ 全部存活向量（按 ID 排序），带 CRC32；写临时文件 →
  `fsync` → 原子 `rename` → `fsync` 目录。任意时刻磁盘上要么是旧快照要么是新快照。
- **预写日志**：`长度前缀 + payload + CRC32` 的帧，写入即 `fsync`。崩溃后要么整帧
  可见、要么该帧被判为损坏，绝不会把半条写入当成好数据。
- **恢复 = 快照 + 重放其后的完整日志帧**。WAL 与快照都保存**入库时的归一化向量**，
  恢复后距离与内存结果逐位一致（避免二次归一化引入浮点偏差）。
- **损坏处理策略**（不会悄悄返回错误结果）：
  - `CorruptionFail`（默认）：CRC 失败 / 截断帧 / 魔数不符 → 返回 `ErrCorrupt`。
  - `CorruptionIgnoreJournal`：仅对日志，丢弃损坏尾部，保留最近快照 + 完整前缀。
  - `CorruptionRebuild`：无法信任文件时重建空索引并重置日志，可重新落盘。

---

## 测试

```bash
go test ./... -race -count=1
```

覆盖场景：

- **确定性**：顺序/乱序/分批/重复插入结果一致；同种子重建一致；upsert 生效。
- **调参可观测**：召回随 `Radius` 单调不降、访问规模单调上升；全探针召回=1；
  `MaxCandidates` 上限对访问量的约束。
- **并发读写**：多写者持续增删 + 多读者查询，单锁快照内校验无已删/未知 ID、
  无半更新结构；结束后倒排与主记录计数严格自洽（`-race`）。
- **删除回收**：删除后不出现在任何结果；多轮全删全插不倒排膨胀；`Rebuild`
  结果与全新构建一致。
- **持久化恢复**：快照+WAL 往返一致；仅日志重放删除/插入；日志截断坏尾与
  快照 CRC 损坏被识别并按三种策略处理。
- **召回对照**：精确 `GroundTruth` + `EvaluateRecall`，默认参数回归阈值与可复现性。

## 限制与后续可扩展点

- 度量目前仅余弦；L2 可通过增加 p-stable（高斯）LSH 桶扩展。
- `BandWidth` 上限 64（单字内枚举汉明球）；更宽带宽可改用分段枚举。
- 单索引在写路径上串行化（读写锁），适合读多写少；超高频写可在其上做分片。

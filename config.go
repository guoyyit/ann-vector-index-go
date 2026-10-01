package ann

import "fmt"

// Metric 标识索引使用的距离度量。
type Metric int

const (
	// MetricCosine 表示余弦距离（1 - cosine 相似度），向量入库时会被归一化。
	MetricCosine Metric = 1
)

const (
	defaultBits      = 128
	defaultBandWidth = 8
)

// Config 是索引的构建期参数，创建索引后不可变。
//
// 语义：Bits 个随机超平面把一个向量映射为 Bits 位签名；签名按 BandWidth
// 位切成若干 band，每个 band 的值决定该向量落入哪个桶。所有超平面仅由
// Seed 决定，因此相同数据集 + 相同 Seed 下，无论插入顺序、批次如何划分，
// 最终桶成员关系完全一致。
type Config struct {
	// Dim 向量维度，必须 > 0。
	Dim int
	// Bits 随机超平面（签名比特位）数量，必须 > 0。
	Bits int
	// BandWidth 每个 band 的比特宽度，必须 > 0 且能整除 Bits。
	BandWidth int
	// Seed 随机数种子；相同种子生成完全相同的超平面集合。
	Seed int64
	// Metric 距离度量，目前仅支持 MetricCosine。
	Metric Metric
}

// SearchConfig 是单次查询的检索参数，可在召回率与延迟之间权衡：
// Radius 越大，每个 band 探测的近邻桶越多，召回越高、访问规模越大。
type SearchConfig struct {
	// Radius 每个 band 内探测的汉明半径（0 表示只查精确桶）。
	Radius int
	// MaxCandidates 重排候选数上限（按 band 顺序收集满即停），<=0 不限制。
	MaxCandidates int
}

// Result 是单条检索结果。
type Result struct {
	ID       int64
	Distance float64
}

// Stats 是索引的静态/规模统计。
type Stats struct {
	Dim               int
	Bits              int
	BandWidth         int
	NumBands          int
	NumVectors        int
	NumBuckets        int
	NumPostingEntries int
	Journaling        bool
}

// SearchStats 是单次查询的访问规模统计，用于解释召回/延迟权衡。
type SearchStats struct {
	Radius         int
	BandsProbed    int
	BucketsProbed  int
	BucketsHit     int
	Candidates     int
	Distances      int
	Returned       int
	Truncated      bool
	DurationMicros int64
}

// DefaultConfig 返回给定维度下的默认构建参数：
// 128 个超平面、每 band 8 位（16 个 band，每桶期望 1/256 的数据量）。
func DefaultConfig(dim int) Config {
	return Config{
		Dim:       dim,
		Bits:      defaultBits,
		BandWidth: defaultBandWidth,
		Seed:      1,
		Metric:    MetricCosine,
	}
}

// Validate 校验构建参数。
func (c Config) Validate() error {
	if c.Dim <= 0 {
		return fmt.Errorf("%w: Dim must be positive, got %d", ErrInvalidConfig, c.Dim)
	}
	if c.Bits <= 0 {
		return fmt.Errorf("%w: Bits must be positive, got %d", ErrInvalidConfig, c.Bits)
	}
	if c.BandWidth <= 0 {
		return fmt.Errorf("%w: BandWidth must be positive, got %d", ErrInvalidConfig, c.BandWidth)
	}
	if c.BandWidth > c.Bits {
		return fmt.Errorf("%w: BandWidth(%d) must be <= Bits(%d)", ErrInvalidConfig, c.BandWidth, c.Bits)
	}
	if c.Bits%c.BandWidth != 0 {
		return fmt.Errorf("%w: Bits(%d) must be divisible by BandWidth(%d)", ErrInvalidConfig, c.Bits, c.BandWidth)
	}
	if c.Metric != MetricCosine {
		return fmt.Errorf("%w: unsupported metric %d", ErrInvalidConfig, c.Metric)
	}
	return nil
}

// DefaultSearchConfig 返回默认查询参数（Radius=1，候选数不限）。
func DefaultSearchConfig() SearchConfig {
	return SearchConfig{Radius: 1, MaxCandidates: 0}
}

// validate 校验查询参数并把 Radius 收敛到合法范围。
func (s SearchConfig) validate(numBands, bandWidth int) (SearchConfig, error) {
	r := s.Radius
	if r < 0 {
		return s, fmt.Errorf("%w: Radius must be >= 0, got %d", ErrInvalidConfig, r)
	}
	if r > bandWidth {
		// 汉明距离不可能超过 band 宽度。
		r = bandWidth
	}
	return SearchConfig{Radius: r, MaxCandidates: s.MaxCandidates}, nil
}

package ann

// QueryRecall 是单条查询的召回评估结果。
type QueryRecall struct {
	RecallAtK float64 // 命中精确 top-k 的比例（含 ID 重叠 / k）
	Hit       int
	K         int
	Expected  []Result // 精确最近邻
	Got       []Result // 近似最近邻
	Stats     SearchStats
}

// RecallReport 汇总多条查询的评估结果，用于验收调参与回归判定。
type RecallReport struct {
	K          int
	NumQueries int
	MeanRecall float64
	MinRecall  float64
	MaxRecall  float64
	Queries    []QueryRecall
}

// GroundTruth 用精确线性扫描对一组查询计算 top-k 最近邻，作为基准。
func GroundTruth(ix *Index, queries [][]float64, k int) ([][]Result, error) {
	out := make([][]Result, len(queries))
	for i, q := range queries {
		r, err := ix.BruteForce(q, k)
		if err != nil {
			return nil, err
		}
		out[i] = r
	}
	return out, nil
}

// EvaluateRecall 对同一批查询对比近似结果与给定精确基准。
// groundTruth 由 GroundTruth 生成。
func EvaluateRecall(ix *Index, queries [][]float64, groundTruth [][]Result, k int, sc SearchConfig) (RecallReport, error) {
	if k <= 0 {
		return RecallReport{}, ErrK
	}
	if len(queries) != len(groundTruth) {
		return RecallReport{}, ErrInvalidConfig
	}
	rep := RecallReport{K: k, NumQueries: len(queries), MinRecall: 1}
	var sum float64
	for i, q := range queries {
		got, st, err := ix.Search(q, k, sc)
		if err != nil {
			return RecallReport{}, err
		}
		want := groundTruth[i]
		hit := overlapCount(resultIDs(want), resultIDs(got))
		denom := k
		if len(want) < denom {
			denom = len(want)
		}
		if denom == 0 {
			denom = 1
		}
		r := float64(hit) / float64(denom)
		sum += r
		if r < rep.MinRecall {
			rep.MinRecall = r
		}
		if r > rep.MaxRecall {
			rep.MaxRecall = r
		}
		rep.Queries = append(rep.Queries, QueryRecall{
			RecallAtK: r, Hit: hit, K: denom,
			Expected: want, Got: got, Stats: st,
		})
	}
	if rep.NumQueries > 0 {
		rep.MeanRecall = sum / float64(rep.NumQueries)
	} else {
		rep.MinRecall = 0
	}
	return rep, nil
}

func resultIDs(rs []Result) map[int64]struct{} {
	m := make(map[int64]struct{}, len(rs))
	for _, r := range rs {
		m[r.ID] = struct{}{}
	}
	return m
}

func overlapCount(a, b map[int64]struct{}) int {
	n := 0
	for id := range a {
		if _, ok := b[id]; ok {
			n++
		}
	}
	return n
}

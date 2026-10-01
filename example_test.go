package ann_test

import (
	"fmt"

	ann "github.com/guoyyit/ann-vector-index-go"
)

// 演示：构建、调参查询、精确对照评估召回率。
func Example() {
	cfg := ann.DefaultConfig(8) // 8 维向量，128 超平面 / 每 band 8 位 / 种子 1
	cfg.Seed = 42

	ix, err := ann.New(cfg)
	if err != nil {
		panic(err)
	}

	// 批量构建（与逐条插入、乱序插入、分批插入结果一致）。
	ids := []int64{1, 2, 3, 4}
	vecs := [][]float64{
		{1, 0, 0, 0, 0, 0, 0, 0},
		{0.99, 0.01, 0, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 1},
	}
	if err := ix.InsertMany(ids, vecs); err != nil {
		panic(err)
	}

	q := []float64{1, 0, 0, 0, 0, 0, 0, 0}
	res, stats, err := ix.Search(q, 2, ann.SearchConfig{Radius: 2})
	if err != nil {
		panic(err)
	}
	for _, r := range res {
		fmt.Printf("id=%d d=%.4f\n", r.ID, r.Distance)
	}
	fmt.Printf("candidates=%d buckets=%d\n", stats.Candidates, stats.BucketsProbed)

	// 精确对照与召回评估。
	gt, _ := ann.GroundTruth(ix, [][]float64{q}, 2)
	report, _ := ann.EvaluateRecall(ix, [][]float64{q}, gt, 2, ann.SearchConfig{Radius: 2})
	fmt.Printf("recall@2=%.2f\n", report.MeanRecall)
	// Output:
	// id=1 d=0.0000
	// id=2 d=0.0001
	// candidates=4 buckets=592
	// recall@2=1.00
}

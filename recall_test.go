package ann

import "testing"

// TestRecallEvaluation 验证精确对照、召回评估接口，并用它做参数回归判定。
func TestRecallEvaluation(t *testing.T) {
	cfg := DefaultConfig(16)
	cfg.Seed = 2025
	ix, _ := New(cfg)
	const n = 800
	ids, vecs := genData(n, 16, 4242)
	if err := ix.InsertMany(ids, vecs); err != nil {
		t.Fatal(err)
	}
	queries := vecs[40:90]
	const k = 10

	gt, err := GroundTruth(ix, queries, k)
	if err != nil {
		t.Fatal(err)
	}
	if len(gt) != len(queries) || len(gt[0]) != k {
		t.Fatal("ground truth shape wrong")
	}

	// 精确扫描对自身评估必然召回 1（近似结果即精确结果）。
	repFull, err := EvaluateRecall(ix, queries, gt, k, SearchConfig{Radius: cfg.BandWidth})
	if err != nil {
		t.Fatal(err)
	}
	if repFull.MeanRecall != 1.0 || repFull.MinRecall != 1.0 {
		t.Fatalf("full probe recall = %v/%v, want 1.0", repFull.MeanRecall, repFull.MinRecall)
	}

	// 回归基线：默认参数应达到较高召回，访问规模显著小于全量。
	rep, err := EvaluateRecall(ix, queries, gt, k, DefaultSearchConfig())
	if err != nil {
		t.Fatal(err)
	}
	totalCand, totalBuckets := 0, 0
	for _, q := range rep.Queries {
		totalCand += q.Stats.Candidates
		totalBuckets += q.Stats.BucketsProbed
	}
	avgCand := totalCand / len(rep.Queries)
	t.Logf("default: meanRecall=%.4f minRecall=%.2f avgCandidates=%d/%d avgBucketsProbed=%d",
		rep.MeanRecall, rep.MinRecall, avgCand, n, totalBuckets/len(rep.Queries))
	if rep.MeanRecall < 0.90 {
		t.Fatalf("default recall regression: %.3f", rep.MeanRecall)
	}
	if avgCand >= n {
		t.Fatal("default search did not reduce candidate visits")
	}

	// radius=0 平均访问规模应更小，召回一般更低（不做硬召回断言，只验排序可解释）
	rep0, _ := EvaluateRecall(ix, queries, gt, k, SearchConfig{Radius: 0})
	c0, c1 := 0, 0
	for i := range rep0.Queries {
		c0 += rep0.Queries[i].Stats.Candidates
		c1 += rep.Queries[i].Stats.Candidates
	}
	if c0 > c1 {
		t.Fatal("radius=0 visited more candidates than radius=1")
	}

	// 评估结果可复现：重复评估数值一致
	rep2, _ := EvaluateRecall(ix, queries, gt, k, DefaultSearchConfig())
	if rep.MeanRecall != rep2.MeanRecall || rep.MinRecall != rep2.MinRecall {
		t.Fatal("recall evaluation not reproducible")
	}
}

// TestMaxCandidatesCap 验证候选上限对访问规模与统计可观测性的影响。
func TestMaxCandidatesCap(t *testing.T) {
	cfg := DefaultConfig(16)
	ix, _ := New(cfg)
	ids, vecs := genData(500, 16, 77)
	_ = ix.InsertMany(ids, vecs)
	q := vecs[0]
	_, stU, err := ix.Search(q, 10, SearchConfig{Radius: 3, MaxCandidates: 0})
	if err != nil {
		t.Fatal(err)
	}
	capped, stC, err := ix.Search(q, 10, SearchConfig{Radius: 3, MaxCandidates: 30})
	if err != nil {
		t.Fatal(err)
	}
	if !stC.Truncated || stC.Candidates > 30 {
		t.Fatalf("cap not honored: %+v", stC)
	}
	if stC.Candidates >= stU.Candidates {
		t.Fatalf("cap did not reduce candidates: %d vs %d", stC.Candidates, stU.Candidates)
	}
	// 受限搜索距离计算量必须受限于候选上限，且结果条数不超过 k。
	if stC.Distances > 30 || len(capped) > 10 {
		t.Fatalf("cap did not bound work: %+v", stC)
	}
	// 提高上限应单调放松访问规模。
	_, stC2, err := ix.Search(q, 10, SearchConfig{Radius: 3, MaxCandidates: 100})
	if err != nil {
		t.Fatal(err)
	}
	if stC2.Candidates < stC.Candidates {
		t.Fatal("larger cap visited fewer candidates")
	}
}

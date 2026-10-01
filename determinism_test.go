package ann

import (
	"reflect"
	"testing"
)

func TestDeterminism_OrderBatchesDuplicates(t *testing.T) {
	cfg := DefaultConfig(12)
	cfg.Seed = 123
	const n, k = 300, 10
	ids, vecs := genData(n, 12, 99)
	queries := vecs[50:70]

	build := func() *Index {
		ix, err := New(cfg)
		if err != nil {
			t.Fatal(err)
		}
		return ix
	}
	searchAll := func(ix *Index, sc SearchConfig) [][]Result {
		out := make([][]Result, len(queries))
		for i, q := range queries {
			r, _, err := ix.Search(q, k, sc)
			if err != nil {
				t.Fatal(err)
			}
			out[i] = r
		}
		return out
	}

	// A: 一次性批量
	a := build()
	if err := a.InsertMany(ids, vecs); err != nil {
		t.Fatal(err)
	}
	resA := searchAll(a, SearchConfig{Radius: 2})

	// B: 逐条，正序
	b := build()
	for i := range ids {
		if err := b.Insert(ids[i], vecs[i]); err != nil {
			t.Fatal(err)
		}
	}
	resB := searchAll(b, SearchConfig{Radius: 2})

	// C: 逐条，乱序
	c := build()
	perm := rngPerm(n, 2024)
	for _, i := range perm {
		if err := c.Insert(ids[i], vecs[i]); err != nil {
			t.Fatal(err)
		}
	}
	resC := searchAll(c, SearchConfig{Radius: 2})

	// D: 切成不同批次
	d := build()
	if err := d.InsertMany(ids[:37], vecs[:37]); err != nil {
		t.Fatal(err)
	}
	if err := d.InsertMany(ids[37:200], vecs[37:200]); err != nil {
		t.Fatal(err)
	}
	if err := d.InsertMany(ids[200:], vecs[200:]); err != nil {
		t.Fatal(err)
	}
	resD := searchAll(d, SearchConfig{Radius: 2})

	if !reflect.DeepEqual(resA, resB) {
		t.Fatal("batch vs sequential order mismatch")
	}
	if !reflect.DeepEqual(resA, resC) {
		t.Fatal("insert order mismatch")
	}
	if !reflect.DeepEqual(resA, resD) {
		t.Fatal("batch partitioning mismatch")
	}

	// E: 重复插入相同向量（幂等），结果与规模都不变
	beforeStats := a.Stats()
	for i := 0; i < n; i += 3 {
		if err := a.Insert(ids[i], vecs[i]); err != nil {
			t.Fatal(err)
		}
	}
	if a.Len() != n {
		t.Fatalf("duplicate insert changed Len: %d", a.Len())
	}
	afterStats := a.Stats()
	if afterStats.NumPostingEntries != beforeStats.NumPostingEntries {
		t.Fatalf("duplicate insert inflated postings: %d vs %d",
			afterStats.NumPostingEntries, beforeStats.NumPostingEntries)
	}
	resE := searchAll(a, SearchConfig{Radius: 2})
	if !reflect.DeepEqual(resA, resE) {
		t.Fatal("duplicate insert changed results")
	}

	// F: 全新索引同种子重建，结果一致
	f := build()
	if err := f.InsertMany(ids, vecs); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(resA, searchAll(f, SearchConfig{Radius: 2})) {
		t.Fatal("same seed rebuild mismatch")
	}
}

func TestUpsertChangesResult(t *testing.T) {
	cfg := DefaultConfig(8)
	ix, _ := New(cfg)
	if err := ix.Insert(1, []float64{1, 0, 0, 0, 0, 0, 0, 0}); err != nil {
		t.Fatal(err)
	}
	// 用同一个 ID 写入完全相反的向量
	if err := ix.Insert(1, []float64{-1, 0, 0, 0, 0, 0, 0, 0}); err != nil {
		t.Fatal(err)
	}
	if ix.Len() != 1 {
		t.Fatalf("upsert changed count: %d", ix.Len())
	}
	// 精确桶已不再包含它（签名完全相反），radius=0 应查不到
	if got, _, _ := ix.Search([]float64{1, 0, 0, 0, 0, 0, 0, 0}, 1, SearchConfig{Radius: 0}); len(got) != 0 {
		t.Fatalf("stale vector still in exact bucket: %+v", got)
	}
	// 全半径重排应得到它，且距离接近 2
	got, _, err := ix.Search([]float64{1, 0, 0, 0, 0, 0, 0, 0}, 1, SearchConfig{Radius: cfg.BandWidth})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].ID != 1 {
		t.Fatalf("unexpected %+v", got)
	}
	if got[0].Distance < 1.9 {
		t.Fatalf("upsert did not take effect, distance=%v", got[0].Distance)
	}
}

func TestParameterTuningMonotonic(t *testing.T) {
	cfg := DefaultConfig(16)
	cfg.Seed = 5
	ix, _ := New(cfg)
	ids, vecs := genData(600, 16, 31)
	if err := ix.InsertMany(ids, vecs); err != nil {
		t.Fatal(err)
	}
	queries := vecs[:30]
	gt, err := GroundTruth(ix, queries, 10)
	if err != nil {
		t.Fatal(err)
	}
	var prevRecall, prevCand float64 = -1, -1
	for _, r := range []int{0, 1, 2, 3, 4, 8} {
		rep, err := EvaluateRecall(ix, queries, gt, 10, SearchConfig{Radius: r})
		if err != nil {
			t.Fatal(err)
		}
		avgCand := 0
		for _, q := range rep.Queries {
			avgCand += q.Stats.Candidates
		}
		avgCand /= len(rep.Queries)
		t.Logf("radius=%d recall=%.3f avgCandidates=%d avgBuckets=%d",
			r, rep.MeanRecall, avgCand, avgBuckets(rep))
		if rep.MeanRecall+1e-9 < prevRecall {
			t.Fatalf("recall decreased when radius grew: %.3f -> %.3f", prevRecall, rep.MeanRecall)
		}
		if float64(avgCand)+1e-9 < prevCand {
			t.Fatalf("candidate set shrank when radius grew")
		}
		prevRecall, prevCand = rep.MeanRecall, float64(avgCand)
	}
	// radius=0 召回严格低于全量；radius 足够大应达到精确召回
	full, _ := EvaluateRecall(ix, queries, gt, 10, SearchConfig{Radius: cfg.BandWidth})
	if full.MeanRecall != 1.0 {
		t.Fatalf("full-radius recall not 1.0: %v", full.MeanRecall)
	}
}

func avgBuckets(r RecallReport) int {
	s := 0
	for _, q := range r.Queries {
		s += q.Stats.BucketsProbed
	}
	return s / len(r.Queries)
}

// rngPerm 生成确定性的乱序索引。
func rngPerm(n int, seed int64) []int {
	r := newRNG(seed)
	p := make([]int, n)
	for i := range p {
		p[i] = i
	}
	for i := n - 1; i > 0; i-- {
		j := int(r.Uint64() % uint64(i+1))
		p[i], p[j] = p[j], p[i]
	}
	return p
}

func TestValidation(t *testing.T) {
	if _, err := New(Config{Dim: 0}); err == nil {
		t.Fatal("expected invalid Dim")
	}
	if _, err := New(Config{Dim: 4, Bits: 10, BandWidth: 3}); err == nil {
		t.Fatal("expected non-divisible config error")
	}
	ix, _ := New(DefaultConfig(4))
	if err := ix.Insert(1, []float64{1, 2}); err != ErrDimensionMismatch {
		t.Fatalf("expected dimension mismatch, got %v", err)
	}
	if _, _, err := ix.Search([]float64{1, 2}, 1, DefaultSearchConfig()); err != ErrDimensionMismatch {
		t.Fatalf("expected dimension mismatch, got %v", err)
	}
	if _, _, err := ix.Search([]float64{1, 0, 0, 0}, 0, DefaultSearchConfig()); err != ErrK {
		t.Fatalf("expected ErrK, got %v", err)
	}
}

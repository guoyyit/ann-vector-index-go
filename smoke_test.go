package ann

import (
	"testing"
)

func genData(n, dim int, seed int64) ([]int64, [][]float64) {
	r := newRNG(seed)
	ids := make([]int64, n)
	vecs := make([][]float64, n)
	for i := 0; i < n; i++ {
		ids[i] = int64(i + 1)
		v := make([]float64, dim)
		for d := range v {
			v[d] = r.Normal()
		}
		vecs[i] = v
	}
	return ids, vecs
}

func TestSmokeRecall(t *testing.T) {
	cfg := DefaultConfig(16)
	cfg.Seed = 42
	ix, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	ids, vecs := genData(400, 16, 7)
	if err := ix.InsertMany(ids, vecs); err != nil {
		t.Fatal(err)
	}
	queries := vecs[:20]
	gt, err := GroundTruth(ix, queries, 10)
	if err != nil {
		t.Fatal(err)
	}
	for _, radius := range []int{0, 1, 2, 3} {
		rep, err := EvaluateRecall(ix, queries, gt, 10, SearchConfig{Radius: radius})
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("radius=%d meanRecall=%.3f min=%.3f avgCand=%d",
			radius, rep.MeanRecall, rep.MinRecall, avgCandidates(rep))
	}
}

func avgCandidates(r RecallReport) int {
	s := 0
	for _, q := range r.Queries {
		s += q.Stats.Candidates
	}
	if len(r.Queries) == 0 {
		return 0
	}
	return s / len(r.Queries)
}

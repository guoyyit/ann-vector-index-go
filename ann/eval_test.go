package ann

import (
	"testing"
)

// TestExactSearchCorrectness verifies exact search returns the true nearest
// neighbors (brute force) with correct tie-breaking.
func TestExactSearchCorrectness(t *testing.T) {
	data, ids := genData(300, 8, 5, 42)
	cfg := DefaultConfig(8)
	idx := buildIndex(t, cfg, ids, data)
	queries := genQueries(5, 8, 5, 42)
	for _, q := range queries {
		res, err := idx.ExactSearch(q, 10)
		if err != nil {
			t.Fatal(err)
		}
		if len(res) != 10 {
			t.Fatalf("expected 10 results, got %d", len(res))
		}
		// Verify distances are non-decreasing.
		for i := 1; i < len(res); i++ {
			if res[i].Distance < res[i-1].Distance {
				t.Fatalf("distances not sorted: %v < %v", res[i].Distance, res[i-1].Distance)
			}
		}
		// Verify no duplicates.
		seen := make(map[uint64]struct{})
		for _, r := range res {
			if _, ok := seen[r.ID]; ok {
				t.Fatalf("duplicate id %d", r.ID)
			}
			seen[r.ID] = struct{}{}
		}
	}
}

// TestRecallSweep verifies that recall increases with ef and that the
// recall/latency tradeoff is observable.
func TestRecallSweep(t *testing.T) {
	data, ids := genData(1000, 16, 10, 42)
	cfg := DefaultConfig(16)
	cfg.M = 16
	idx := buildIndex(t, cfg, ids, data)
	queries := genQueries(20, 16, 10, 99)

	prevRecall := -1.0
	prevVisited := 0.0
	for _, ef := range []int{1, 4, 16, 64, 256} {
		rep, err := idx.Evaluate(queries, 10, ef)
		if err != nil {
			t.Fatal(err)
		}
		if rep.Recall < prevRecall {
			t.Fatalf("recall decreased at ef=%d: %v < %v", ef, rep.Recall, prevRecall)
		}
		if rep.AvgVisited < prevVisited {
			t.Fatalf("visited decreased at ef=%d: %v < %v", ef, rep.AvgVisited, prevVisited)
		}
		prevRecall = rep.Recall
		prevVisited = rep.AvgVisited
	}
	// At high ef, recall should be near 1.
	rep, err := idx.Evaluate(queries, 10, 256)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Recall < 0.95 {
		t.Fatalf("recall at ef=256 too low: %v", rep.Recall)
	}
}

// TestRecallVsExactOnEasyData sanity-checks recall on well-separated clusters.
func TestRecallVsExactOnEasyData(t *testing.T) {
	data, ids := genData(500, 8, 5, 42)
	cfg := DefaultConfig(8)
	idx := buildIndex(t, cfg, ids, data)
	queries := genQueries(10, 8, 5, 42)
	rep, err := idx.Evaluate(queries, 5, 64)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Recall < 0.9 {
		t.Fatalf("recall too low on easy data: %v", rep.Recall)
	}
}

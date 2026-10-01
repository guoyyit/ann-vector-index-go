package ann

import (
	"errors"
	"math/rand"
	"testing"
)

// genData generates n clustered vectors of dimension dim with k clusters.
func genData(n, dim, k int, seed int64) (data [][]float32, ids []uint64) {
	rng := rand.New(rand.NewSource(seed))
	centers := make([][]float32, k)
	for c := range centers {
		v := make([]float32, dim)
		for i := range v {
			v[i] = rng.Float32()*2 - 1
		}
		centers[c] = v
	}
	data = make([][]float32, n)
	ids = make([]uint64, n)
	for i := 0; i < n; i++ {
		c := rng.Intn(k)
		v := make([]float32, dim)
		for j := range v {
			v[j] = centers[c][j] + float32(rng.NormFloat64()*0.1)
		}
		data[i] = v
		ids[i] = uint64(i + 1)
	}
	return
}

func genQueries(m, dim, k int, seed int64) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	centers := make([][]float32, k)
	for c := range centers {
		v := make([]float32, dim)
		for i := range v {
			v[i] = rng.Float32()*2 - 1
		}
		centers[c] = v
	}
	queries := make([][]float32, m)
	for i := range queries {
		c := rng.Intn(k)
		v := make([]float32, dim)
		for j := range v {
			v[j] = centers[c][j] + float32(rng.NormFloat64()*0.1)
		}
		queries[i] = v
	}
	return queries
}

func buildIndex(t *testing.T, cfg Config, ids []uint64, data [][]float32) *Index {
	t.Helper()
	idx, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i := range data {
		if err := idx.Insert(ids[i], data[i]); err != nil {
			t.Fatal(err)
		}
	}
	if err := idx.Verify(); err != nil {
		t.Fatal(err)
	}
	return idx
}

func sameResults(t *testing.T, a, b *Index, q []float32, k, ef int) bool {
	t.Helper()
	ra, err := a.Search(q, k, ef)
	if err != nil {
		t.Fatal(err)
	}
	rb, err := b.Search(q, k, ef)
	if err != nil {
		t.Fatal(err)
	}
	if len(ra) != len(rb) {
		return false
	}
	for i := range ra {
		if ra[i] != rb[i] {
			return false
		}
	}
	return true
}

func TestDeterminism(t *testing.T) {
	data, ids := genData(1000, 16, 10, 42)
	cfg := DefaultConfig(16)
	cfg.M = 16
	queries := genQueries(20, 16, 10, 99)

	// Reference: inserted in id order.
	ref := buildIndex(t, cfg, ids, data)

	// Random insertion order.
	idx2, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	perm := rand.Perm(len(ids))
	for _, i := range perm {
		if err := idx2.Insert(ids[i], data[i]); err != nil {
			t.Fatal(err)
		}
	}
	if err := idx2.Verify(); err != nil {
		t.Fatal(err)
	}

	// Batch insertion with varying batch sizes.
	idx3, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for b := 0; b < len(ids); b += 137 {
		end := b + 137
		if end > len(ids) {
			end = len(ids)
		}
		if err := idx3.InsertBatch(ids[b:end], data[b:end]); err != nil {
			t.Fatal(err)
		}
	}
	if err := idx3.Verify(); err != nil {
		t.Fatal(err)
	}

	// Duplicate inserts of the same id must not change results.
	for i := 0; i < 50; i++ {
		if err := idx3.Insert(ids[i], data[i]); err != nil {
			t.Fatal(err)
		}
	}
	if err := idx3.Verify(); err != nil {
		t.Fatal(err)
	}

	for _, q := range queries {
		if !sameResults(t, ref, idx2, q, 10, 64) {
			t.Fatal("results differ for random insertion order")
		}
		if !sameResults(t, ref, idx3, q, 10, 64) {
			t.Fatal("results differ for batch/duplicate insertion")
		}
	}
}

func TestSeedDeterminism(t *testing.T) {
	data, ids := genData(500, 8, 5, 7)
	cfg := DefaultConfig(8)
	cfg.Seed = 123
	a := buildIndex(t, cfg, ids, data)
	b := buildIndex(t, cfg, ids, data)
	queries := genQueries(10, 8, 5, 7)
	for _, q := range queries {
		if !sameResults(t, a, b, q, 5, 32) {
			t.Fatal("same seed produced different results")
		}
	}
	// Different seed must not crash; results may differ.
	cfg2 := cfg
	cfg2.Seed = 456
	c := buildIndex(t, cfg2, ids, data)
	if err := c.Verify(); err != nil {
		t.Fatal(err)
	}
}

func TestDelete(t *testing.T) {
	data, ids := genData(500, 8, 5, 42)
	cfg := DefaultConfig(8)
	idx := buildIndex(t, cfg, ids, data)

	if err := idx.Delete(ids[0]); err != nil {
		t.Fatal(err)
	}
	if err := idx.Delete(ids[1]); err != nil {
		t.Fatal(err)
	}
	if err := idx.Delete(ids[0]); !errors.Is(err, ErrNotFound) {
		t.Fatalf("expected ErrNotFound, got %v", err)
	}
	if idx.Len() != 498 {
		t.Fatalf("expected 498 live, got %d", idx.Len())
	}
	queries := genQueries(10, 8, 5, 42)
	for _, q := range queries {
		res, err := idx.Search(q, 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range res {
			if r.ID == ids[0] || r.ID == ids[1] {
				t.Fatalf("deleted id %d appeared in results", r.ID)
			}
		}
	}
	if err := idx.Verify(); err != nil {
		t.Fatal(err)
	}
}

func TestCompactReclaimsSpace(t *testing.T) {
	data, ids := genData(500, 8, 5, 42)
	cfg := DefaultConfig(8)
	idx := buildIndex(t, cfg, ids, data)

	// Delete half the vectors.
	for i := 0; i < 250; i++ {
		if err := idx.Delete(ids[i]); err != nil {
			t.Fatal(err)
		}
	}
	if idx.Stats().Slots != 500 {
		t.Fatalf("expected 500 slots before compact, got %d", idx.Stats().Slots)
	}
	if err := idx.Compact(); err != nil {
		t.Fatal(err)
	}
	st := idx.Stats()
	if st.Slots != 250 {
		t.Fatalf("expected 250 slots after compact, got %d", st.Slots)
	}
	if st.Live != 250 {
		t.Fatalf("expected 250 live after compact, got %d", st.Live)
	}
	if err := idx.Verify(); err != nil {
		t.Fatal(err)
	}
	// Results after compact must match a fresh build of the live set.
	liveIds := ids[250:]
	liveData := data[250:]
	ref := buildIndex(t, cfg, liveIds, liveData)
	queries := genQueries(10, 8, 5, 42)
	for _, q := range queries {
		if !sameResults(t, idx, ref, q, 10, 64) {
			t.Fatal("results differ after compact")
		}
	}
}

func TestInsertDeleteCycles(t *testing.T) {
	data, ids := genData(300, 8, 5, 42)
	cfg := DefaultConfig(8)
	idx, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	// Repeated insert/delete cycles must not inflate storage after compact.
	for cycle := 0; cycle < 5; cycle++ {
		for i := range ids {
			if err := idx.Insert(ids[i], data[i]); err != nil {
				t.Fatal(err)
			}
		}
		for i := range ids {
			if err := idx.Delete(ids[i]); err != nil {
				t.Fatal(err)
			}
		}
	}
	if err := idx.Compact(); err != nil {
		t.Fatal(err)
	}
	if idx.Stats().Slots != 0 || idx.Len() != 0 {
		t.Fatalf("expected empty after cycles, slots=%d live=%d", idx.Stats().Slots, idx.Len())
	}
	// Re-insert and verify it works.
	for i := 0; i < 100; i++ {
		if err := idx.Insert(ids[i], data[i]); err != nil {
			t.Fatal(err)
		}
	}
	if err := idx.Verify(); err != nil {
		t.Fatal(err)
	}
}

func TestMetrics(t *testing.T) {
	data, ids := genData(500, 16, 8, 42)
	queries := genQueries(10, 16, 8, 42)
	for _, m := range []Metric{MetricL2, MetricCosine, MetricIP} {
		cfg := DefaultConfig(16)
		cfg.Metric = m
		idx := buildIndex(t, cfg, ids, data)
		rep, err := idx.Evaluate(queries, 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		if rep.Recall < 0.9 {
			t.Fatalf("metric %d recall too low: %v", m, rep.Recall)
		}
	}
}

func TestErrorHandling(t *testing.T) {
	cfg := DefaultConfig(8)
	idx, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	vec := make([]float32, 8)
	// Empty index search.
	if _, err := idx.Search(vec, 5, 32); !errors.Is(err, ErrEmptyIndex) {
		t.Fatalf("expected ErrEmptyIndex, got %v", err)
	}
	// Insert wrong dim.
	bad := make([]float32, 7)
	if err := idx.Insert(1, bad); !errors.Is(err, ErrDimMismatch) {
		t.Fatalf("expected ErrDimMismatch, got %v", err)
	}
	// InsertBatch length mismatch.
	if err := idx.InsertBatch([]uint64{1, 2}, [][]float32{vec}); !errors.Is(err, ErrDimMismatch) {
		t.Fatalf("expected ErrDimMismatch, got %v", err)
	}
	// Search wrong dim.
	idx.Insert(1, vec)
	if _, err := idx.Search(bad, 1, 1); !errors.Is(err, ErrDimMismatch) {
		t.Fatalf("expected ErrDimMismatch, got %v", err)
	}
	// k <= 0 (index has a vector).
	idx.Insert(1, vec)
	if _, err := idx.Search(vec, 0, 1); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("expected ErrInvalidConfig, got %v", err)
	}
	// Bad config.
	if _, err := New(Config{Dim: 0}); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("expected ErrInvalidConfig for dim=0, got %v", err)
	}
	if _, err := New(Config{Dim: 8, M: 0}); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("expected ErrInvalidConfig for M=0, got %v", err)
	}
}

func TestCosineZeroVector(t *testing.T) {
	cfg := DefaultConfig(4)
	cfg.Metric = MetricCosine
	idx, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	zero := make([]float32, 4)
	if err := idx.Insert(1, zero); err != nil {
		t.Fatal(err)
	}
	if err := idx.Insert(2, []float32{1, 0, 0, 0}); err != nil {
		t.Fatal(err)
	}
	if err := idx.Verify(); err != nil {
		t.Fatal(err)
	}
	res, err := idx.Search([]float32{1, 0, 0, 0}, 2, 16)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 2 {
		t.Fatalf("expected 2 results, got %d", len(res))
	}
}

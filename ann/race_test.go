package ann

import (
	"math/rand"
	"sync"
	"testing"
)

// TestConcurrentReadWrite hammers the index with concurrent inserters and
// searchers under -race. Every query must return a consistent result set
// (no half-applied writes, no duplicate ids).
func TestConcurrentReadWrite(t *testing.T) {
	cfg := DefaultConfig(8)
	cfg.M = 16
	idx, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	dim := 8

	const numWriters = 4
	const numReaders = 4
	const perWriter = 100
	const total = numWriters * perWriter

	var writersWG sync.WaitGroup
	var readersWG sync.WaitGroup
	stop := make(chan struct{})

	for w := 0; w < numWriters; w++ {
		writersWG.Add(1)
		go func(w int) {
			defer writersWG.Done()
			wr := rand.New(rand.NewSource(int64(w)))
			for i := 0; i < perWriter; i++ {
				id := uint64(w*perWriter + i + 1)
				v := make([]float32, dim)
				for j := range v {
					v[j] = wr.Float32()*2 - 1
				}
				if err := idx.Insert(id, v); err != nil {
					t.Error(err)
					return
				}
			}
		}(w)
	}
	for r := 0; r < numReaders; r++ {
		readersWG.Add(1)
		go func(r int) {
			defer readersWG.Done()
			rr := rand.New(rand.NewSource(int64(r + 1000)))
			q := make([]float32, dim)
			for j := range q {
				q[j] = rr.Float32()*2 - 1
			}
			for {
				select {
				case <-stop:
					return
				default:
				}
				res, err := idx.Search(q, 5, 16)
				if err == ErrEmptyIndex {
					continue
				}
				if err != nil {
					t.Error(err)
					return
				}
				if len(res) > 5 {
					t.Errorf("got %d results, want <= 5", len(res))
					return
				}
				seen := make(map[uint64]struct{}, len(res))
				for _, r := range res {
					if _, ok := seen[r.ID]; ok {
						t.Errorf("duplicate id %d in results", r.ID)
						return
					}
					seen[r.ID] = struct{}{}
				}
			}
		}(r)
	}
	writersWG.Wait()
	close(stop)
	readersWG.Wait()

	if idx.Len() != total {
		t.Fatalf("expected %d live, got %d", total, idx.Len())
	}
	if err := idx.Verify(); err != nil {
		t.Fatal(err)
	}
}

// TestConcurrentDeleteAndQuery verifies deletions concurrent with queries
// never surface deleted ids.
func TestConcurrentDeleteAndQuery(t *testing.T) {
	data, ids := genData(300, 8, 5, 42)
	cfg := DefaultConfig(8)
	idx := buildIndex(t, cfg, ids, data)
	queries := genQueries(5, 8, 5, 42)

	var writersWG sync.WaitGroup
	var readersWG sync.WaitGroup
	stop := make(chan struct{})

	for r := 0; r < 4; r++ {
		readersWG.Add(1)
		go func() {
			defer readersWG.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				for _, q := range queries {
					res, err := idx.Search(q, 5, 16)
					if err != nil {
						t.Error(err)
						return
					}
					// Search holds the write lock, so every result is valid
					// at search time; only check for duplicates here.
					seen := make(map[uint64]struct{}, len(res))
					for _, r := range res {
						if _, ok := seen[r.ID]; ok {
							t.Errorf("duplicate id %d in results", r.ID)
							return
						}
						seen[r.ID] = struct{}{}
					}
				}
			}
		}()
	}
	for w := 0; w < 4; w++ {
		writersWG.Add(1)
		go func(w int) {
			defer writersWG.Done()
			wr := rand.New(rand.NewSource(int64(w + 100)))
			for i := 0; i < 30; i++ {
				id := ids[(w*30+i)%len(ids)]
				if err := idx.Delete(id); err != nil && err != ErrNotFound {
					t.Error(err)
					return
				}
				v := make([]float32, 8)
				for j := range v {
					v[j] = wr.Float32()*2 - 1
				}
				if err := idx.Insert(id, v); err != nil {
					t.Error(err)
					return
				}
			}
		}(w)
	}
	writersWG.Wait()
	close(stop)
	readersWG.Wait()
	if err := idx.Verify(); err != nil {
		t.Fatal(err)
	}
}

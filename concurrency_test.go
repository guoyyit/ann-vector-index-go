package ann

import (
	"sync"
	"testing"
)

func TestConcurrentReadWrite(t *testing.T) {
	cfg := DefaultConfig(12)
	cfg.Seed = 77
	ix, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	const total = 1000
	ids, vecs := genData(total, 12, 555)
	if err := ix.InsertMany(ids, vecs); err != nil {
		t.Fatal(err)
	}

	stop := make(chan struct{})
	var workers sync.WaitGroup // 写者 + 查询者
	var statsWG sync.WaitGroup

	// 写者：反复删除并重新插入一批 ID，造成持续的桶变更。
	for w := 0; w < 6; w++ {
		workers.Add(1)
		go func(w int) {
			defer workers.Done()
			r := newRNG(int64(900 + w))
			for it := 0; it < 300; it++ {
				id := int64(r.Uint64()%total) + 1
				if it%2 == 0 {
					_, _ = ix.Delete(id)
				} else {
					_ = ix.Insert(id, vecs[id-1])
				}
			}
		}(w)
	}

	// 读者：持续查询，返回结果必须有序且所有 ID 均为当前存活向量。
	for rd := 0; rd < 8; rd++ {
		workers.Add(1)
		go func(rd int) {
			defer workers.Done()
			r := newRNG(int64(100 + rd))
			for it := 0; it < 800; it++ {
				q := normalized(vecs[r.Uint64()%total])
				sc := SearchConfig{Radius: 2}
				// 在单把读锁区间内检索并校验，针对同一份状态快照，
				// 真正检验“查询读不到半更新状态”。
				ix.mu.RLock()
				got, _, err := ix.searchLocked(q, 10, sc)
				if err != nil {
					ix.mu.RUnlock()
					t.Error(err)
					return
				}
				for j := 1; j < len(got); j++ {
					if got[j-1].Distance > got[j].Distance+1e-12 ||
						(got[j-1].Distance == got[j].Distance && got[j-1].ID > got[j].ID) {
						ix.mu.RUnlock()
						t.Error("result not sorted")
						return
					}
				}
				for _, g := range got {
					v, ok := ix.vectors[g.ID]
					if !ok {
						ix.mu.RUnlock()
						t.Error("search returned a deleted/unknown id")
						return
					}
					if _, ok := ix.sigs[g.ID]; !ok || v == nil {
						ix.mu.RUnlock()
						t.Error("half-updated state observed")
						return
					}
				}
				ix.mu.RUnlock()
			}
		}(rd)
	}

	// 统计读者：并发读取 Stats/Len。
	statsWG.Add(1)
	go func() {
		defer statsWG.Done()
		for {
			select {
			case <-stop:
				return
			default:
				_ = ix.Stats()
				_ = ix.Len()
			}
		}
	}()

	workers.Wait()
	close(stop)
	statsWG.Wait()
}

// TestPostConcurrencyConsistency 在并发结束后校验最终状态结构一致。
func TestPostConcurrencyConsistency(t *testing.T) {
	cfg := DefaultConfig(10)
	cfg.Seed = 3
	ix, _ := New(cfg)
	const total = 400
	ids, vecs := genData(total, 10, 44)
	if err := ix.InsertMany(ids, vecs); err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	var dmu sync.Mutex
	deleted := map[int64]bool{}
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			r := newRNG(int64(w + 1))
			for it := 0; it < 200; it++ {
				id := int64(r.Uint64()%total) + 1
				if r.Uint64()%2 == 0 {
					existed, err := ix.Delete(id)
					if err == nil && existed {
						dmu.Lock()
						deleted[id] = true
						dmu.Unlock()
					}
				} else {
					_ = ix.Insert(id, vecs[id-1])
					dmu.Lock()
					delete(deleted, id)
					dmu.Unlock()
				}
			}
		}(w)
	}
	wg.Wait()

	st := ix.Stats()
	if st.NumVectors != ix.Len() {
		t.Fatal("stats count mismatch")
	}
	entries := 0
	for _, band := range ix.buckets {
		for _, post := range band {
			seen := map[int64]bool{}
			for _, id := range post {
				if seen[id] {
					t.Fatal("duplicate id in posting")
				}
				seen[id] = true
				if _, ok := ix.vectors[id]; !ok {
					t.Fatal("posting references missing vector")
				}
				entries++
			}
		}
	}
	if entries != st.NumVectors*st.NumBands {
		t.Fatalf("posting entries %d != vectors*bands %d", entries, st.NumVectors*st.NumBands)
	}
	for id := range deleted {
		if _, ok := ix.vectors[id]; ok {
			t.Fatalf("id %d marked deleted but still present", id)
		}
	}
}

package ann

import (
	"reflect"
	"testing"
)

func TestDeleteRemovesFromAllResults(t *testing.T) {
	cfg := DefaultConfig(10)
	cfg.Seed = 11
	ix, _ := New(cfg)
	const n, k = 200, 5
	ids, vecs := genData(n, 10, 5)
	if err := ix.InsertMany(ids, vecs); err != nil {
		t.Fatal(err)
	}

	// 删除偶数 ID
	var deleted []int64
	for _, id := range ids {
		if id%2 == 0 {
			existed, err := ix.Delete(id)
			if err != nil || !existed {
				t.Fatalf("delete %d: existed=%v err=%v", id, existed, err)
			}
			deleted = append(deleted, id)
		}
	}
	if ix.Len() != n/2 {
		t.Fatalf("Len=%d", ix.Len())
	}
	// 重复删除不存在的 ID 返回 false
	if ok, _ := ix.Delete(deleted[0]); ok {
		t.Fatal("delete missing id should report false")
	}

	// 用全半径搜索保证每个存活向量都能被命中；结果里不能有已删 ID。
	for i, q := range vecs {
		got, _, err := ix.Search(q, k, SearchConfig{Radius: cfg.BandWidth})
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range got {
			if r.ID%2 == 0 {
				t.Fatalf("deleted id %d returned in query %d", r.ID, i)
			}
		}
	}

	// BruteForce 也不能包含被删 ID
	bf, err := ix.BruteForce(vecs[0], n)
	if err != nil {
		t.Fatal(err)
	}
	if len(bf) != n/2 {
		t.Fatalf("bruteforce count %d", len(bf))
	}
}

func TestRepeatedInsertDeleteNoBloat(t *testing.T) {
	cfg := DefaultConfig(8)
	ix, _ := New(cfg)
	const n = 100
	ids, vecs := genData(n, 8, 88)
	if err := ix.InsertMany(ids, vecs); err != nil {
		t.Fatal(err)
	}
	base := ix.Stats()

	// 反复删除全部再插回全部，多轮后倒排项数不得增长。
	for round := 0; round < 20; round++ {
		for _, id := range ids {
			if _, err := ix.Delete(id); err != nil {
				t.Fatal(err)
			}
		}
		if ix.Len() != 0 || ix.Stats().NumPostingEntries != 0 || ix.Stats().NumBuckets != 0 {
			t.Fatalf("after full delete: %+v", ix.Stats())
		}
		for i, id := range ids {
			if err := ix.Insert(id, vecs[i]); err != nil {
				t.Fatal(err)
			}
		}
	}
	st := ix.Stats()
	if st.NumPostingEntries != base.NumPostingEntries {
		t.Fatalf("posting entries grew: %d -> %d", base.NumPostingEntries, st.NumPostingEntries)
	}
	if st.NumVectors != n || st.NumBuckets != base.NumBuckets {
		t.Fatalf("unexpected stats: %+v base buckets=%d", st, base.NumBuckets)
	}

	// 结果仍与全新构建一致
	fresh, _ := New(cfg)
	if err := fresh.InsertMany(ids, vecs); err != nil {
		t.Fatal(err)
	}
	for _, q := range vecs {
		a, _, _ := ix.Search(q, 10, SearchConfig{Radius: 2})
		b, _, _ := fresh.Search(q, 10, SearchConfig{Radius: 2})
		if !reflect.DeepEqual(a, b) {
			t.Fatal("results diverged after churn")
		}
	}
}

func TestRebuildReclaimsAndConsistent(t *testing.T) {
	cfg := DefaultConfig(12)
	cfg.Seed = 5
	ix, _ := New(cfg)
	const n = 300
	ids, vecs := genData(n, 12, 17)
	if err := ix.InsertMany(ids, vecs); err != nil {
		t.Fatal(err)
	}
	// 删除 60%
	for i, id := range ids {
		if i%5 != 0 { // 保留约 1/5
			if _, err := ix.Delete(id); err != nil {
				t.Fatal(err)
			}
		}
	}

	before := ix.Stats()
	if err := ix.Rebuild(); err != nil {
		t.Fatal(err)
	}
	after := ix.Stats()
	if after.NumVectors != before.NumVectors {
		t.Fatal("rebuild changed vector count")
	}
	if after.NumPostingEntries != before.NumPostingEntries {
		t.Fatalf("rebuild should keep posting entries equal: %d vs %d",
			after.NumPostingEntries, before.NumPostingEntries)
	}

	// 与全新只含存活数据的索引结果一致
	var keepIDs []int64
	var keepVecs [][]float64
	for i, id := range ids {
		if i%5 == 0 {
			keepIDs = append(keepIDs, id)
			keepVecs = append(keepVecs, vecs[i])
		}
	}
	fresh, _ := New(cfg)
	if err := fresh.InsertMany(keepIDs, keepVecs); err != nil {
		t.Fatal(err)
	}
	for _, q := range vecs {
		a, _, _ := ix.Search(q, 10, SearchConfig{Radius: 3})
		b, _, _ := fresh.Search(q, 10, SearchConfig{Radius: 3})
		if !reflect.DeepEqual(a, b) {
			t.Fatal("rebuilt index results differ from fresh build")
		}
	}
}

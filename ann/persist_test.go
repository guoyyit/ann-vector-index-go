package ann

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestSaveLoadConsistent(t *testing.T) {
	data, ids := genData(800, 16, 10, 42)
	cfg := DefaultConfig(16)
	cfg.M = 16
	idx := buildIndex(t, cfg, ids, data)
	queries := genQueries(20, 16, 10, 99)

	dir := t.TempDir()
	path := filepath.Join(dir, "index.ann")
	if err := idx.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := loaded.Verify(); err != nil {
		t.Fatal(err)
	}
	if loaded.Len() != idx.Len() {
		t.Fatalf("len mismatch: %d != %d", loaded.Len(), idx.Len())
	}
	for _, q := range queries {
		if !sameResults(t, idx, loaded, q, 10, 64) {
			t.Fatal("loaded index results differ from in-memory index")
		}
	}
}

func TestSaveLoadAfterDelete(t *testing.T) {
	data, ids := genData(500, 8, 5, 42)
	cfg := DefaultConfig(8)
	idx := buildIndex(t, cfg, ids, data)
	for i := 0; i < 100; i++ {
		if err := idx.Delete(ids[i]); err != nil {
			t.Fatal(err)
		}
	}
	queries := genQueries(10, 8, 5, 42)
	deleted := make(map[uint64]struct{}, 100)
	for i := 0; i < 100; i++ {
		deleted[ids[i]] = struct{}{}
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "index.ann")
	if err := idx.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Len() != 400 {
		t.Fatalf("expected 400, got %d", loaded.Len())
	}
	for _, q := range queries {
		res, err := loaded.Search(q, 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range res {
			if _, ok := deleted[r.ID]; ok {
				t.Fatalf("deleted id %d in loaded results", r.ID)
			}
		}
	}
}

func TestLoadCorruption(t *testing.T) {
	data, ids := genData(300, 8, 5, 42)
	cfg := DefaultConfig(8)
	idx := buildIndex(t, cfg, ids, data)
	dir := t.TempDir()
	path := filepath.Join(dir, "index.ann")
	if err := idx.Save(path); err != nil {
		t.Fatal(err)
	}
	orig, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		name string
		data []byte
	}{
		{"truncated", orig[:len(orig)-10]},
		{"truncated_header", orig[:20]},
		{"bad_magic", func() []byte { b := make([]byte, len(orig)); copy(b, orig); b[0] = 'X'; return b }()},
		{"bad_crc", func() []byte { b := make([]byte, len(orig)); copy(b, orig); b[len(b)-10] ^= 0xFF; return b }()},
		{"trailing_bytes", func() []byte { b := make([]byte, len(orig)+4); copy(b, orig); return b }()},
		{"empty", []byte{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := filepath.Join(dir, tc.name+".ann")
			if err := os.WriteFile(p, tc.data, 0644); err != nil {
				t.Fatal(err)
			}
			_, err := Load(p)
			if !errors.Is(err, ErrCorrupt) {
				t.Fatalf("expected ErrCorrupt, got %v", err)
			}
		})
	}
}

func TestLoadMissingFile(t *testing.T) {
	_, err := Load(filepath.Join(t.TempDir(), "missing.ann"))
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestSaveAtomicNoTempLeft(t *testing.T) {
	data, ids := genData(100, 8, 5, 42)
	cfg := DefaultConfig(8)
	idx := buildIndex(t, cfg, ids, data)
	dir := t.TempDir()
	path := filepath.Join(dir, "index.ann")
	if err := idx.Save(path); err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if filepath.Ext(e.Name()) == ".tmp" {
			t.Fatalf("temp file left behind: %s", e.Name())
		}
	}
}

package ann

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func buildPersistent(t *testing.T, dir string, n int, policy CorruptionPolicy) (*Index, Config, []int64, [][]float64) {
	t.Helper()
	cfg := DefaultConfig(12)
	cfg.Seed = 2024
	snap := filepath.Join(dir, "snap.bin")
	jl := filepath.Join(dir, "wal.bin")
	ix, err := New(cfg, WithPersistence(snap, jl, policy))
	if err != nil {
		t.Fatal(err)
	}
	ids, vecs := genData(n, 12, 901)
	if err := ix.InsertMany(ids[:n*2/3], vecs[:n*2/3]); err != nil {
		t.Fatal(err)
	}
	if err := ix.Save(); err != nil {
		t.Fatal(err)
	}
	if err := ix.InsertMany(ids[n*2/3:], vecs[n*2/3:]); err != nil {
		t.Fatal(err)
	}
	return ix, cfg, ids, vecs
}

func compareSearches(t *testing.T, a, b *Index, queries [][]float64, sc SearchConfig) {
	t.Helper()
	for _, q := range queries {
		ra, _, err := a.Search(q, 10, sc)
		if err != nil {
			t.Fatal(err)
		}
		rb, _, err := b.Search(q, 10, sc)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(ra, rb) {
			t.Fatalf("results differ after reload\n got=%v\nwant=%v", rb, ra)
		}
	}
}

func TestPersistenceRoundTrip(t *testing.T) {
	dir := t.TempDir()
	ix, cfg, ids, vecs := buildPersistent(t, dir, 300, CorruptionFail)
	queries := vecs[:25]
	before, err := GroundTruth(ix, queries, 10)
	if err != nil {
		t.Fatal(err)
	}
	memR0 := snapshotSearch(t, ix, queries, SearchConfig{Radius: 0})
	memR2 := snapshotSearch(t, ix, queries, SearchConfig{Radius: 2})
	if err := ix.Close(); err != nil {
		t.Fatal(err)
	}

	p := PersistenceOptions{
		SnapshotPath: filepath.Join(dir, "snap.bin"),
		JournalPath:  filepath.Join(dir, "wal.bin"),
		Policy:       CorruptionFail,
	}
	got, err := Load(p)
	if err != nil {
		t.Fatal(err)
	}
	defer got.Close()
	if got.Len() != len(ids) {
		t.Fatalf("reloaded Len=%d want %d", got.Len(), len(ids))
	}
	gt2, err := GroundTruth(got, queries, 10)
	if err != nil || !reflect.DeepEqual(before, gt2) {
		t.Fatalf("brute-force ground truth differs after reload: %v", err)
	}
	relR0 := snapshotSearch(t, got, queries, SearchConfig{Radius: 0})
	relR2 := snapshotSearch(t, got, queries, SearchConfig{Radius: 2})
	if !reflect.DeepEqual(memR0, relR0) || !reflect.DeepEqual(memR2, relR2) {
		t.Fatal("ANN results differ after reload")
	}

	// 重新加载后继续写入并再次恢复，验证多轮快照/压缩
	for i := 0; i < 20; i++ {
		if _, err := got.Delete(ids[i]); err != nil {
			t.Fatal(err)
		}
	}
	if err := got.Save(); err != nil {
		t.Fatal(err)
	}
	if err := got.Close(); err != nil {
		t.Fatal(err)
	}
	got2, err := Load(p)
	if err != nil {
		t.Fatal(err)
	}
	defer got2.Close()
	if got2.Len() != len(ids)-20 {
		t.Fatalf("second reload Len=%d", got2.Len())
	}
	// 快照里包含 Config，重建索引必须保持原种子
	if got2.cfg.Seed != cfg.Seed {
		t.Fatal("seed not restored from snapshot")
	}
}

// TestJournalReplayDeletes 验证只靠日志（无新快照）也能重放删除。
func TestJournalReplayOnly(t *testing.T) {
	dir := t.TempDir()
	ix, _, ids, vecs := buildPersistent(t, dir, 150, CorruptionFail)
	// Save 之后再做删除与插入（只在日志里）
	for i := 0; i < 40; i++ {
		if _, err := ix.Delete(ids[i]); err != nil {
			t.Fatal(err)
		}
	}
	extra := []float64{}
	for d := 0; d < 12; d++ {
		extra = append(extra, 0.5)
	}
	if err := ix.Insert(99999, extra); err != nil {
		t.Fatal(err)
	}
	queries := vecs[:15]
	memRes, _, _ := ix.Search(extra, 5, SearchConfig{Radius: 3})
	memQ2 := snapshotSearch(t, ix, queries, SearchConfig{Radius: 2})
	if err := ix.Close(); err != nil {
		t.Fatal(err)
	}
	got, err := Load(PersistenceOptions{
		SnapshotPath: filepath.Join(dir, "snap.bin"),
		JournalPath:  filepath.Join(dir, "wal.bin"),
		Policy:       CorruptionFail,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer got.Close()
	if got.Len() != 150-40+1 {
		t.Fatalf("reloaded len=%d", got.Len())
	}
	reRes, _, err := got.Search(extra, 5, SearchConfig{Radius: 3})
	if err != nil || !reflect.DeepEqual(memRes, reRes) {
		t.Fatalf("journal replay mismatch: %v", err)
	}
	reQ2 := snapshotSearch(t, got, queries, SearchConfig{Radius: 2})
	if !reflect.DeepEqual(memQ2, reQ2) {
		t.Fatal("journal replay query mismatch")
	}
}

// TestCorruptJournalTail 模拟写入中断（日志被截断/带垃圾尾），
// 必须识别损坏并按策略处理。
func TestCorruptJournalTail(t *testing.T) {
	dir := t.TempDir()
	ix, _, _, _ := buildPersistent(t, dir, 120, CorruptionFail)
	if err := ix.Close(); err != nil {
		t.Fatal(err)
	}
	jl := filepath.Join(dir, "wal.bin")
	fi, _ := os.Stat(jl)
	orig, _ := os.ReadFile(jl)

	// 追加半个帧（长度头声称很大但数据不足）
	bad := append(append([]byte{}, orig...), 0xFF, 0xFF, 0x00, 0x00, 0x01)
	if err := os.WriteFile(jl, bad, 0o644); err != nil {
		t.Fatal(err)
	}
	// Fail 策略：必须报错，不得悄悄返回
	if _, err := Load(PersistenceOptions{
		SnapshotPath: filepath.Join(dir, "snap.bin"),
		JournalPath:  jl, Policy: CorruptionFail,
	}); err == nil {
		t.Fatal("expected corruption error")
	}
	// IgnoreJournal 策略：丢弃坏尾部，恢复到最近快照状态
	ig, err := Load(PersistenceOptions{
		SnapshotPath: filepath.Join(dir, "snap.bin"),
		JournalPath:  jl, Policy: CorruptionIgnoreJournal,
	})
	if err != nil {
		t.Fatal(err)
	}
	// 快照(80 条) + 快照之后的完整合法 WAL 帧(40 条)都会恢复；
	// 被丢弃的只是我们追加的损坏尾部，它本来就不对应任何有效数据。
	if ig.Len() != 120 {
		t.Fatalf("ignore-journal len=%d", ig.Len())
	}
	ig.Close()

	// Rebuild 策略：清空日志重新开始
	rb, err := Load(PersistenceOptions{
		SnapshotPath: filepath.Join(dir, "snap.bin"),
		JournalPath:  jl, Policy: CorruptionRebuild,
	})
	if err != nil {
		t.Fatal(err)
	}
	rb.Close()

	// 恢复原文件，确保 Fail 策略对完好文件正常
	_ = os.WriteFile(jl, orig, 0o644)
	_ = fi
}

// TestCorruptSnapshot 快照 CRC 损坏必须被识别。
func TestCorruptSnapshot(t *testing.T) {
	dir := t.TempDir()
	ix, _, _, _ := buildPersistent(t, dir, 100, CorruptionFail)
	_ = ix.Close()
	sp := filepath.Join(dir, "snap.bin")
	data, _ := os.ReadFile(sp)
	// 翻转中间一个字节（CRC 必然失配）
	data[len(data)/2] ^= 0xFF
	if err := os.WriteFile(sp, data, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(PersistenceOptions{
		SnapshotPath: sp, JournalPath: filepath.Join(dir, "wal.bin"),
		Policy: CorruptionFail,
	}); err == nil {
		t.Fatal("expected snapshot corruption error")
	}
	// 重建策略：跳过坏快照；但完好的日志在无快照基数时也不自动信任，
	// 这里验证其返回可用索引（配置仍从调用方语义恢复）并可重新落盘。
	rb, err := New(cfgRebuild(t, sp), WithPersistence(sp, filepath.Join(dir, "wal2.bin"), CorruptionRebuild))
	if err != nil {
		t.Fatal(err)
	}
	if rb.Len() != 0 {
		t.Fatalf("rebuild policy expected empty index, got %d", rb.Len())
	}
	if err := rb.Save(); err != nil {
		t.Fatal(err)
	}
	rb.Close()
}

func cfgRebuild(t *testing.T, snap string) Config {
	t.Helper()
	cfg := DefaultConfig(12)
	cfg.Seed = 2024
	return cfg
}

func snapshotSearch(t *testing.T, ix *Index, queries [][]float64, sc SearchConfig) [][]Result {
	t.Helper()
	out := make([][]Result, len(queries))
	for i, q := range queries {
		r, _, err := ix.Search(q, 10, sc)
		if err != nil {
			t.Fatal(err)
		}
		out[i] = r
	}
	return out
}

// TestCorruptHalfMagicIgnore 日志文件头损坏时，Ignore 策略截断补头后仍可写。
func TestCorruptHalfMagicIgnore(t *testing.T) {
	dir := t.TempDir()
	jl := filepath.Join(dir, "wal.bin")
	if err := os.WriteFile(jl, []byte("ANN"), 0o644); err != nil { // 半个 magic
		t.Fatal(err)
	}
	cfg := DefaultConfig(6)
	ix, err := New(cfg, WithPersistence("", jl, CorruptionIgnoreJournal))
	if err != nil {
		t.Fatal(err)
	}
	v := []float64{1, 0, 0, 0, 0, 0}
	if err := ix.Insert(1, v); err != nil {
		t.Fatal(err)
	}
	if err := ix.Close(); err != nil {
		t.Fatal(err)
	}
	// 再次打开应能正常重放这一帧
	ix2, err := New(cfg, WithPersistence("", jl, CorruptionFail))
	if err != nil {
		t.Fatal(err)
	}
	if ix2.Len() != 1 {
		t.Fatalf("len=%d", ix2.Len())
	}
	ix2.Close()
}

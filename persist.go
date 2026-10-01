package ann

import (
	"encoding/binary"
	"hash/crc32"
	"os"
	"path/filepath"
	"sort"
)

const (
	snapMagic   = "ANNSNP01"
	snapVersion = byte(1)
)

// PersistenceOptions 控制持久化路径与损坏处理策略。
type PersistenceOptions struct {
	// SnapshotPath 快照文件路径（留空则不做快照持久化）。
	SnapshotPath string
	// JournalPath 预写日志路径（留空则不启用日志）。
	JournalPath string
	// Policy 损坏处理策略。
	Policy CorruptionPolicy
}

// PersistenceOption 预留的持久化可选项（当前无需附加项）。
type PersistenceOption func(*PersistenceOptions)

// WithPersistence 启用持久化：New 时若快照/日志已存在会先恢复。
// 后续每次写操作同步落入预写日志，可调用 Save 做快照并压缩日志。
func WithPersistence(snapshotPath, journalPath string, policy CorruptionPolicy) Option {
	p := PersistenceOptions{SnapshotPath: snapshotPath, JournalPath: journalPath, Policy: policy}
	return func(ix *Index) {
		if err := openPersistence(ix, p); err != nil {
			panic(persistError{err})
		}
	}
}

// persistError 把 Option 内的错误延迟到 New 中返回。
type persistError struct{ err error }

func (e persistError) Error() string { return e.err.Error() }

// openPersistence 载入已有快照并重放日志，随后挂接日志句柄。
func openPersistence(ix *Index, p PersistenceOptions) error {
	ix.popts = p
	if p.SnapshotPath != "" {
		if err := loadSnapshotInto(p.SnapshotPath, p.Policy, ix); err != nil {
			return err
		}
	}
	if p.JournalPath != "" {
		j, err := openJournal(p.JournalPath, p.Policy, ix.applyReplay)
		if err != nil {
			return err
		}
		ix.journal = j
	}
	return nil
}

// Load 从快照（其后的预写日志）恢复索引。返回的索引与落盘前内存中
// 的索引给出完全一致的检索结果。
func Load(p PersistenceOptions) (*Index, error) {
	if p.SnapshotPath == "" {
		return nil, ErrInvalidConfig
	}
	cfg, err := readSnapshotConfig(p.SnapshotPath)
	if err != nil {
		return nil, err
	}
	return New(cfg, func(ix *Index) {
		if err := openPersistence(ix, p); err != nil {
			panic(persistError{err})
		}
	})
}

// snapshot 序列化内容（不含尾部 CRC）。
func encodeSnapshot(ix *Index) []byte {
	var b []byte
	put := func(p []byte) { b = append(b, p...) }
	put([]byte(snapMagic))
	put([]byte{snapVersion})
	var u32 [4]byte
	var u64 [8]byte
	binary.LittleEndian.PutUint32(u32[:], uint32(ix.cfg.Dim))
	put(u32[:])
	binary.LittleEndian.PutUint32(u32[:], uint32(ix.cfg.Bits))
	put(u32[:])
	binary.LittleEndian.PutUint32(u32[:], uint32(ix.cfg.BandWidth))
	put(u32[:])
	put([]byte{byte(ix.cfg.Metric)})
	binary.LittleEndian.PutUint64(u64[:], uint64(ix.cfg.Seed))
	put(u64[:])

	ids := make([]int64, 0, len(ix.vectors))
	for id := range ix.vectors {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	binary.LittleEndian.PutUint32(u32[:], uint32(len(ids)))
	put(u32[:])
	for _, id := range ids {
		v := ix.vectors[id]
		binary.LittleEndian.PutUint64(u64[:], uint64(id))
		put(u64[:])
		binary.LittleEndian.PutUint32(u32[:], uint32(len(v)))
		put(u32[:])
		for _, x := range v {
			binary.LittleEndian.PutUint64(u64[:], float64Bits(x))
			put(u64[:])
		}
	}
	return b
}

// saveSnapshotLocked 原子写快照：先写临时文件并 fsync，再 rename，
// 最后 fsync 目录，保证任意时刻磁盘上要么是旧快照要么是新快照。
func saveSnapshotLocked(path string, ix *Index) error {
	body := encodeSnapshot(ix)
	var crc [4]byte
	binary.LittleEndian.PutUint32(crc[:], crc32.ChecksumIEEE(body))
	tmp := path + ".tmp"
	f, err := os.OpenFile(tmp, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	if _, err := f.Write(body); err != nil {
		f.Close()
		return err
	}
	if _, err := f.Write(crc[:]); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		return err
	}
	return syncDir(filepath.Dir(path))
}

func syncDir(dir string) error {
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer d.Close()
	return d.Sync()
}

// parseSnapshot 校验 CRC 并解析快照。
func parseSnapshot(data []byte) (Config, []walFrame, error) {
	if len(data) < len(snapMagic)+1+20+4 {
		return Config{}, nil, ErrCorrupt
	}
	if string(data[:len(snapMagic)]) != snapMagic {
		return Config{}, nil, ErrCorrupt
	}
	tail := len(data) - 4
	want := binary.LittleEndian.Uint32(data[tail:])
	if crc32.ChecksumIEEE(data[:tail]) != want {
		return Config{}, nil, ErrCorrupt
	}
	p := len(snapMagic)
	if data[p] != snapVersion {
		return Config{}, nil, ErrCorrupt
	}
	p++
	dim := int(binary.LittleEndian.Uint32(data[p:]))
	p += 4
	bitsN := int(binary.LittleEndian.Uint32(data[p:]))
	p += 4
	bw := int(binary.LittleEndian.Uint32(data[p:]))
	p += 4
	metric := Metric(data[p])
	p++
	seed := int64(binary.LittleEndian.Uint64(data[p:]))
	p += 8
	cfg := Config{Dim: dim, Bits: bitsN, BandWidth: bw, Metric: metric, Seed: seed}
	if err := cfg.Validate(); err != nil {
		return Config{}, nil, ErrCorrupt
	}
	n := int(binary.LittleEndian.Uint32(data[p:]))
	p += 4
	frames := make([]walFrame, 0, n)
	for i := 0; i < n; i++ {
		if p+12 > tail {
			return Config{}, nil, ErrCorrupt
		}
		id := int64(binary.LittleEndian.Uint64(data[p:]))
		p += 8
		ndim := int(binary.LittleEndian.Uint32(data[p:]))
		p += 4
		if ndim != dim || p+ndim*8 > tail {
			return Config{}, nil, ErrCorrupt
		}
		vec := make([]float64, ndim)
		for d := range vec {
			vec[d] = float64FromBits(binary.LittleEndian.Uint64(data[p:]))
			p += 8
		}
		frames = append(frames, walFrame{op: walOpInsert, id: id, vec: vec})
	}
	if p != tail {
		return Config{}, nil, ErrCorrupt
	}
	return cfg, frames, nil
}

// readSnapshotConfig 只读取快照中的配置（供 Load 在创建索引前调用）。
func readSnapshotConfig(path string) (Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Config{}, err
	}
	cfg, _, err := parseSnapshot(data)
	return cfg, err
}

// loadSnapshotInto 校验快照、比对配置并把数据载入索引。
func loadSnapshotInto(path string, policy CorruptionPolicy, ix *Index) error {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // 全新索引，尚无快照
		}
		return err
	}
	cfg, frames, perr := parseSnapshot(data)
	if perr != nil {
		if policy == CorruptionRebuild {
			return nil
		}
		return perr
	}
	if cfg != ix.cfg {
		return ErrCorrupt // 快照与索引配置不一致
	}
	for _, f := range frames {
		if err := ix.applyReplay(f); err != nil {
			return err
		}
	}
	return nil
}

// Save 写快照并压缩预写日志。
func (ix *Index) Save() error {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	if err := ix.checkClosed(); err != nil {
		return err
	}
	if ix.popts.SnapshotPath == "" {
		return ErrInvalidConfig
	}
	if err := saveSnapshotLocked(ix.popts.SnapshotPath, ix); err != nil {
		return err
	}
	if ix.journal != nil {
		return ix.journal.truncate()
	}
	return nil
}

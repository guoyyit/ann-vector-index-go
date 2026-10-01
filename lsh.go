package ann

import (
	"sort"
	"sync"
	"time"
)

// Index 是线程安全、确定性、可持久化的近似最近邻索引。
//
// 一致性保证：
//   - 超平面仅由 Config.Seed 决定；桶成员关系只取决于向量本身，
//     因此插入顺序、批次划分、重复插入都不会改变最终检索结果。
//   - 所有写操作持有 mu 的写锁，所有读操作持有读锁，查询期间到达的
//     写入会在锁外等待，查询永远看到一个完整一致的状态快照。
type Index struct {
	mu      sync.RWMutex
	cfg     Config
	planes  [][]float64 // 长度 Bits，每个是 Dim 维单位无关的法向量
	vectors map[int64][]float64
	sigs    map[int64][]uint64   // 每个向量的 Bits 位签名
	buckets []map[uint64][]int64 // 按 band 组织的倒排表
	journal *journal
	popts   PersistenceOptions
	closed  bool
}

// Option 用于配置 New。
type Option func(*Index)

// New 创建新索引。
func New(cfg Config, opts ...Option) (*Index, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	r := newRNG(cfg.Seed)
	planes := make([][]float64, cfg.Bits)
	for i := range planes {
		p := make([]float64, cfg.Dim)
		for d := range p {
			p[d] = r.Normal()
		}
		planes[i] = p // 超平面不需要归一化：sign(dot) 不受缩放影响
	}
	numBands := cfg.Bits / cfg.BandWidth
	ix := &Index{
		cfg:     cfg,
		planes:  planes,
		vectors: make(map[int64][]float64),
		sigs:    make(map[int64][]uint64),
		buckets: make([]map[uint64][]int64, numBands),
	}
	for i := range ix.buckets {
		ix.buckets[i] = make(map[uint64][]int64)
	}
	for _, o := range opts {
		if err := runOption(o, ix); err != nil {
			return nil, err
		}
	}
	return ix, nil
}

// runOption 执行 Option，并把持久化初始化错误转换为普通返回值。
func runOption(o Option, ix *Index) (err error) {
	defer func() {
		if r := recover(); r != nil {
			if pe, ok := r.(persistError); ok {
				err = pe.err
				return
			}
			panic(r)
		}
	}()
	o(ix)
	return nil
}

// Close 关闭索引（刷新预写日志并释放文件句柄）。
func (ix *Index) Close() error {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	if ix.closed {
		return ErrClosed
	}
	ix.closed = true
	if ix.journal != nil {
		return ix.journal.Close()
	}
	return nil
}

func (ix *Index) checkClosed() error {
	if ix.closed {
		return ErrClosed
	}
	return nil
}

// signature 计算向量的 Bits 位签名：第 j 位为 1 当且仅当 dot(v, plane_j) >= 0。
func (ix *Index) signature(v []float64) []uint64 {
	words := (ix.cfg.Bits + 63) / 64
	sig := make([]uint64, words)
	for j, p := range ix.planes {
		var s float64
		for d := range v {
			s += v[d] * p[d]
		}
		if s >= 0 {
			sig[j/64] |= 1 << uint(j%64)
		}
	}
	return sig
}

// bandKey 提取第 b 个 band 的桶键（BandWidth <= 64）。
func bandKey(sig []uint64, b, w int) uint64 {
	var key uint64
	for k := 0; k < w; k++ {
		bit := b*w + k
		if sig[bit/64]&(1<<uint(bit%64)) != 0 {
			key |= 1 << uint(k)
		}
	}
	return key
}

// Insert 插入或更新一个向量（upsert，按 ID 幂等）。
// 重复插入相同向量得到完全相同的索引状态。
func (ix *Index) Insert(id int64, vec []float64) error {
	if len(vec) == 0 {
		return ErrEmptyVector
	}
	if len(vec) != ix.cfg.Dim {
		return ErrDimensionMismatch
	}
	nv := normalized(vec)
	ix.mu.Lock()
	defer ix.mu.Unlock()
	if err := ix.checkClosed(); err != nil {
		return err
	}
	ix.upsertLocked(id, nv)
	if ix.journal != nil {
		// 日志记录实际入库的归一化向量，保证重放结果与内存状态逐字节一致，
		// 也与快照路径完全相同（不会被二次归一化引入浮点偏差）。
		if err := ix.journal.append(walFrame{op: walOpInsert, id: id, vec: nv}); err != nil {
			return err
		}
	}
	return nil
}

// InsertMany 批量插入；与逐条调用 Insert 的最终状态完全一致。
func (ix *Index) InsertMany(ids []int64, vecs [][]float64) error {
	if len(ids) != len(vecs) {
		return ErrInvalidConfig
	}
	for i, v := range vecs {
		if err := ix.Insert(ids[i], v); err != nil {
			return err
		}
	}
	return nil
}

// upsertLocked 在已持有写锁时写入。若 ID 已存在先彻底移除旧倒排项，
// 保证同一 ID 在每个桶里恰好出现一次（重复插入不膨胀）。
func (ix *Index) upsertLocked(id int64, nv []float64) {
	if _, ok := ix.vectors[id]; ok {
		ix.removeFromBucketsLocked(id)
	}
	sig := ix.signature(nv)
	ix.vectors[id] = nv
	ix.sigs[id] = sig
	w := ix.cfg.BandWidth
	for b := range ix.buckets {
		key := bandKey(sig, b, w)
		post := ix.buckets[b][key]
		// 防御性去重（正常路径 removeFromBucketsLocked 已保证不存在）。
		found := false
		for _, x := range post {
			if x == id {
				found = true
				break
			}
		}
		if !found {
			post = append(post, id)
			sort.Slice(post, func(i, j int) bool { return post[i] < post[j] })
		}
		ix.buckets[b][key] = post
	}
}

// removeFromBucketsLocked 从所有 band 的倒排表中删除该 ID（就地压缩，
// 不留墓碑，反复插入/删除不会使倒排切片无界增长）。
func (ix *Index) removeFromBucketsLocked(id int64) {
	old, ok := ix.sigs[id]
	if !ok {
		return
	}
	w := ix.cfg.BandWidth
	for b := range ix.buckets {
		key := bandKey(old, b, w)
		post := ix.buckets[b][key]
		out := post[:0]
		for _, x := range post {
			if x != id {
				out = append(out, x)
			}
		}
		if len(out) == 0 {
			delete(ix.buckets[b], key)
		} else {
			ix.buckets[b][key] = out
		}
	}
	delete(ix.sigs, id)
	delete(ix.vectors, id)
}

// applyReplay 回放一条预写日志（不再二次写日志）。用于启动恢复，
// 幂等：重复回放与首次回放得到相同状态。
func (ix *Index) applyReplay(f walFrame) error {
	switch f.op {
	case walOpInsert:
		if len(f.vec) != ix.cfg.Dim {
			return ErrCorrupt
		}
		// 日志/快照中存的就是入库归一化向量，直接装入，避免二次归一化
		// 带来 ulp 级差异，保证恢复后距离与内存逐位一致。
		ix.upsertLocked(f.id, f.vec)
	case walOpDelete:
		ix.removeFromBucketsLocked(f.id)
	default:
		return ErrCorrupt
	}
	return nil
}

// Delete 删除一个向量；返回 true 表示此前存在。删除后该 ID 不会再
// 出现在任何检索结果中。
func (ix *Index) Delete(id int64) (bool, error) {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	if err := ix.checkClosed(); err != nil {
		return false, err
	}
	if _, ok := ix.vectors[id]; !ok {
		return false, nil
	}
	ix.removeFromBucketsLocked(id)
	if ix.journal != nil {
		if err := ix.journal.append(walFrame{op: walOpDelete, id: id}); err != nil {
			return true, err
		}
	}
	return true, nil
}

// Search 近似检索。整个候选收集与重排都在读锁内完成，看到的是
// 调用瞬间的一致快照；Stats 描述本次访问规模。
func (ix *Index) Search(query []float64, k int, sc SearchConfig) ([]Result, SearchStats, error) {
	if k <= 0 {
		return nil, SearchStats{}, ErrK
	}
	if len(query) != ix.cfg.Dim {
		return nil, SearchStats{}, ErrDimensionMismatch
	}
	sc, err := sc.validate(len(ix.buckets), ix.cfg.BandWidth)
	if err != nil {
		return nil, SearchStats{}, err
	}
	q := normalized(query)

	ix.mu.RLock()
	defer ix.mu.RUnlock()
	if ix.closed {
		return nil, SearchStats{}, ErrClosed
	}
	return ix.searchLocked(q, k, sc)
}

// searchLocked 在调用方已持有 ix.mu（读锁或写锁）时执行检索。
// 整段在一次加锁区间内完成，因此看到的是严格一致的单一状态快照。
func (ix *Index) searchLocked(q []float64, k int, sc SearchConfig) ([]Result, SearchStats, error) {
	start := time.Now()
	st := SearchStats{Radius: sc.Radius}
	seen := make(map[int64]struct{})
	var cand []int64
	w := ix.cfg.BandWidth
	qsig := ix.signature(q)
	for b := range ix.buckets {
		st.BandsProbed++
		base := bandKey(qsig, b, w)
		for _, mask := range probeMasks(w, sc.Radius) {
			st.BucketsProbed++
			post := ix.buckets[b][base^mask]
			if len(post) == 0 {
				continue
			}
			st.BucketsHit++
			for _, id := range post {
				if _, dup := seen[id]; dup {
					continue
				}
				seen[id] = struct{}{}
				cand = append(cand, id)
			}
		}
		if sc.MaxCandidates > 0 && len(cand) >= sc.MaxCandidates {
			cand = cand[:sc.MaxCandidates]
			st.Truncated = true
			break
		}
	}
	// 按 ID 排序后重排：距离计算集合与顺序与插入顺序无关，
	// 统计量与 MaxCandidates 截断结果都可复现。
	sort.Slice(cand, func(i, j int) bool { return cand[i] < cand[j] })
	st.Candidates = len(cand)
	tk := newTopK(k)
	for _, id := range cand {
		d := cosineDistance(q, ix.vectors[id])
		st.Distances++
		tk.consider(id, d)
	}
	res := tk.results()
	st.Returned = len(res)
	st.DurationMicros = time.Since(start).Microseconds()
	return res, st, nil
}

// BruteForce 对当前存活向量做精确线性扫描，用于召回率对照。
func (ix *Index) BruteForce(query []float64, k int) ([]Result, error) {
	if k <= 0 {
		return nil, ErrK
	}
	if len(query) != ix.cfg.Dim {
		return nil, ErrDimensionMismatch
	}
	q := normalized(query)
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	if ix.closed {
		return nil, ErrClosed
	}
	return exactSearch(q, ix.vectors, k), nil
}

// Rebuild 全量重建倒排结构并做一次快照+日志压缩，回收删除/更新
// 累积的空间占用。重建结果与等价数据重新构建完全一致。
func (ix *Index) Rebuild() error {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	if err := ix.checkClosed(); err != nil {
		return err
	}
	return ix.rebuildLocked()
}

// rebuildLocked 在已持有写锁时重建。
func (ix *Index) rebuildLocked() error {
	vectors := ix.vectors
	nb := make([]map[uint64][]int64, len(ix.buckets))
	for i := range nb {
		nb[i] = make(map[uint64][]int64)
	}
	ids := make([]int64, 0, len(vectors))
	for id := range vectors {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	w := ix.cfg.BandWidth
	for _, id := range ids {
		sig := ix.sigs[id]
		for b := range nb {
			key := bandKey(sig, b, w)
			nb[b][key] = append(nb[b][key], id) // ids 已排序，posting 天然有序
		}
	}
	ix.buckets = nb
	if ix.popts.SnapshotPath != "" {
		if err := saveSnapshotLocked(ix.popts.SnapshotPath, ix); err != nil {
			return err
		}
		if ix.journal != nil {
			return ix.journal.truncate()
		}
	}
	return nil
}

// Stats 返回索引规模统计（快照值）。
func (ix *Index) Stats() Stats {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	var entries int
	for _, band := range ix.buckets {
		for _, post := range band {
			entries += len(post)
		}
	}
	return Stats{
		Dim:               ix.cfg.Dim,
		Bits:              ix.cfg.Bits,
		BandWidth:         ix.cfg.BandWidth,
		NumBands:          len(ix.buckets),
		NumVectors:        len(ix.vectors),
		NumBuckets:        countBuckets(ix.buckets),
		NumPostingEntries: entries,
		Journaling:        ix.journal != nil,
	}
}

func countBuckets(buckets []map[uint64][]int64) int {
	n := 0
	for _, b := range buckets {
		n += len(b)
	}
	return n
}

// Len 返回存活向量数量。
func (ix *Index) Len() int {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return len(ix.vectors)
}

// probeMasks 返回 band 内汉明距离 <= radius 的全部异或掩码，
// 按 (汉明重量, 掩码数值) 排序，保证探测顺序确定、可解释。
func probeMasks(width, radius int) []uint64 {
	var masks []uint64
	masks = append(masks, 0)
	for r := 1; r <= radius; r++ {
		var combos []uint64
		var gen func(start, left int, mask uint64)
		gen = func(start, left int, mask uint64) {
			if left == 0 {
				combos = append(combos, mask)
				return
			}
			for i := start; i <= width-left; i++ {
				gen(i+1, left-1, mask|1<<uint(i))
			}
		}
		gen(0, r, 0)
		sort.Slice(combos, func(i, j int) bool { return combos[i] < combos[j] })
		masks = append(masks, combos...)
	}
	return masks
}

// 避免未使用导入（bits 预留给后续 popcount 统计/校验）。

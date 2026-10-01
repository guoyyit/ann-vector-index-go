package ann

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"math"
	"os"
	"path/filepath"
	"sort"
)

const (
	persistMagic   = "ANNIDX01"
	persistVersion = uint32(1)
	headerSize     = 8 + 4*6 + 8 // magic + 6 uint32 fields + seed uint64
)

// Save writes the index to path atomically: data goes to a temp file in the
// same directory, is fsynced, then renamed over path. A CRC32 checksum
// covers every payload byte, so torn writes or bit rot are detected on load.
func (idx *Index) Save(path string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	// Rebuild if dirty so the serialized graph is canonical and references
	// only live nodes.
	if idx.dirty {
		idx.rebuildGraphLocked()
		idx.dirty = false
	}

	var buf bytes.Buffer
	buf.WriteString(persistMagic)
	hdr := []uint32{
		persistVersion,
		uint32(idx.cfg.Dim),
		uint32(idx.cfg.Metric),
		uint32(idx.cfg.M),
		uint32(idx.cfg.MaxLevel),
		uint32(idx.lenLocked()),
		uint32(len(idx.graph) - 1),
	}
	for _, v := range hdr {
		if err := binary.Write(&buf, binary.LittleEndian, v); err != nil {
			return err
		}
	}
	if err := binary.Write(&buf, binary.LittleEndian, idx.cfg.Seed); err != nil {
		return err
	}

	slots := make([]int, 0, len(idx.idMap))
	for slot, nd := range idx.nodes {
		if nd.alive {
			slots = append(slots, slot)
		}
	}
	sort.Slice(slots, func(i, j int) bool { return idx.nodes[slots[i]].id < idx.nodes[slots[j]].id })

	for _, slot := range slots {
		if err := binary.Write(&buf, binary.LittleEndian, idx.nodes[slot].id); err != nil {
			return err
		}
	}
	for _, slot := range slots {
		for _, v := range idx.nodes[slot].vec {
			if err := binary.Write(&buf, binary.LittleEndian, v); err != nil {
				return err
			}
		}
	}
	for _, slot := range slots {
		if err := binary.Write(&buf, binary.LittleEndian, uint32(idx.nodes[slot].level)); err != nil {
			return err
		}
	}
	for _, slot := range slots {
		for l := 0; l <= idx.nodes[slot].level; l++ {
			list := idx.graph[l][slot]
			if err := binary.Write(&buf, binary.LittleEndian, uint32(len(list))); err != nil {
				return err
			}
			for _, nb := range list {
				if err := binary.Write(&buf, binary.LittleEndian, idx.nodes[nb].id); err != nil {
					return err
				}
			}
		}
	}

	crc := crc32.ChecksumIEEE(buf.Bytes())
	if err := binary.Write(&buf, binary.LittleEndian, crc); err != nil {
		return err
	}

	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".ann-index-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	cleanup := true
	defer func() {
		if cleanup {
			os.Remove(tmpName)
		}
	}()
	if _, err := tmp.Write(buf.Bytes()); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpName, path); err != nil {
		return err
	}
	cleanup = false
	// fsync the directory so the rename itself is durable.
	if d, err := os.Open(dir); err == nil {
		_ = d.Sync()
		_ = d.Close()
	}
	return nil
}

// Load reads an index from path. Truncation, bad magic/version, checksum
// mismatch or any structural defect is reported as ErrCorrupt; no index is
// ever returned in a partially-loaded state.
func Load(path string) (*Index, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data) < headerSize+4 {
		return nil, fmt.Errorf("%w: file too short (%d bytes)", ErrCorrupt, len(data))
	}
	if string(data[:len(persistMagic)]) != persistMagic {
		return nil, fmt.Errorf("%w: bad magic", ErrCorrupt)
	}
	pos := len(persistMagic)
	rd := func() (uint32, error) {
		if pos+4 > len(data)-4 {
			return 0, fmt.Errorf("%w: truncated header", ErrCorrupt)
		}
		v := binary.LittleEndian.Uint32(data[pos:])
		pos += 4
		return v, nil
	}
	version, err := rd()
	if err != nil {
		return nil, err
	}
	if version != persistVersion {
		return nil, fmt.Errorf("%w: unsupported version %d", ErrCorrupt, version)
	}
	dim, err := rd()
	if err != nil {
		return nil, err
	}
	metric, err := rd()
	if err != nil {
		return nil, err
	}
	m, err := rd()
	if err != nil {
		return nil, err
	}
	maxLevel, err := rd()
	if err != nil {
		return nil, err
	}
	count, err := rd()
	if err != nil {
		return nil, err
	}
	maxLevelUsed, err := rd()
	if err != nil {
		return nil, err
	}
	if pos+8 > len(data)-4 {
		return nil, fmt.Errorf("%w: truncated seed", ErrCorrupt)
	}
	seed := binary.LittleEndian.Uint64(data[pos:])
	pos += 8

	// Verify CRC over everything except the trailing checksum.
	wantCRC := binary.LittleEndian.Uint32(data[len(data)-4:])
	if crc32.ChecksumIEEE(data[:len(data)-4]) != wantCRC {
		return nil, fmt.Errorf("%w: checksum mismatch", ErrCorrupt)
	}

	cfg := Config{
		Dim:      int(dim),
		Metric:   Metric(metric),
		M:        int(m),
		Seed:     seed,
		MaxLevel: int(maxLevel),
	}
	if err := cfg.validate(); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCorrupt, err)
	}
	if maxLevelUsed > maxLevel {
		return nil, fmt.Errorf("%w: max level %d exceeds cap %d", ErrCorrupt, maxLevelUsed, maxLevel)
	}

	need := func(n int) error {
		if pos+n > len(data)-4 {
			return fmt.Errorf("%w: truncated payload", ErrCorrupt)
		}
		return nil
	}

	ids := make([]uint64, count)
	for i := range ids {
		if err := need(8); err != nil {
			return nil, err
		}
		ids[i] = binary.LittleEndian.Uint64(data[pos:])
		pos += 8
		if i > 0 && ids[i] <= ids[i-1] {
			return nil, fmt.Errorf("%w: ids not strictly increasing", ErrCorrupt)
		}
	}
	vecs := make([][]float32, count)
	for i := range vecs {
		if err := need(int(dim) * 4); err != nil {
			return nil, err
		}
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = math.Float32frombits(binary.LittleEndian.Uint32(data[pos:]))
			pos += 4
		}
		vecs[i] = vec
	}
	levels := make([]int, count)
	for i := range levels {
		v, err := rd()
		if err != nil {
			return nil, err
		}
		if v > maxLevelUsed {
			return nil, fmt.Errorf("%w: level %d exceeds used max %d", ErrCorrupt, v, maxLevelUsed)
		}
		levels[i] = int(v)
	}

	idx := &Index{
		cfg:   cfg,
		idMap: make(map[uint64]int, count),
		entry: -1,
		nodes: make([]node, count),
	}
	idx.graph = make([][][]int, maxLevelUsed+1)
	idx.rev = make([][][]int, maxLevelUsed+1)
	for l := range idx.graph {
		idx.graph[l] = make([][]int, count)
		idx.rev[l] = make([][]int, count)
	}
	for i, id := range ids {
		idx.nodes[i] = node{id: id, vec: vecs[i], level: levels[i], alive: true}
		idx.idMap[id] = i
	}
	for i := range ids {
		for l := 0; l <= levels[i]; l++ {
			cnt, err := rd()
			if err != nil {
				return nil, err
			}
			if err := need(int(cnt) * 8); err != nil {
				return nil, err
			}
			list := make([]int, cnt)
			for j := range list {
				nbID := binary.LittleEndian.Uint64(data[pos:])
				pos += 8
				nbSlot, ok := idx.idMap[nbID]
				if !ok {
					return nil, fmt.Errorf("%w: neighbor id %d unknown", ErrCorrupt, nbID)
				}
				if nbSlot == i {
					return nil, fmt.Errorf("%w: self loop", ErrCorrupt)
				}
				if idx.nodes[nbSlot].level < l {
					return nil, fmt.Errorf("%w: neighbor %d below layer %d", ErrCorrupt, nbID, l)
				}
				list[j] = nbSlot
			}
			idx.graph[l][i] = list
		}
	}
	// Rebuild incoming adjacency as the exact reverse of outgoing edges.
	for i := range ids {
		for l := 0; l <= levels[i]; l++ {
			for _, nb := range idx.graph[l][i] {
				idx.rev[l][nb] = append(idx.rev[l][nb], i)
			}
		}
	}
	for l := range idx.rev {
		for i := range ids {
			idx.rev[l][i] = idx.sortNeighbors(idx.rev[l][i], i)
		}
	}
	if pos != len(data)-4 {
		return nil, fmt.Errorf("%w: %d trailing bytes", ErrCorrupt, len(data)-4-pos)
	}
	idx.entry = idx.entryLocked()
	if err := idx.verifyLocked(); err != nil {
		return nil, err
	}
	return idx, nil
}

// verifyLocked checks structural consistency and the neighbor invariant.
// It returns ErrCorrupt on any defect.
func (idx *Index) verifyLocked() error {
	// idMap <-> nodes consistency.
	if len(idx.idMap) != idx.lenLocked() {
		return fmt.Errorf("%w: idMap size %d != live %d", ErrCorrupt, len(idx.idMap), idx.lenLocked())
	}
	for id, slot := range idx.idMap {
		if slot < 0 || slot >= len(idx.nodes) || !idx.nodes[slot].alive || idx.nodes[slot].id != id {
			return fmt.Errorf("%w: idMap/nodes mismatch for id %d", ErrCorrupt, id)
		}
	}
	for slot, nd := range idx.nodes {
		if nd.alive {
			if idx.idMap[nd.id] != slot {
				return fmt.Errorf("%w: node %d not in idMap", ErrCorrupt, nd.id)
			}
			if len(nd.vec) != idx.cfg.Dim {
				return fmt.Errorf("%w: node %d dim %d", ErrCorrupt, nd.id, len(nd.vec))
			}
			if nd.level < 0 || nd.level > idx.cfg.MaxLevel {
				return fmt.Errorf("%w: node %d level %d out of range", ErrCorrupt, nd.id, nd.level)
			}
		} else {
			for _, s := range idx.free {
				if s == slot {
					goto freeOK
				}
			}
			return fmt.Errorf("%w: dead slot %d not in free list", ErrCorrupt, slot)
		}
	freeOK:
	}
	for _, s := range idx.free {
		if s < 0 || s >= len(idx.nodes) || idx.nodes[s].alive {
			return fmt.Errorf("%w: free list contains live slot %d", ErrCorrupt, s)
		}
	}

	// Graph structure.
	if len(idx.graph) == 0 && idx.lenLocked() > 0 {
		return fmt.Errorf("%w: empty graph with live nodes", ErrCorrupt)
	}
	maxL := 0
	for _, nd := range idx.nodes {
		if nd.alive && nd.level > maxL {
			maxL = nd.level
		}
	}
	if len(idx.graph) != maxL+1 {
		return fmt.Errorf("%w: graph layers %d != maxLevel+1 %d", ErrCorrupt, len(idx.graph), maxL+1)
	}
	if len(idx.rev) != maxL+1 {
		return fmt.Errorf("%w: rev layers %d != maxLevel+1 %d", ErrCorrupt, len(idx.rev), maxL+1)
	}
	checkList := func(l, slot int, list []int) error {
		prevDist := math.Inf(-1)
		var prevID uint64
		for _, nb := range list {
			if nb == slot {
				return fmt.Errorf("%w: node %d self loop at layer %d", ErrCorrupt, idx.nodes[slot].id, l)
			}
			if nb < 0 || nb >= len(idx.nodes) || !idx.nodes[nb].alive {
				return fmt.Errorf("%w: node %d layer %d neighbor %d invalid", ErrCorrupt, idx.nodes[slot].id, l, nb)
			}
			if idx.nodes[nb].level < l {
				return fmt.Errorf("%w: node %d layer %d neighbor %d below layer", ErrCorrupt, idx.nodes[slot].id, l, nb)
			}
			d := dist(idx.nodes[slot].vec, idx.nodes[nb].vec, idx.cfg.Metric)
			if d < prevDist || (d == prevDist && idx.nodes[nb].id < prevID) {
				return fmt.Errorf("%w: node %d layer %d neighbor list not sorted", ErrCorrupt, idx.nodes[slot].id, l)
			}
			prevDist = d
			prevID = idx.nodes[nb].id
		}
		return nil
	}
	for l := 0; l <= maxL; l++ {
		mmax := idx.neighborBudget(l)
		for slot, nd := range idx.nodes {
			out := idx.graph[l][slot]
			in := idx.rev[l][slot]
			if !nd.alive || nd.level < l {
				if out != nil || in != nil {
					return fmt.Errorf("%w: layer %d slot %d should have no edges", ErrCorrupt, l, slot)
				}
				continue
			}
			if len(out) > mmax {
				return fmt.Errorf("%w: node %d layer %d has %d outgoing > %d", ErrCorrupt, nd.id, l, len(out), mmax)
			}
			if err := checkList(l, slot, out); err != nil {
				return err
			}
			if err := checkList(l, slot, in); err != nil {
				return err
			}
		}
	}

	// Entry point.
	wantEntry := idx.entryLocked()
	if idx.entry != wantEntry {
		return fmt.Errorf("%w: entry %d != expected %d", ErrCorrupt, idx.entry, wantEntry)
	}

	// Deep invariant: each node's outgoing list at layer l equals the
	// deterministic heuristic selection, and rev is the exact reverse.
	for l := 0; l <= maxL; l++ {
		members := idx.layerMembers(l)
		for _, a := range members {
			want := idx.heuristicNeighbors(l, a)
			got := idx.graph[l][a]
			if len(got) != len(want) {
				return fmt.Errorf("%w: node %d layer %d outgoing count %d != %d", ErrCorrupt, idx.nodes[a].id, l, len(got), len(want))
			}
			for i := range want {
				if got[i] != want[i] {
					return fmt.Errorf("%w: node %d layer %d outgoing mismatch at position %d", ErrCorrupt, idx.nodes[a].id, l, i)
				}
			}
			// rev[a] = {b : a in graph[l][b]}.
			wantIn := make([]int, 0)
			for _, b := range members {
				if b == a {
					continue
				}
				for _, nb := range idx.graph[l][b] {
					if nb == a {
						wantIn = append(wantIn, b)
						break
					}
				}
			}
			wantIn = idx.sortNeighbors(wantIn, a)
			gotIn := idx.rev[l][a]
			if len(gotIn) != len(wantIn) {
				return fmt.Errorf("%w: node %d layer %d incoming count %d != %d", ErrCorrupt, idx.nodes[a].id, l, len(gotIn), len(wantIn))
			}
			for i := range wantIn {
				if gotIn[i] != wantIn[i] {
					return fmt.Errorf("%w: node %d layer %d incoming mismatch at position %d", ErrCorrupt, idx.nodes[a].id, l, i)
				}
			}
		}
	}
	return nil
}

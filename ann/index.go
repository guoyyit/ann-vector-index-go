package ann

import (
	"sync"
)

// Result is a single search hit.
type Result struct {
	ID       uint64
	Distance float64
}

// Stats is a snapshot of index state for observability.
type Stats struct {
	Live      int
	Slots     int
	MaxLevel  int
	LayerSize []int
	BuildMode string
}

// Index is an in-memory approximate nearest-neighbor index.
//
// The index is safe for concurrent use: searches take a read lock and
// mutations take the write lock, so a query never observes a half-applied
// write. The graph is a deterministic function of the live (id, vector)
// set and the configured seed: insertion order, batch boundaries and
// duplicate inserts do not change the final graph.
type Index struct {
	mu  sync.RWMutex
	cfg Config

	nodes []node // dense slots; nodes[slot].alive == false means the slot is free
	free  []int  // recycled slots
	idMap map[uint64]int

	// graph[l][slot] = outgoing neighbor slots at layer l (the m nearest).
	graph [][][]int
	// rev[l][slot] = incoming neighbor slots at layer l (nodes that list
	// slot). Search traverses both directions, giving the reachability of
	// an undirected graph while every edge remains a pure function of the
	// live set.
	rev   [][][]int
	entry int // slot of the top-layer entry point; -1 when empty

	// dirty is set by deletions: neighbor lists may be short until the next
	// rebuild. Searches and Verify rebuild before reading, so callers never
	// observe a stale graph.
	dirty bool
}

type node struct {
	id    uint64
	vec   []float32
	level int
	alive bool
}

// New creates an empty index with the given configuration.
func New(cfg Config) (*Index, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	idx := &Index{
		cfg:   cfg,
		idMap: make(map[uint64]int),
		entry: -1,
	}
	return idx, nil
}

// Len returns the number of live vectors.
func (idx *Index) Len() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.lenLocked()
}

func (idx *Index) lenLocked() int {
	n := 0
	for _, nd := range idx.nodes {
		if nd.alive {
			n++
		}
	}
	return n
}

// Stats returns a snapshot of the index state.
func (idx *Index) Stats() Stats {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.statsLocked()
}

func (idx *Index) statsLocked() Stats {
	st := Stats{
		Live:      idx.lenLocked(),
		Slots:     len(idx.nodes),
		MaxLevel:  len(idx.graph) - 1,
		BuildMode: "heuristic",
	}
	st.LayerSize = make([]int, len(idx.graph))
	for l := range idx.graph {
		for slot := range idx.graph[l] {
			if idx.nodes[slot].alive && idx.nodes[slot].level >= l {
				st.LayerSize[l]++
			}
		}
	}
	return st
}

// Insert adds or replaces the vector for id. Inserting the same id with
// the same or a different vector leaves a single live entry; the final
// graph depends only on the live set, not on insertion order.
func (idx *Index) Insert(id uint64, vec []float32) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	return idx.insertLocked(id, vec)
}

// InsertBatch inserts many vectors. It is equivalent to inserting them one
// by one in any order: the final graph is identical.
func (idx *Index) InsertBatch(ids []uint64, vecs [][]float32) error {
	if len(ids) != len(vecs) {
		return ErrDimMismatch
	}
	idx.mu.Lock()
	defer idx.mu.Unlock()
	for i := range ids {
		if err := idx.insertLocked(ids[i], vecs[i]); err != nil {
			return err
		}
	}
	return nil
}

// Delete removes id. Deleted vectors never appear in results. The slot is
// recycled; call Compact to shrink the internal storage to the live set.
func (idx *Index) Delete(id uint64) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	return idx.deleteLocked(id)
}

// Search returns the k nearest neighbors of q. ef controls the search
// breadth (ef >= k); larger ef improves recall at higher cost.
//
// Search takes the write lock: if deletions have left the graph stale, the
// graph is rebuilt before searching, so a query never observes a deleted
// vector or a half-applied write. Concurrent searches are serialized, but
// every query sees a consistent state.
func (idx *Index) Search(q []float32, k int, ef int) ([]Result, error) {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	return idx.searchLocked(q, k, ef)
}

// ExactSearch returns the exact k nearest neighbors by brute force. It is
// used as the ground truth for recall evaluation.
func (idx *Index) ExactSearch(q []float32, k int) ([]Result, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.exactSearchLocked(q, k)
}

// Rebuild reconstructs the graph from the live vectors. It restores the
// exact neighbor invariant after churn and reclaims graph memory.
func (idx *Index) Rebuild() error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.rebuildGraphLocked()
	idx.dirty = false
	return nil
}

// Compact rebuilds the graph and shrinks internal storage to the live set,
// reclaiming all space held by deleted vectors.
func (idx *Index) Compact() error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	if err := idx.compactLocked(); err != nil {
		return err
	}
	idx.dirty = false
	return nil
}

// Verify checks internal consistency and returns ErrCorrupt on any defect.
// If deletions have left the graph stale, Verify rebuilds first so that the
// canonical invariant is checked.
func (idx *Index) Verify() error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	if idx.dirty {
		idx.rebuildGraphLocked()
		idx.dirty = false
	}
	return idx.verifyLocked()
}

package ann

import (
	"container/heap"
	"fmt"
	"math"
)

func (idx *Index) checkVec(vec []float32) error {
	if len(vec) != idx.cfg.Dim {
		return fmt.Errorf("%w: got %d, want %d", ErrDimMismatch, len(vec), idx.cfg.Dim)
	}
	for _, v := range vec {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("%w: vector contains non-finite values", ErrInvalidConfig)
		}
	}
	return nil
}

func (idx *Index) insertLocked(id uint64, vec []float32) error {
	if err := idx.checkVec(vec); err != nil {
		return err
	}
	stored := prepareVector(vec, idx.cfg.Metric)
	if slot, ok := idx.idMap[id]; ok {
		// Replacing an existing id: update the vector; the graph is
		// rebuilt on the next search.
		idx.nodes[slot].vec = stored
		idx.nodes[slot].level = levelForID(id, idx.cfg.Seed, idx.cfg.M, idx.cfg.MaxLevel)
		idx.dirty = true
		return nil
	}
	var slot int
	if len(idx.free) > 0 {
		slot = idx.free[len(idx.free)-1]
		idx.free = idx.free[:len(idx.free)-1]
	} else {
		slot = len(idx.nodes)
		idx.nodes = append(idx.nodes, node{})
	}
	idx.nodes[slot] = node{
		id:    id,
		vec:   stored,
		level: levelForID(id, idx.cfg.Seed, idx.cfg.M, idx.cfg.MaxLevel),
		alive: true,
	}
	idx.idMap[id] = slot
	idx.dirty = true
	return nil
}

func (idx *Index) deleteLocked(id uint64) error {
	slot, ok := idx.idMap[id]
	if !ok {
		return ErrNotFound
	}
	idx.nodes[slot] = node{}
	delete(idx.idMap, id)
	idx.free = append(idx.free, slot)
	// The graph is rebuilt on the next search; deleted vectors are skipped
	// in the meantime.
	idx.dirty = true
	return nil
}

// allNeighbors returns the union of outgoing and incoming neighbors at
// layer l for slot. Search traverses both directions for undirected-graph
// reachability.
func (idx *Index) allNeighbors(layer, slot int) []int {
	g := idx.graph[layer][slot]
	r := idx.rev[layer][slot]
	out := make([]int, 0, len(g)+len(r))
	out = append(out, g...)
	out = append(out, r...)
	return out
}

// searchLayerLocked runs best-first search on one layer starting from
// entrySlot, keeping ef best nodes. It returns the best candidates
// nearest-first and the number of distance evaluations (visited nodes).
func (idx *Index) searchLayerLocked(q []float32, layer int, entrySlot int, ef int, visited []int, stamp int) ([]candidate, int) {
	vis := stamp
	visited[entrySlot] = vis
	visitedCount := 1

	candidates := &candMinHeap{}
	results := &resultMaxHeap{}
	d0 := dist(q, idx.nodes[entrySlot].vec, idx.cfg.Metric)
	heap.Push(candidates, candidate{dist: d0, id: idx.nodes[entrySlot].id, slot: entrySlot})
	heap.Push(results, candidate{dist: d0, id: idx.nodes[entrySlot].id, slot: entrySlot})

	for candidates.Len() > 0 {
		c := heap.Pop(candidates).(candidate)
		if results.Len() >= ef && c.dist > (*results)[0].dist {
			break
		}
		for _, nb := range idx.allNeighbors(layer, c.slot) {
			if visited[nb] == vis {
				continue
			}
			visited[nb] = vis
			visitedCount++
			nd := idx.nodes[nb]
			if !nd.alive {
				continue
			}
			d := dist(q, nd.vec, idx.cfg.Metric)
			heap.Push(candidates, candidate{dist: d, id: nd.id, slot: nb})
			if results.Len() < ef || d < (*results)[0].dist ||
				(d == (*results)[0].dist && nd.id < (*results)[0].id) {
				heap.Push(results, candidate{dist: d, id: nd.id, slot: nb})
				if results.Len() > ef {
					heap.Pop(results)
				}
			}
		}
	}

	out := make([]candidate, 0, results.Len())
	for results.Len() > 0 {
		out = append(out, heap.Pop(results).(candidate))
	}
	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return out, visitedCount
}

func (idx *Index) searchLocked(q []float32, k int, ef int) ([]Result, error) {
	res, _, err := idx.searchVisitedLocked(q, k, ef)
	return res, err
}

func (idx *Index) searchVisitedLocked(q []float32, k int, ef int) ([]Result, int, error) {
	if err := idx.checkVec(q); err != nil {
		return nil, 0, err
	}
	if k <= 0 {
		return nil, 0, fmt.Errorf("%w: k must be positive, got %d", ErrInvalidConfig, k)
	}
	if idx.entry == -1 {
		return nil, 0, ErrEmptyIndex
	}
	if ef < k {
		ef = k
	}
	if idx.dirty {
		idx.rebuildGraphLocked()
		idx.dirty = false
	}
	query := prepareVector(q, idx.cfg.Metric)
	visited := make([]int, len(idx.nodes))
	stamp := 0
	visitedCount := 0

	cur := idx.entry
	for l := len(idx.graph) - 1; l >= 1; l-- {
		stamp++
		upper, vc := idx.searchLayerLocked(query, l, cur, 1, visited, stamp)
		visitedCount += vc
		if len(upper) == 0 {
			return nil, visitedCount, ErrCorrupt
		}
		cur = upper[0].slot
	}
	stamp++
	cands, vc := idx.searchLayerLocked(query, 0, cur, ef, visited, stamp)
	visitedCount += vc
	if len(cands) > k {
		cands = cands[:k]
	}
	out := make([]Result, len(cands))
	for i, c := range cands {
		out[i] = Result{ID: idx.nodes[c.slot].id, Distance: c.dist}
	}
	return out, visitedCount, nil
}

func (idx *Index) exactSearchLocked(q []float32, k int) ([]Result, error) {
	if err := idx.checkVec(q); err != nil {
		return nil, err
	}
	if k <= 0 {
		return nil, fmt.Errorf("%w: k must be positive, got %d", ErrInvalidConfig, k)
	}
	if idx.entry == -1 {
		return nil, ErrEmptyIndex
	}
	query := prepareVector(q, idx.cfg.Metric)
	best := &resultMaxHeap{}
	for slot, nd := range idx.nodes {
		if !nd.alive {
			continue
		}
		d := dist(query, nd.vec, idx.cfg.Metric)
		c := candidate{dist: d, id: nd.id, slot: slot}
		if best.Len() < k {
			heap.Push(best, c)
			continue
		}
		top := (*best)[0]
		if c.dist < top.dist || (c.dist == top.dist && c.id < top.id) {
			heap.Pop(best)
			heap.Push(best, c)
		}
	}
	out := make([]Result, best.Len())
	for i := len(out) - 1; i >= 0; i-- {
		c := heap.Pop(best).(candidate)
		out[i] = Result{ID: c.id, Distance: c.dist}
	}
	return out, nil
}

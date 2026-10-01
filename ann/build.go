package ann

import (
	"sort"
)

// neighborBudget returns the neighbor budget for layer l: layer 0 uses
// 2*M, upper layers use M.
func (idx *Index) neighborBudget(l int) int {
	m := idx.cfg.M
	if l == 0 {
		m = 2 * idx.cfg.M
	}
	return m
}

// layerMembers returns the live slots whose level is >= l.
func (idx *Index) layerMembers(l int) []int {
	members := make([]int, 0, len(idx.nodes))
	for slot, nd := range idx.nodes {
		if nd.alive && nd.level >= l {
			members = append(members, slot)
		}
	}
	return members
}

// sortNeighbors sorts list by (distance from a, neighbor id) and returns it.
func (idx *Index) sortNeighbors(list []int, a int) []int {
	metric := idx.cfg.Metric
	avec := idx.nodes[a].vec
	sort.Slice(list, func(i, j int) bool {
		bi, bj := list[i], list[j]
		di := dist(avec, idx.nodes[bi].vec, metric)
		dj := dist(avec, idx.nodes[bj].vec, metric)
		if di != dj {
			return di < dj
		}
		return idx.nodes[bi].id < idx.nodes[bj].id
	})
	return list
}

// containsSlot reports whether list contains slot.
func containsSlot(list []int, slot int) bool {
	for _, s := range list {
		if s == slot {
			return true
		}
	}
	return false
}

// heuristicNeighbors computes the deterministic neighbor selection for node
// a at layer l: candidates are visited nearest-first and kept only if not
// closer to an already-selected neighbor than to a (HNSW diversity
// heuristic). The result is a pure function of the live set and seed.
func (idx *Index) heuristicNeighbors(l, a int) []int {
	m := idx.neighborBudget(l)
	members := idx.layerMembers(l)
	type pair struct {
		d    float64
		id   uint64
		slot int
	}
	pairs := make([]pair, 0, len(members)-1)
	for _, b := range members {
		if b == a {
			continue
		}
		pairs = append(pairs, pair{
			d:    dist(idx.nodes[a].vec, idx.nodes[b].vec, idx.cfg.Metric),
			id:   idx.nodes[b].id,
			slot: b,
		})
	}
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].d != pairs[j].d {
			return pairs[i].d < pairs[j].d
		}
		return pairs[i].id < pairs[j].id
	})
	selected := make([]int, 0, m)
	for _, p := range pairs {
		if len(selected) >= m {
			break
		}
		useful := true
		for _, s := range selected {
			if dist(idx.nodes[p.slot].vec, idx.nodes[s].vec, idx.cfg.Metric) < p.d {
				useful = false
				break
			}
		}
		if useful {
			selected = append(selected, p.slot)
		}
	}
	return selected
}

// rebuildGraphLocked reconstructs the graph from the live vectors. For each
// node we select m neighbors using the HNSW diversity heuristic. This
// prefers diverse, non-redundant edges (including long-range cross-cluster
// links) while remaining a pure function of the live set and seed, so the
// result is independent of insertion order. The incoming adjacency is the
// exact reverse of the outgoing edges.
func (idx *Index) rebuildGraphLocked() {
	maxL := 0
	for _, nd := range idx.nodes {
		if nd.alive && nd.level > maxL {
			maxL = nd.level
		}
	}
	graph := make([][][]int, maxL+1)
	rev := make([][][]int, maxL+1)
	for l := 0; l <= maxL; l++ {
		graph[l] = make([][]int, len(idx.nodes))
		rev[l] = make([][]int, len(idx.nodes))
		members := idx.layerMembers(l)
		for _, a := range members {
			graph[l][a] = idx.heuristicNeighbors(l, a)
		}
		// Incoming edges are the exact reverse of outgoing edges.
		for _, a := range members {
			for _, b := range graph[l][a] {
				rev[l][b] = append(rev[l][b], a)
			}
		}
		for _, a := range members {
			rev[l][a] = idx.sortNeighbors(rev[l][a], a)
		}
	}
	idx.graph = graph
	idx.rev = rev
	idx.entry = idx.entryLocked()
}

// entryLocked returns the slot with the highest level (tie-break smallest
// id), or -1 if the index is empty.
func (idx *Index) entryLocked() int {
	best := -1
	for slot, nd := range idx.nodes {
		if !nd.alive {
			continue
		}
		if best == -1 || nd.level > idx.nodes[best].level ||
			(nd.level == idx.nodes[best].level && nd.id < idx.nodes[best].id) {
			best = slot
		}
	}
	return best
}

// compactLocked rebuilds dense storage from live nodes and reconstructs
// the graph, reclaiming every slot held by deleted vectors.
func (idx *Index) compactLocked() error {
	live := make([]int, 0, len(idx.nodes))
	for slot, nd := range idx.nodes {
		if nd.alive {
			live = append(live, slot)
		}
	}
	sort.Slice(live, func(i, j int) bool {
		return idx.nodes[live[i]].id < idx.nodes[live[j]].id
	})
	newNodes := make([]node, len(live))
	idx.idMap = make(map[uint64]int, len(live))
	for i, slot := range live {
		newNodes[i] = idx.nodes[slot]
		idx.idMap[newNodes[i].id] = i
	}
	idx.nodes = newNodes
	idx.free = nil
	idx.rebuildGraphLocked()
	return nil
}

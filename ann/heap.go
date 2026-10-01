package ann

// candidate is a (distance, id, slot) triple used during graph search.
type candidate struct {
	dist float64
	id   uint64
	slot int
}

// candMinHeap is a min-heap ordered by (dist, id): nearest first.
type candMinHeap []candidate

func (h candMinHeap) Len() int { return len(h) }
func (h candMinHeap) Less(i, j int) bool {
	if h[i].dist != h[j].dist {
		return h[i].dist < h[j].dist
	}
	return h[i].id < h[j].id
}
func (h candMinHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *candMinHeap) Push(x any)   { *h = append(*h, x.(candidate)) }
func (h *candMinHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// resultMaxHeap keeps the worst (largest dist, then largest id) on top so
// that pruning to a fixed size is cheap.
type resultMaxHeap []candidate

func (h resultMaxHeap) Len() int { return len(h) }
func (h resultMaxHeap) Less(i, j int) bool {
	if h[i].dist != h[j].dist {
		return h[i].dist > h[j].dist
	}
	return h[i].id > h[j].id
}
func (h resultMaxHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *resultMaxHeap) Push(x any)   { *h = append(*h, x.(candidate)) }
func (h *resultMaxHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

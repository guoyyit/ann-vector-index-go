package ann

import "sort"

// exactSearch 对给定数据集做精确线性扫描 kNN。
// 结果按 (距离升序, ID 升序) 排列，与 map 遍历顺序无关，完全确定。
func exactSearch(query []float64, vectors map[int64][]float64, k int) []Result {
	if k <= 0 || len(vectors) == 0 {
		return nil
	}
	tk := newTopK(k)
	for id, v := range vectors {
		tk.consider(id, cosineDistance(query, v))
	}
	return tk.results()
}

// topKCollector 保留距离最小的 k 个；堆顶是当前最差候选
// （距离最大；距离相同则 ID 最大），用有界堆控制内存。
type topKCollector struct {
	k    int
	heap []Result
}

func newTopK(k int) *topKCollector {
	return &topKCollector{k: k, heap: make([]Result, 0, k+1)}
}

// worse 判断 a 是否比 b 更差（应更靠近堆顶被淘汰）。
func worse(a, b Result) bool {
	if a.Distance != b.Distance {
		return a.Distance > b.Distance
	}
	return a.ID > b.ID
}

func (c *topKCollector) consider(id int64, d float64) {
	r := Result{ID: id, Distance: d}
	if len(c.heap) < c.k {
		c.heap = append(c.heap, r)
		c.up(len(c.heap) - 1)
		return
	}
	if worse(c.heap[0], r) {
		c.heap[0] = r
		c.down(0)
	}
}

func (c *topKCollector) up(i int) {
	for i > 0 {
		p := (i - 1) / 2
		// 父节点应比子节点"更差"；否则上浮。
		if worse(c.heap[i], c.heap[p]) {
			c.heap[i], c.heap[p] = c.heap[p], c.heap[i]
			i = p
		} else {
			break
		}
	}
}

func (c *topKCollector) down(i int) {
	n := len(c.heap)
	for {
		left, right, best := 2*i+1, 2*i+2, i
		if left < n && worse(c.heap[left], c.heap[best]) {
			best = left
		}
		if right < n && worse(c.heap[right], c.heap[best]) {
			best = right
		}
		if best == i {
			return
		}
		c.heap[i], c.heap[best] = c.heap[best], c.heap[i]
		i = best
	}
}

func (c *topKCollector) results() []Result {
	out := make([]Result, len(c.heap))
	copy(out, c.heap)
	sort.Slice(out, func(i, j int) bool {
		if out[i].Distance != out[j].Distance {
			return out[i].Distance < out[j].Distance
		}
		return out[i].ID < out[j].ID
	})
	return out
}

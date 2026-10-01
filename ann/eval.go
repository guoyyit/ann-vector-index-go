package ann

// EvalReport summarizes recall and search cost for a query workload.
type EvalReport struct {
	K           int
	EF          int
	NumQueries  int
	Recall      float64 // mean recall@k
	AvgVisited  float64 // mean number of distance evaluations per query
	AvgDistance float64 // mean distance to the k-th result (sanity)
}

// Evaluate runs exact and ANN search over queries and reports mean recall@k
// together with the average number of distance evaluations (visited nodes).
// Recall is computed as |ann ∩ exact| / k per query, averaged.
func (idx *Index) Evaluate(queries [][]float32, k, ef int) (EvalReport, error) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if idx.entry == -1 {
		return EvalReport{}, ErrEmptyIndex
	}
	if k <= 0 {
		return EvalReport{}, ErrInvalidConfig
	}
	if ef < k {
		ef = k
	}

	report := EvalReport{K: k, EF: ef, NumQueries: len(queries)}
	for _, q := range queries {
		if err := idx.checkVec(q); err != nil {
			return report, err
		}
		exact, err := idx.exactSearchLocked(q, k)
		if err != nil {
			return report, err
		}
		ann, visited, err := idx.searchVisitedLocked(q, k, ef)
		if err != nil {
			return report, err
		}
		hit := 0
		exactSet := make(map[uint64]struct{}, len(exact))
		for _, r := range exact {
			exactSet[r.ID] = struct{}{}
		}
		for _, r := range ann {
			if _, ok := exactSet[r.ID]; ok {
				hit++
			}
		}
		n := len(exact)
		if n == 0 {
			n = k
		}
		report.Recall += float64(hit) / float64(n)
		report.AvgVisited += float64(visited)
		if len(ann) > 0 {
			report.AvgDistance += ann[len(ann)-1].Distance
		}
	}
	if len(queries) > 0 {
		report.Recall /= float64(len(queries))
		report.AvgVisited /= float64(len(queries))
		report.AvgDistance /= float64(len(queries))
	}
	return report, nil
}

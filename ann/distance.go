package ann

import "math"

// prepareVector returns the vector stored in the index. For cosine the
// vector is L2-normalized so that distance = 1 - dot product.
func prepareVector(vec []float32, metric Metric) []float32 {
	if metric != MetricCosine {
		return vec
	}
	var norm float64
	for _, v := range vec {
		norm += float64(v) * float64(v)
	}
	if norm == 0 {
		return vec
	}
	inv := float32(1.0 / math.Sqrt(norm))
	out := make([]float32, len(vec))
	for i, v := range vec {
		out[i] = v * inv
	}
	return out
}

// dist returns the distance between two stored vectors. Smaller is closer.
func dist(a, b []float32, metric Metric) float64 {
	switch metric {
	case MetricL2:
		var s float64
		for i := range a {
			d := float64(a[i]) - float64(b[i])
			s += d * d
		}
		return s
	case MetricCosine:
		var dot float64
		for i := range a {
			dot += float64(a[i]) * float64(b[i])
		}
		return 1 - dot
	case MetricIP:
		var dot float64
		for i := range a {
			dot += float64(a[i]) * float64(b[i])
		}
		return -dot
	}
	return 0
}

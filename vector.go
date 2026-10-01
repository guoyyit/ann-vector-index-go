package ann

import "math"

// normalized 返回向量的归一化副本；零向量返回同维零向量（不做特殊处理，
// 其与任何向量的余弦距离按 1 计，保持确定性）。
func normalized(v []float64) []float64 {
	var n float64
	for _, x := range v {
		n += x * x
	}
	out := make([]float64, len(v))
	if n == 0 {
		return out
	}
	inv := 1.0 / math.Sqrt(n)
	for i, x := range v {
		out[i] = x * inv
	}
	return out
}

// cosineDistance 计算余弦距离 1 - cos(a,b)，入参期望已归一化。
// 结果夹在 [0,2]，并消除浮点抖动夹回。
func cosineDistance(a, b []float64) float64 {
	s := dot(a, b)
	// 浮点误差可能略微越过 ±1。
	if s > 1 {
		s = 1
	} else if s < -1 {
		s = -1
	}
	return 1 - s
}

// dot 计算点积。
func dot(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

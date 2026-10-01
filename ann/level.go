package ann

import "math"

// splitmix64 is a deterministic 64-bit mixer.
func splitmix64(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}

// levelForID assigns a deterministic layer for an id. The assignment is a
// pure function of (id, seed), so it is independent of insertion order and
// of any random stream.
func levelForID(id, seed uint64, m, maxLevel int) int {
	h := splitmix64(id ^ seed)
	// Uniform draw in (0,1); h is in [0, 2^64).
	u := float64(h>>11) / float64(uint64(1)<<53)
	if u <= 0 {
		return maxLevel
	}
	l := int(math.Floor(-math.Log(u) / math.Log(float64(m))))
	if l > maxLevel {
		l = maxLevel
	}
	return l
}

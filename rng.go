package ann

import "math"

// rng 是确定性伪随机数生成器（SplitMix64），不依赖 math/rand，
// 保证所有 Go 版本/平台上相同种子产出相同序列。
type rng struct {
	state    uint64
	spare    float64
	hasSpare bool
}

func newRNG(seed int64) *rng {
	s := uint64(seed)
	if s == 0 {
		s = 0x9e3779b97f4a7c15
	}
	return &rng{state: s}
}

// Uint64 返回下一个随机 uint64（SplitMix64）。
func (r *rng) Uint64() uint64 {
	r.state += 0x9e3779b97f4a7c15
	z := r.state
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// Float64 返回 [0,1) 的均匀随机数。
func (r *rng) Float64() float64 {
	return float64(r.Uint64()>>11) * 0x1.0p-53
}

// Normal 返回标准正态随机数（Box-Muller，第二采样缓存在生成器内）。
func (r *rng) Normal() float64 {
	if r.hasSpare {
		r.hasSpare = false
		return r.spare
	}
	u1 := 0.0
	for u1 == 0 { // 避免 log(0)
		u1 = r.Float64()
	}
	u2 := r.Float64()
	mag := math.Sqrt(-2.0 * math.Log(u1))
	theta := 2 * math.Pi * u2
	r.spare = mag * math.Sin(theta)
	r.hasSpare = true
	return mag * math.Cos(theta)
}

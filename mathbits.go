package ann

import "math"

func float64Bits(f float64) uint64     { return math.Float64bits(f) }
func float64FromBits(b uint64) float64 { return math.Float64frombits(b) }

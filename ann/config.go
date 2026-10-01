package ann

import (
	"errors"
	"fmt"
)

// Metric is the distance/similarity metric used by the index.
type Metric int

const (
	// MetricL2 is squared Euclidean distance (smaller is closer).
	MetricL2 Metric = iota
	// MetricCosine is cosine distance (1 - cosine similarity). Vectors are
	// normalized at insert time, so distance equals 1 - dot product.
	MetricCosine
	// MetricIP is inner product distance (-dot product); larger dot is closer.
	MetricIP
)

// Config controls index construction. All fields are immutable after the
// index is created; persistence stores the config and loading rejects
// mismatched parameters.
type Config struct {
	// Dim is the vector dimension. Must be positive.
	Dim int
	// Metric selects the distance metric.
	Metric Metric
	// M is the neighbor budget per node. Layer 0 uses 2*M, upper layers use M.
	// Larger M improves recall at higher build/search cost.
	M int
	// Seed feeds the deterministic per-ID level assignment. Two indexes with
	// the same (Seed, dataset) produce bit-for-bit identical graphs.
	Seed uint64
	// MaxLevel caps the highest layer a node can occupy (safety bound).
	MaxLevel int
}

// DefaultConfig returns a sane configuration for dimension dim.
func DefaultConfig(dim int) Config {
	return Config{
		Dim:      dim,
		Metric:   MetricL2,
		M:        16,
		Seed:     0,
		MaxLevel: 16,
	}
}

func (c *Config) validate() error {
	if c.Dim <= 0 {
		return fmt.Errorf("%w: dim must be positive, got %d", ErrInvalidConfig, c.Dim)
	}
	if c.M <= 0 {
		return fmt.Errorf("%w: M must be positive, got %d", ErrInvalidConfig, c.M)
	}
	if c.MaxLevel <= 0 {
		c.MaxLevel = 16
	}
	switch c.Metric {
	case MetricL2, MetricCosine, MetricIP:
	default:
		return fmt.Errorf("%w: unknown metric %d", ErrInvalidConfig, c.Metric)
	}
	return nil
}

// Common errors.
var (
	// ErrInvalidConfig is returned for bad construction parameters.
	ErrInvalidConfig = errors.New("ann: invalid config")
	// ErrDimMismatch is returned when a vector has the wrong dimension.
	ErrDimMismatch = errors.New("ann: vector dimension mismatch")
	// ErrNotFound is returned when an ID is absent.
	ErrNotFound = errors.New("ann: id not found")
	// ErrEmptyIndex is returned when searching an empty index.
	ErrEmptyIndex = errors.New("ann: empty index")
	// ErrCorrupt is returned when persisted data fails validation. The index
	// is never returned in a partially-loaded state.
	ErrCorrupt = errors.New("ann: corrupt persisted index")
)

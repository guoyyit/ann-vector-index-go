// Command annbench builds a synthetic clustered vector set and reports the
// recall@k vs. search-cost trade-off for a range of efSearch values. It is a
// small, reproducible demonstration of how build/search parameters affect
// recall and the number of distance evaluations (visited nodes).
package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/guoyyit/ann-vector-index-go/ann"
)

func main() {
	n := flag.Int("n", 5000, "number of vectors")
	dim := flag.Int("dim", 32, "vector dimension")
	clusters := flag.Int("clusters", 20, "number of synthetic clusters")
	m := flag.Int("M", 16, "neighbor budget per node")
	seed := flag.Int64("seed", 42, "random seed for data and index")
	k := flag.Int("k", 10, "number of nearest neighbors to return")
	queries := flag.Int("queries", 100, "number of query vectors")
	flag.Parse()

	rng := rand.New(rand.NewSource(*seed))
	centers := make([][]float32, *clusters)
	for c := range centers {
		v := make([]float32, *dim)
		for i := range v {
			v[i] = rng.Float32()*2 - 1
		}
		centers[c] = v
	}
	data := make([][]float32, *n)
	ids := make([]uint64, *n)
	for i := 0; i < *n; i++ {
		c := rng.Intn(*clusters)
		v := make([]float32, *dim)
		for j := range v {
			v[j] = centers[c][j] + float32(rng.NormFloat64()*0.1)
		}
		data[i] = v
		ids[i] = uint64(i + 1)
	}
	qvecs := make([][]float32, *queries)
	for i := range qvecs {
		c := rng.Intn(*clusters)
		v := make([]float32, *dim)
		for j := range v {
			v[j] = centers[c][j] + float32(rng.NormFloat64()*0.1)
		}
		qvecs[i] = v
	}

	cfg := ann.DefaultConfig(*dim)
	cfg.M = *m
	cfg.Seed = uint64(*seed)

	buildStart := time.Now()
	idx, err := ann.New(cfg)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	for i := range data {
		if err := idx.Insert(ids[i], data[i]); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
	if err := idx.Verify(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	buildDur := time.Since(buildStart)
	st := idx.Stats()
	fmt.Printf("index: n=%d dim=%d M=%d seed=%d layers=%v build=%v\n",
		*n, *dim, *m, *seed, st.LayerSize, buildDur)
	fmt.Printf("%6s  %8s  %10s  %12s\n", "ef", "recall@k", "visited", "latency/us")

	efs := []int{1, 2, 4, 8, 16, 32, 64, 128, 256}
	for _, ef := range efs {
		start := time.Now()
		rep, err := idx.Evaluate(qvecs, *k, ef)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		dur := time.Since(start)
		perQuery := dur / time.Duration(*queries)
		fmt.Printf("%6d  %8.4f  %10.1f  %12d\n", ef, rep.Recall, rep.AvgVisited, perQuery.Microseconds())
	}
}

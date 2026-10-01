package ann

import "errors"

// 索引基础错误。
var (
	// ErrInvalidConfig 表示构建/查询参数非法。
	ErrInvalidConfig = errors.New("ann: invalid config")
	// ErrDimensionMismatch 表示向量维度与索引配置不一致。
	ErrDimensionMismatch = errors.New("ann: vector dimension mismatch")
	// ErrEmptyVector 表示插入了空向量。
	ErrEmptyVector = errors.New("ann: empty vector")
	// ErrNotFound 表示目标 ID 不存在。
	ErrNotFound = errors.New("ann: id not found")
	// ErrK 表示 k <= 0。
	ErrK = errors.New("ann: k must be positive")
	// ErrCorrupt 表示持久化文件校验失败或结构损坏。
	ErrCorrupt = errors.New("ann: corrupted index file")
	// ErrClosed 表示索引已关闭。
	ErrClosed = errors.New("ann: index closed")
)

// CorruptionPolicy 定义持久化文件损坏时的处理策略。
type CorruptionPolicy int

const (
	// CorruptionFail 损坏即返回错误（默认）。
	CorruptionFail CorruptionPolicy = iota
	// CorruptionIgnoreJournal 忽略损坏的预写日志，仅用快照恢复。
	CorruptionIgnoreJournal
	// CorruptionRebuild 无法恢复时重建空索引。
	CorruptionRebuild
)

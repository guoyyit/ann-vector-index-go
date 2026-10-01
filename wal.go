package ann

import (
	"encoding/binary"
	"errors"
	"hash/crc32"
	"io"
	"os"
)

// walFrame 是一条预写日志记录。
type walFrame struct {
	op  byte
	id  int64
	vec []float64 // insert 时有效
}

const (
	walOpInsert byte = 1
	walOpDelete byte = 2
)

const walMagic = "ANNWAL01"

// journal 是预写日志句柄。文件布局：
//
//	magic(8) | frame*
//	frame = len(4 LE) | payload | crc32(payload)(4 LE)
//	payload = op(1) | id(8 LE) | ndim(4 LE) | vec(ndim*8 LE)
//
// 长度前缀 + CRC 使崩溃造成的截断帧与内容损坏都能被精确定位。
type journal struct {
	path string
	f    *os.File
}

// openJournal 打开（不存在则创建）日志，校验并回放全部完整帧。
// 若文件尾部存在不完整/损坏帧，按 policy 处理：
// CorruptionFail 返回错误；IgnoreJournal 截掉坏尾部后回放前缀；
// Rebuild 清空整个日志。
func openJournal(path string, policy CorruptionPolicy, apply func(walFrame) error) (*journal, error) {
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o644)
	if err != nil {
		return nil, err
	}
	data, err := io.ReadAll(f)
	if err != nil {
		f.Close()
		return nil, err
	}
	j := &journal{path: path, f: f}

	off := 0
	good := 0
	if len(data) >= len(walMagic) {
		if string(data[:len(walMagic)]) != walMagic {
			switch policy {
			case CorruptionRebuild:
				return j.reset()
			default:
				f.Close()
				return nil, ErrCorrupt
			}
		}
		off = len(walMagic)
		good = off
	} else if len(data) > 0 {
		// 只有半个 magic。
		return j.handleTail(policy, good, apply, f, true)
	}

	corrupt := false
	for off < len(data) {
		frame, n, perr := parseFrame(data[off:])
		if perr != nil {
			corrupt = true
			break
		}
		if err := apply(frame); err != nil {
			f.Close()
			return nil, err
		}
		off += n
		good = off
	}
	if corrupt {
		return j.handleTail(policy, good, apply, f, false)
	}
	// 文件是全新的：补写 magic。
	if len(data) == 0 {
		if _, err := f.WriteString(walMagic); err != nil {
			f.Close()
			return nil, err
		}
		if err := f.Sync(); err != nil {
			f.Close()
			return nil, err
		}
	}
	return j, nil
}

func (j *journal) handleTail(policy CorruptionPolicy, good int, apply func(walFrame) error, f *os.File, magicBad bool) (*journal, error) {
	switch policy {
	case CorruptionFail:
		f.Close()
		return nil, ErrCorrupt
	case CorruptionIgnoreJournal:
		if err := f.Truncate(int64(good)); err != nil {
			f.Close()
			return nil, err
		}
		if _, err := f.Seek(0, io.SeekStart); err != nil {
			f.Close()
			return nil, err
		}
		if good == 0 {
			// 连 magic 都损坏了：截断到 0 后必须补写 magic，
			// 否则后续 append 会产生没有文件头的日志。
			if _, err := f.WriteString(walMagic); err != nil {
				f.Close()
				return nil, err
			}
		}
		if err := f.Sync(); err != nil {
			f.Close()
			return nil, err
		}
		return j, nil
	case CorruptionRebuild:
		return j.reset()
	default:
		f.Close()
		return nil, ErrCorrupt
	}
}

// reset 清空日志，只保留 magic。
func (j *journal) reset() (*journal, error) {
	if err := j.f.Truncate(0); err != nil {
		j.f.Close()
		return nil, err
	}
	if _, err := j.f.Seek(0, io.SeekStart); err != nil {
		j.f.Close()
		return nil, err
	}
	if _, err := j.f.WriteString(walMagic); err != nil {
		j.f.Close()
		return nil, err
	}
	if err := j.f.Sync(); err != nil {
		j.f.Close()
		return nil, err
	}
	return j, nil
}

var errShortFrame = errors.New("short frame")

// parseFrame 从 buf 解析一帧，返回帧与占用字节数。
func parseFrame(buf []byte) (walFrame, int, error) {
	if len(buf) < 4 {
		return walFrame{}, 0, errShortFrame
	}
	n := int(binary.LittleEndian.Uint32(buf[:4]))
	total := 4 + n + 4
	if n < 13 || total > len(buf) {
		return walFrame{}, 0, ErrCorrupt
	}
	payload := buf[4 : 4+n]
	want := binary.LittleEndian.Uint32(buf[4+n : 4+n+4])
	if crc32.ChecksumIEEE(payload) != want {
		return walFrame{}, 0, ErrCorrupt
	}
	op := payload[0]
	id := int64(binary.LittleEndian.Uint64(payload[1:9]))
	ndim := int(binary.LittleEndian.Uint32(payload[9:13]))
	if 13+ndim*8 != n {
		return walFrame{}, 0, ErrCorrupt
	}
	fr := walFrame{op: op, id: id}
	if op == walOpInsert {
		fr.vec = make([]float64, ndim)
		for i := range fr.vec {
			bits := binary.LittleEndian.Uint64(payload[13+i*8 : 13+(i+1)*8])
			fr.vec[i] = float64FromBits(bits)
		}
	} else if op != walOpDelete || ndim != 0 {
		return walFrame{}, 0, ErrCorrupt
	}
	return fr, total, nil
}

// append 写入一帧并 fsync（先写数据与 CRC，后刷盘，保证崩溃后要么
// 完整可见，要么根本不存在/被判为损坏帧，绝不会被当成好数据）。
func (j *journal) append(fr walFrame) error {
	ndim := 0
	if fr.op == walOpInsert {
		ndim = len(fr.vec)
	}
	n := 13 + ndim*8
	payload := make([]byte, n)
	payload[0] = fr.op
	binary.LittleEndian.PutUint64(payload[1:9], uint64(fr.id))
	binary.LittleEndian.PutUint32(payload[9:13], uint32(ndim))
	for i, v := range fr.vec {
		binary.LittleEndian.PutUint64(payload[13+i*8:13+(i+1)*8], float64Bits(v))
	}
	var hdr [4]byte
	binary.LittleEndian.PutUint32(hdr[:], uint32(n))
	var crc [4]byte
	binary.LittleEndian.PutUint32(crc[:], crc32.ChecksumIEEE(payload))
	if _, err := j.f.Write(hdr[:]); err != nil {
		return err
	}
	if _, err := j.f.Write(payload); err != nil {
		return err
	}
	if _, err := j.f.Write(crc[:]); err != nil {
		return err
	}
	return j.f.Sync()
}

// truncate 压缩日志（快照完成后调用），保留 magic。
func (j *journal) truncate() error {
	if err := j.f.Truncate(0); err != nil {
		return err
	}
	if _, err := j.f.Seek(0, io.SeekStart); err != nil {
		return err
	}
	if _, err := j.f.WriteString(walMagic); err != nil {
		return err
	}
	return j.f.Sync()
}

// Close 刷盘并关闭。
func (j *journal) Close() error {
	if err := j.f.Sync(); err != nil {
		j.f.Close()
		return err
	}
	return j.f.Close()
}

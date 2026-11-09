package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"io"
	"log"
	"os"
	"path/filepath"
)

type memoryJournalEnvelope struct {
	Version  int             `json:"version"`
	Checksum uint32          `json:"checksum"`
	Record   json.RawMessage `json:"record"`
}

func (ms *MemoryStore) loadSnapshot() error {
	info, err := os.Stat(ms.filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("inspect memory store: %w", err)
	}
	if info.Size() > ms.maxBytes {
		return fmt.Errorf("memory file is %d bytes; configured maximum is %d bytes", info.Size(), ms.maxBytes)
	}
	data, err := os.ReadFile(ms.filePath)
	if err != nil {
		return fmt.Errorf("read memory store: %w", err)
	}
	if int64(len(data)) > ms.maxBytes {
		return fmt.Errorf("memory file grew to %d bytes; configured maximum is %d bytes", len(data), ms.maxBytes)
	}
	if len(data) == 0 {
		return nil
	}
	if err := json.Unmarshal(data, &ms.memories); err != nil {
		return fmt.Errorf("parse memory store: %w", err)
	}
	return nil
}

func (ms *MemoryStore) replayJournal() error {
	data, err := os.ReadFile(ms.journalPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("read memory journal: %w", err)
	}
	if int64(len(data)) > ms.maxBytes {
		return fmt.Errorf("memory journal is %d bytes; configured maximum is %d bytes", len(data), ms.maxBytes)
	}
	lastNewline := bytes.LastIndexByte(data, '\n')
	complete := data[:lastNewline+1]
	if len(complete) != len(data) {
		if err := os.Truncate(ms.journalPath, int64(len(complete))); err != nil {
			return fmt.Errorf("truncate incomplete memory journal tail: %w", err)
		}
	}
	lines := bytes.Split(complete, []byte{'\n'})
	for lineNumber, line := range lines {
		if len(line) == 0 {
			continue
		}
		var envelope memoryJournalEnvelope
		if err := json.Unmarshal(line, &envelope); err != nil {
			return fmt.Errorf("parse memory journal record %d: %w", lineNumber+1, err)
		}
		if envelope.Version != memoryJournalVersion {
			return fmt.Errorf("unsupported memory journal version %d in record %d", envelope.Version, lineNumber+1)
		}
		if crc32.ChecksumIEEE(envelope.Record) != envelope.Checksum {
			return fmt.Errorf("checksum mismatch in memory journal record %d", lineNumber+1)
		}
		var record memoryJournalRecord
		if err := json.Unmarshal(envelope.Record, &record); err != nil {
			return fmt.Errorf("decode memory journal payload %d: %w", lineNumber+1, err)
		}
		if record.Version != envelope.Version {
			return fmt.Errorf("journal envelope and payload versions differ in record %d", lineNumber+1)
		}
		if record.Clear {
			ms.memories = make([]Memory, 0)
		}
		if len(record.Delete) > 0 {
			deleted := make(map[string]struct{}, len(record.Delete))
			for _, id := range record.Delete {
				deleted[id] = struct{}{}
			}
			remaining := ms.memories[:0]
			for _, memory := range ms.memories {
				if _, ok := deleted[memory.ID]; !ok {
					remaining = append(remaining, memory)
				}
			}
			ms.memories = remaining
		}
		for _, memory := range record.Upsert {
			if memory.ID == "" || len(memory.Content) > ms.maxEntryBytes {
				return fmt.Errorf("invalid memory in journal record %d", lineNumber+1)
			}
			memory.Category = normalizeCategory(memory.Category)
			memory.Importance = normalizeImportance(memory.Importance)
			memory.Embedding = localEmbedding(memory.Content)
			updated := false
			for index := range ms.memories {
				if ms.memories[index].ID == memory.ID {
					ms.memories[index] = memory
					updated = true
					break
				}
			}
			if !updated {
				ms.memories = append(ms.memories, memory)
			}
		}
		if len(ms.memories) > ms.maxItems {
			return fmt.Errorf("memory journal exceeds configured item maximum %d", ms.maxItems)
		}
	}
	ms.journalBytes = int64(len(complete))
	ms.journalRecords = len(lines) - 1
	return nil
}

func (ms *MemoryStore) appendJournalLocked(record memoryJournalRecord) error {
	if ms.filePath == "" {
		return nil
	}
	record.Version = memoryJournalVersion
	payload, err := json.Marshal(record)
	if err != nil {
		return fmt.Errorf("encode memory journal payload: %w", err)
	}
	data, err := json.Marshal(memoryJournalEnvelope{
		Version:  memoryJournalVersion,
		Checksum: crc32.ChecksumIEEE(payload),
		Record:   payload,
	})
	if err != nil {
		return fmt.Errorf("encode memory journal record: %w", err)
	}
	data = append(data, '\n')
	if int64(len(data)) > ms.maxBytes {
		return fmt.Errorf("memory journal record exceeds configured maximum of %d bytes", ms.maxBytes)
	}
	if err := os.MkdirAll(filepath.Dir(ms.filePath), 0o700); err != nil {
		return fmt.Errorf("create memory directory: %w", err)
	}
	if ms.journalBytes+int64(len(data)) > ms.maxBytes {
		return fmt.Errorf("memory journal would exceed configured maximum of %d bytes", ms.maxBytes)
	}
	file, err := os.OpenFile(ms.journalPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return fmt.Errorf("open memory journal: %w", err)
	}
	written, writeErr := file.Write(data)
	if writeErr == nil && written != len(data) {
		writeErr = io.ErrShortWrite
	}
	if writeErr == nil {
		writeErr = file.Sync()
	}
	closeErr := file.Close()
	if writeErr != nil {
		if truncateErr := os.Truncate(ms.journalPath, ms.journalBytes); truncateErr != nil {
			return fmt.Errorf("append memory journal: %v; restore journal tail: %w", writeErr, truncateErr)
		}
		return fmt.Errorf("append memory journal: %w", writeErr)
	}
	if closeErr != nil {
		if truncateErr := os.Truncate(ms.journalPath, ms.journalBytes); truncateErr != nil {
			return fmt.Errorf("close memory journal: %v; restore journal tail: %w", closeErr, truncateErr)
		}
		return fmt.Errorf("close memory journal: %w", closeErr)
	}
	ms.journalRecords++
	ms.journalBytes += int64(len(data))
	if ms.journalRecords >= memoryJournalCheckpointRecords || ms.journalBytes >= memoryJournalCheckpointBytes {
		if err := ms.checkpointLocked(); err != nil {
			log.Printf("memory journal checkpoint deferred: %v", err)
		}
	}
	return nil
}

func (ms *MemoryStore) checkpointLocked() error {
	if ms.filePath == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(ms.filePath), 0o700); err != nil {
		return fmt.Errorf("create memory directory: %w", err)
	}
	data, err := json.MarshalIndent(ms.memories, "", "  ")
	if err != nil {
		return fmt.Errorf("encode memory snapshot: %w", err)
	}
	if int64(len(data)) > ms.maxBytes {
		return fmt.Errorf("memory snapshot would be %d bytes; configured maximum is %d bytes", len(data), ms.maxBytes)
	}
	temporaryPath := ms.filePath + ".tmp"
	file, err := os.OpenFile(temporaryPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
	if err != nil {
		return fmt.Errorf("open memory snapshot: %w", err)
	}
	written, writeErr := file.Write(data)
	if writeErr == nil && written != len(data) {
		writeErr = io.ErrShortWrite
	}
	if writeErr == nil {
		writeErr = file.Sync()
	}
	closeErr := file.Close()
	if writeErr != nil {
		_ = os.Remove(temporaryPath)
		return fmt.Errorf("write memory snapshot: %w", writeErr)
	}
	if closeErr != nil {
		_ = os.Remove(temporaryPath)
		return fmt.Errorf("close memory snapshot: %w", closeErr)
	}
	if err := os.Rename(temporaryPath, ms.filePath); err != nil {
		_ = os.Remove(temporaryPath)
		return fmt.Errorf("replace memory snapshot: %w", err)
	}
	if directory, err := os.Open(filepath.Dir(ms.filePath)); err == nil {
		_ = directory.Sync()
		_ = directory.Close()
	}
	journal, err := os.OpenFile(ms.journalPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
	if err != nil {
		return fmt.Errorf("truncate checkpointed memory journal: %w", err)
	}
	if err := journal.Sync(); err != nil {
		_ = journal.Close()
		return fmt.Errorf("sync checkpointed memory journal: %w", err)
	}
	if err := journal.Close(); err != nil {
		return fmt.Errorf("close checkpointed memory journal: %w", err)
	}
	ms.journalRecords = 0
	ms.journalBytes = 0
	return nil
}

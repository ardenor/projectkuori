package main

import (
	"fmt"
	"hash/fnv"
	"math"
	"sort"
	"strings"
	"sync"
	"time"
)

// Memory is one persistent thought node and its derived, in-memory search vector.
type Memory struct {
	ID         string    `json:"id"`
	Content    string    `json:"content"`
	Timestamp  time.Time `json:"timestamp"`
	Category   string    `json:"category,omitempty"`
	Importance string    `json:"importance,omitempty"`
	RelatedIDs []string  `json:"related_ids,omitempty"`
	Embedding  []float32 `json:"-"`
}

// MemoryStore protects saved nodes and enforces disk, count, and entry-size caps.
type MemoryStore struct {
	sync.RWMutex
	memories        []Memory
	searchPostings  map[uint64]map[string]struct{}
	searchFeatures  map[string][]uint64
	searchPositions map[string]int
	filePath        string
	journalPath     string
	journalRecords  int
	journalBytes    int64
	maxBytes        int64
	maxItems        int
	maxEntryBytes   int
}

// GlobalMemoryStore is initialized from disk by main before the server accepts clients.
var GlobalMemoryStore *MemoryStore

// BehavioralMetrics summarizes recent memory text for low-cost response context.
type BehavioralMetrics struct {
	StressScore    int
	LateNightCount int
	RoutineStatus  string
}

const embeddingDimensions = 512
const memoryJournalVersion = 1
const memoryJournalCheckpointRecords = 64
const memoryJournalCheckpointBytes = 256 * 1024

type memoryJournalRecord struct {
	Version int      `json:"version"`
	Upsert  []Memory `json:"upsert,omitempty"`
	Delete  []string `json:"delete,omitempty"`
	Clear   bool     `json:"clear,omitempty"`
}

type MemoryLimits struct {
	MaxBytes      int64
	MaxItems      int
	MaxEntryBytes int
}

// DefaultMemoryLimits provide a small storage budget suitable for local development.
func DefaultMemoryLimits() MemoryLimits {
	return MemoryLimits{
		MaxBytes:      5 * 1024 * 1024,
		MaxItems:      1000,
		MaxEntryBytes: 8 * 1024,
	}
}

func NewMemoryStore(filePath string) (*MemoryStore, error) {
	return NewMemoryStoreWithLimits(filePath, DefaultMemoryLimits())
}

// NewMemoryStoreWithLimits loads a saved file only after checking its configured size caps.
func NewMemoryStoreWithLimits(filePath string, limits MemoryLimits) (*MemoryStore, error) {
	if limits.MaxBytes <= 0 || limits.MaxItems <= 0 || limits.MaxEntryBytes <= 0 {
		return nil, fmt.Errorf("memory limits must all be positive")
	}
	store := &MemoryStore{
		memories:        make([]Memory, 0),
		searchPostings:  make(map[uint64]map[string]struct{}),
		searchFeatures:  make(map[string][]uint64),
		searchPositions: make(map[string]int),
		filePath:        filePath,
		journalPath:     filePath + ".wal",
		maxBytes:        limits.MaxBytes,
		maxItems:        limits.MaxItems,
		maxEntryBytes:   limits.MaxEntryBytes,
	}
	if filePath != "" {
		if err := store.loadSnapshot(); err != nil {
			return nil, err
		}
		if err := store.replayJournal(); err != nil {
			return nil, err
		}
	}
	if len(store.memories) > store.maxItems {
		return nil, fmt.Errorf("memory store has %d records; configured maximum is %d", len(store.memories), store.maxItems)
	}
	// Rebuild derived search vectors after loading both snapshot and journal.
	for i := range store.memories {
		if store.memories[i].ID == "" {
			return nil, fmt.Errorf("memory record has an empty ID")
		}
		if len(store.memories[i].Content) > store.maxEntryBytes {
			return nil, fmt.Errorf("memory record %q exceeds the configured %d-byte entry limit", store.memories[i].ID, store.maxEntryBytes)
		}
		store.memories[i].Category = normalizeCategory(store.memories[i].Category)
		store.memories[i].Importance = normalizeImportance(store.memories[i].Importance)
		store.memories[i].Embedding = localEmbedding(store.memories[i].Content)
		store.addSearchIndexLocked(store.memories[i], i)
	}
	return store, nil
}

func localEmbedding(text string) []float32 {
	// Hash words, adjacent word pairs, and character n-grams into a normalized typo-tolerant vector.
	vector := make([]float32, embeddingDimensions)
	words := strings.Fields(strings.ToLower(text))
	for i, word := range words {
		addFeature(vector, word)
		addCharacterNgrams(vector, word)
		if i > 0 {
			addFeature(vector, words[i-1]+" "+word)
		}
	}
	var norm float64
	for _, value := range vector {
		norm += float64(value * value)
	}
	if norm > 0 {
		norm = math.Sqrt(norm)
		for i := range vector {
			vector[i] /= float32(norm)
		}
	}
	return vector
}

func addCharacterNgrams(vector []float32, word string) {
	characters := []rune(word)
	for width := 2; width <= 3; width++ {
		for i := 0; i+width <= len(characters); i++ {
			addFeature(vector, fmt.Sprintf("char%d:%s", width, string(characters[i:i+width])))
		}
	}
}

func textFeatureHashes(text string) ([]uint64, []uint64) {
	words := strings.Fields(strings.ToLower(text))
	exactFeatures := make(map[uint64]struct{}, len(words)*2)
	fuzzyFeatures := make(map[uint64]struct{}, len(words)*4)
	for index, word := range words {
		exactFeatures[featureHash("word:"+word)] = struct{}{}
		characters := []rune(word)
		for width := 2; width <= 3; width++ {
			for start := 0; start+width <= len(characters); start++ {
				fuzzyFeatures[featureHash(fmt.Sprintf("char%d:%s", width, string(characters[start:start+width])))] = struct{}{}
			}
		}
		if index > 0 {
			exactFeatures[featureHash("pair:"+words[index-1]+" "+word)] = struct{}{}
		}
	}
	toSlice := func(features map[uint64]struct{}) []uint64 {
		hashes := make([]uint64, 0, len(features))
		for hash := range features {
			hashes = append(hashes, hash)
		}
		return hashes
	}
	return toSlice(exactFeatures), toSlice(fuzzyFeatures)
}

func featureHash(feature string) uint64 {
	hasher := fnv.New64a()
	_, _ = hasher.Write([]byte(feature))
	return hasher.Sum64()
}

func (ms *MemoryStore) addSearchIndexLocked(memory Memory, position int) {
	exactFeatures, fuzzyFeatures := textFeatureHashes(memory.Content)
	features := append(exactFeatures, fuzzyFeatures...)
	ms.searchFeatures[memory.ID] = features
	ms.searchPositions[memory.ID] = position
	for _, feature := range features {
		if ms.searchPostings[feature] == nil {
			ms.searchPostings[feature] = make(map[string]struct{})
		}
		ms.searchPostings[feature][memory.ID] = struct{}{}
	}
}

func (ms *MemoryStore) removeSearchIndexLocked(memoryID string) {
	for _, feature := range ms.searchFeatures[memoryID] {
		delete(ms.searchPostings[feature], memoryID)
		if len(ms.searchPostings[feature]) == 0 {
			delete(ms.searchPostings, feature)
		}
	}
	delete(ms.searchFeatures, memoryID)
	delete(ms.searchPositions, memoryID)
}

func (ms *MemoryStore) rebuildSearchPositionsLocked() {
	clear(ms.searchPositions)
	for index, memory := range ms.memories {
		ms.searchPositions[memory.ID] = index
	}
}

func (ms *MemoryStore) searchCandidatePositionsLocked(query string) []int {
	candidateIDs := make(map[string]struct{})
	exactFeatures, fuzzyFeatures := textFeatureHashes(query)
	for _, feature := range exactFeatures {
		for id := range ms.searchPostings[feature] {
			candidateIDs[id] = struct{}{}
		}
	}
	fuzzyPostingLimit := max(32, len(ms.memories)/8)
	for _, feature := range fuzzyFeatures {
		posting := ms.searchPostings[feature]
		if len(posting) > fuzzyPostingLimit {
			continue
		}
		for id := range posting {
			candidateIDs[id] = struct{}{}
		}
	}
	positions := make([]int, 0, len(candidateIDs))
	for id := range candidateIDs {
		if position, ok := ms.searchPositions[id]; ok {
			positions = append(positions, position)
		}
	}
	sort.Ints(positions)
	return positions
}

// addFeature maps a text feature into the fixed-size vector using a stable hash.
func addFeature(vector []float32, feature string) {
	hasher := fnv.New32a()
	_, _ = hasher.Write([]byte(feature))
	index := hasher.Sum32() % uint32(len(vector))
	vector[index] += 1
}

func cosineSimilarity(v1, v2 []float32) float32 {
	if len(v1) != len(v2) {
		return 0
	}
	var dotProduct, normA, normB float32
	for i := range v1 {
		dotProduct += v1[i] * v2[i]
		normA += v1[i] * v1[i]
		normB += v2[i] * v2[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

// AddMemory appends a node only if every configured cap and disk write succeeds.
func (ms *MemoryStore) AddMemory(content string) (Memory, error) {
	return ms.AddMemoryWithMetadata(content, "other", "medium", nil)
}

func (ms *MemoryStore) AddMemoryWithMetadata(content, category, importance string, relatedIDs []string) (Memory, error) {
	content = strings.TrimSpace(content)
	if content == "" {
		return Memory{}, fmt.Errorf("memory content cannot be empty")
	}
	if len(content) > ms.maxEntryBytes {
		return Memory{}, fmt.Errorf("memory entry is %d bytes; configured maximum is %d bytes", len(content), ms.maxEntryBytes)
	}
	ms.Lock()
	defer ms.Unlock()
	if len(ms.memories) >= ms.maxItems {
		return Memory{}, fmt.Errorf("memory node limit reached (%d); no data was changed", ms.maxItems)
	}
	memory := Memory{
		ID:         fmt.Sprintf("mem-%d", time.Now().UnixNano()),
		Content:    content,
		Timestamp:  time.Now().UTC(),
		Category:   normalizeCategory(category),
		Importance: normalizeImportance(importance),
		Embedding:  localEmbedding(content),
	}

	requestedRelations := make(map[string]bool, len(relatedIDs))
	for _, id := range relatedIDs {
		if id != "" && id != memory.ID {
			requestedRelations[id] = true
		}
	}
	nextMemories := make([]Memory, len(ms.memories)+1)
	upserts := []Memory{memory}
	for i, existing := range ms.memories {
		existing.RelatedIDs = append([]string(nil), existing.RelatedIDs...)
		if requestedRelations[existing.ID] {
			memory.RelatedIDs = appendUniqueID(memory.RelatedIDs, existing.ID)
			existing.RelatedIDs = appendUniqueID(existing.RelatedIDs, memory.ID)
		}
		nextMemories[i] = existing
		if requestedRelations[existing.ID] {
			upserts = append(upserts, existing)
		}
	}
	upserts[0] = memory
	previousMemories := ms.memories
	nextMemories[len(ms.memories)] = memory
	ms.memories = nextMemories
	if err := ms.appendJournalLocked(memoryJournalRecord{Upsert: upserts}); err != nil {
		ms.memories = previousMemories
		return Memory{}, err
	}
	ms.addSearchIndexLocked(memory, len(ms.memories)-1)
	return memory, nil
}

func normalizeCategory(category string) string {
	switch strings.ToLower(strings.TrimSpace(category)) {
	case "health", "work", "relationships", "goals", "preferences", "home":
		return strings.ToLower(strings.TrimSpace(category))
	default:
		return "other"
	}
}

func normalizeImportance(importance string) string {
	switch strings.ToLower(strings.TrimSpace(importance)) {
	case "low", "high":
		return strings.ToLower(strings.TrimSpace(importance))
	default:
		return "medium"
	}
}

func appendUniqueID(ids []string, id string) []string {
	for _, existingID := range ids {
		if existingID == id {
			return ids
		}
	}
	return append(ids, id)
}

func (ms *MemoryStore) UpdateMemory(id, addition string) (Memory, error) {
	// Keep a copy so a size or persistence failure can restore the original record.
	addition = strings.TrimSpace(addition)
	ms.Lock()
	defer ms.Unlock()
	for i := range ms.memories {
		if ms.memories[i].ID != id {
			continue
		}
		previous := ms.memories[i]
		ms.memories[i].Content = strings.TrimSpace(previous.Content + "\n" + addition)
		if len(ms.memories[i].Content) > ms.maxEntryBytes {
			ms.memories[i] = previous
			return Memory{}, fmt.Errorf("updated memory would exceed the configured %d-byte entry limit", ms.maxEntryBytes)
		}
		ms.memories[i].Timestamp = time.Now().UTC()
		ms.memories[i].Embedding = localEmbedding(ms.memories[i].Content)
		if err := ms.appendJournalLocked(memoryJournalRecord{Upsert: []Memory{ms.memories[i]}}); err != nil {
			ms.memories[i] = previous
			return Memory{}, err
		}
		ms.removeSearchIndexLocked(id)
		ms.addSearchIndexLocked(ms.memories[i], i)
		return ms.memories[i], nil
	}
	return Memory{}, fmt.Errorf("memory %q not found", id)
}

func (ms *MemoryStore) ReplaceMemory(id, content string) (Memory, error) {
	content = strings.TrimSpace(content)
	if content == "" {
		return Memory{}, fmt.Errorf("memory content cannot be empty")
	}
	if len(content) > ms.maxEntryBytes {
		return Memory{}, fmt.Errorf("memory entry is %d bytes; configured maximum is %d bytes", len(content), ms.maxEntryBytes)
	}
	ms.Lock()
	defer ms.Unlock()
	for i := range ms.memories {
		if ms.memories[i].ID != id {
			continue
		}
		previous := ms.memories[i]
		ms.memories[i].Content = content
		ms.memories[i].Timestamp = time.Now().UTC()
		ms.memories[i].Embedding = localEmbedding(content)
		if err := ms.appendJournalLocked(memoryJournalRecord{Upsert: []Memory{ms.memories[i]}}); err != nil {
			ms.memories[i] = previous
			return Memory{}, err
		}
		ms.removeSearchIndexLocked(id)
		ms.addSearchIndexLocked(ms.memories[i], i)
		return ms.memories[i], nil
	}
	return Memory{}, fmt.Errorf("memory %q not found", id)
}

type MemoryMatch struct {
	Memory Memory
	Score  float32
}

func (ms *MemoryStore) FindRelevantMemories(query string, limit int) []MemoryMatch {
	ms.RLock()
	defer ms.RUnlock()
	if len(ms.memories) == 0 || limit <= 0 {
		return nil
	}
	searchText := expandMemoryQuery(query)
	queryVec := localEmbedding(searchText)
	minimumScore := float32(0.18)
	if searchText != query {
		minimumScore = 0.1
	}
	candidatePositions := ms.searchCandidatePositionsLocked(searchText)
	matches := make([]MemoryMatch, 0, len(candidatePositions))

	for _, position := range candidatePositions {
		mem := ms.memories[position]
		score := cosineSimilarity(queryVec, mem.Embedding)
		if score >= minimumScore {
			matches = append(matches, MemoryMatch{Memory: mem, Score: score})
		}
	}
	sort.Slice(matches, func(i, j int) bool { return matches[i].Score > matches[j].Score })
	if len(matches) > limit {
		matches = matches[:limit]
	}
	return matches
}

func expandMemoryQuery(query string) string {
	lower := strings.ToLower(strings.TrimSpace(query))
	for _, phrase := range []string{"what is my name", "what's my name", "what is your name", "who am i", "what should you call me", "what do you call me", "what did i ask you to call me", "what name should you use"} {
		if strings.Contains(lower, phrase) {
			return query + " name call me my name is refer to me as"
		}
	}
	return query
}

func (ms *MemoryStore) FindRelevantMemory(query string) (Memory, float32, bool) {
	matches := ms.FindRelevantMemories(query, 1)
	if len(matches) == 0 {
		return Memory{}, 0, false
	}
	return matches[0].Memory, matches[0].Score, true
}

func (ms *MemoryStore) SearchRelevantMemory(query string) string {
	memory, _, found := ms.FindRelevantMemory(query)
	if found {
		return memory.Content
	}
	return ""
}

// List returns a snapshot so callers cannot mutate the store's backing slice.
func (ms *MemoryStore) List() []Memory {
	ms.RLock()
	defer ms.RUnlock()
	memories := make([]Memory, len(ms.memories))
	copy(memories, ms.memories)
	for i := range memories {
		memories[i].RelatedIDs = append([]string(nil), memories[i].RelatedIDs...)
	}
	return memories
}

func (ms *MemoryStore) DeleteMemories(ids []string) ([]Memory, error) {
	return ms.DeleteMemoriesIfCount(ids, -1)
}

func (ms *MemoryStore) DeleteMemoriesIfCount(ids []string, expectedCount int) ([]Memory, error) {
	ms.Lock()
	defer ms.Unlock()
	requested := make(map[string]bool, len(ids))
	for _, id := range ids {
		if id != "" {
			requested[id] = true
		}
	}
	if len(requested) == 0 {
		return nil, fmt.Errorf("no memory IDs were selected")
	}

	deleted := make([]Memory, 0, len(requested))
	nextMemories := make([]Memory, 0, len(ms.memories))
	upserts := make([]Memory, 0)
	for _, memory := range ms.memories {
		if requested[memory.ID] {
			deleted = append(deleted, memory)
			continue
		}
		copyOfMemory := memory
		copyOfMemory.RelatedIDs = make([]string, 0, len(memory.RelatedIDs))
		for _, relatedID := range memory.RelatedIDs {
			if !requested[relatedID] {
				copyOfMemory.RelatedIDs = append(copyOfMemory.RelatedIDs, relatedID)
			}
		}
		if len(copyOfMemory.RelatedIDs) != len(memory.RelatedIDs) {
			upserts = append(upserts, copyOfMemory)
		}
		nextMemories = append(nextMemories, copyOfMemory)
	}
	if len(deleted) == 0 {
		return nil, fmt.Errorf("no selected memories were found")
	}
	if expectedCount >= 0 && len(deleted) != expectedCount {
		return nil, fmt.Errorf("selection changed before deletion; expected %d nodes but found %d", expectedCount, len(deleted))
	}

	deleteIDs := make([]string, 0, len(deleted))
	for _, memory := range deleted {
		deleteIDs = append(deleteIDs, memory.ID)
	}
	previousMemories := ms.memories
	ms.memories = nextMemories
	if err := ms.appendJournalLocked(memoryJournalRecord{Upsert: upserts, Delete: deleteIDs}); err != nil {
		ms.memories = previousMemories
		return nil, err
	}
	for _, memory := range deleted {
		ms.removeSearchIndexLocked(memory.ID)
	}
	ms.rebuildSearchPositionsLocked()
	return deleted, nil
}

func (ms *MemoryStore) DeleteAllMemories() ([]Memory, error) {
	return ms.DeleteAllMemoriesIfCount(-1)
}

func (ms *MemoryStore) DeleteAllMemoriesIfCount(expectedCount int) ([]Memory, error) {
	ms.Lock()
	defer ms.Unlock()
	if expectedCount >= 0 && len(ms.memories) != expectedCount {
		return nil, fmt.Errorf("memory count changed before clear; expected %d nodes but found %d", expectedCount, len(ms.memories))
	}
	if len(ms.memories) == 0 {
		return nil, nil
	}
	deleted := ms.memories
	ms.memories = make([]Memory, 0)
	if err := ms.appendJournalLocked(memoryJournalRecord{Clear: true}); err != nil {
		ms.memories = deleted
		return nil, err
	}
	ms.searchPostings = make(map[uint64]map[string]struct{})
	ms.searchFeatures = make(map[string][]uint64)
	ms.searchPositions = make(map[string]int)
	return deleted, nil
}

func (ms *MemoryStore) AnalyzePatterns() BehavioralMetrics {
	// Scan only the latest 15 records for simple stress-word and late-hour signals.
	ms.RLock()
	defer ms.RUnlock()

	metrics := BehavioralMetrics{RoutineStatus: "Stable"}
	stressKeywords := []string{"buried", "overwhelmed", "exhausted", "impossible", "stressed", "stuck", "tired"}

	startIdx := len(ms.memories) - 15
	if startIdx < 0 {
		startIdx = 0
	}

	for i := startIdx; i < len(ms.memories); i++ {
		mem := ms.memories[i]
		lowerContent := strings.ToLower(mem.Content)

		for _, word := range stressKeywords {
			if strings.Contains(lowerContent, word) {
				metrics.StressScore++
			}
		}

		hour := mem.Timestamp.Hour()
		if hour >= 22 || hour <= 4 {
			metrics.LateNightCount++
		}
	}

	if metrics.StressScore > 4 && metrics.LateNightCount > 2 {
		metrics.RoutineStatus = "High Burnout Risk (Linguistic Overload + Late-Night Fatigue)"
	} else if metrics.StressScore > 3 {
		metrics.RoutineStatus = "Elevated Stress Threshold Detected"
	} else if metrics.LateNightCount > 3 {
		metrics.RoutineStatus = "Irregular Circadian Disruption"
	}

	return metrics
}

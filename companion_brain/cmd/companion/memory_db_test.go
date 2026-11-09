package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"testing"
)

// TestMemoryStorePersistsAndUpdatesMemory covers create, retrieve, follow-up, and reload.
func TestMemoryStorePersistsAndUpdatesMemory(t *testing.T) {
	path := t.TempDir() + "/memories.json"
	store, err := NewMemoryStore(path)
	if err != nil {
		t.Fatal(err)
	}

	created, err := store.AddMemory("Work deadlines are making me tired")
	if err != nil {
		t.Fatal(err)
	}
	if created.ID == "" {
		t.Fatal("created memory has no ID")
	}

	match, score, found := store.FindRelevantMemory("Work deadlines are making me tired")
	if !found || match.ID != created.ID || score < 0.99 {
		t.Fatalf("expected exact local match, got found=%v score=%f memory=%+v", found, score, match)
	}

	updated, err := store.UpdateMemory(created.ID, "I finished one task this morning")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(updated.Content, "I finished one task this morning") {
		t.Fatalf("follow-up was not appended: %q", updated.Content)
	}

	reloaded, err := NewMemoryStore(path)
	if err != nil {
		t.Fatal(err)
	}
	memories := reloaded.List()
	if len(memories) != 1 || memories[0].ID != created.ID || memories[0].Content != updated.Content {
		t.Fatalf("unexpected persisted memories: %+v", memories)
	}
}

func TestMemoryJournalReplaysLegacySnapshotAndRepairsPartialTail(t *testing.T) {
	path := t.TempDir() + "/legacy.json"
	legacyData := `[{"id":"legacy-1","content":"I prefer tea","timestamp":"2026-11-01T00:00:00Z","category":"preferences","importance":"high"}]`
	if err := os.WriteFile(path, []byte(legacyData), 0o600); err != nil {
		t.Fatal(err)
	}
	store, err := NewMemoryStore(path)
	if err != nil {
		t.Fatal(err)
	}
	created, err := store.AddMemoryWithMetadata("I work near the river", "work", "medium", nil)
	if err != nil {
		t.Fatal(err)
	}

	snapshot, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var legacySnapshot []Memory
	if err := json.Unmarshal(snapshot, &legacySnapshot); err != nil {
		t.Fatal(err)
	}
	if len(legacySnapshot) != 1 || legacySnapshot[0].ID != "legacy-1" {
		t.Fatalf("mutation unexpectedly rewrote the legacy snapshot: %+v", legacySnapshot)
	}
	journalPath := path + ".wal"
	validJournal, err := os.ReadFile(journalPath)
	if err != nil {
		t.Fatal(err)
	}
	file, err := os.OpenFile(journalPath, os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := file.WriteString(`{"version":1,"upsert":[`); err != nil {
		_ = file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}

	reloaded, err := NewMemoryStore(path)
	if err != nil {
		t.Fatal(err)
	}
	memories := reloaded.List()
	if len(memories) != 2 || memories[1].ID != created.ID {
		t.Fatalf("journal replay lost a memory: %+v", memories)
	}
	repairedJournal, err := os.ReadFile(journalPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(repairedJournal) != string(validJournal) {
		t.Fatal("incomplete final journal record was not truncated during recovery")
	}
}

func TestMemoryJournalRejectsChecksumMismatch(t *testing.T) {
	path := t.TempDir() + "/checksum.json"
	store, err := NewMemoryStore(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.AddMemory("A journaled memory"); err != nil {
		t.Fatal(err)
	}
	journalPath := path + ".wal"
	data, err := os.ReadFile(journalPath)
	if err != nil {
		t.Fatal(err)
	}
	var envelope memoryJournalEnvelope
	if err := json.Unmarshal(bytes.TrimSpace(data), &envelope); err != nil {
		t.Fatal(err)
	}
	envelope.Checksum++
	corruptRecord, err := json.Marshal(envelope)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(journalPath, append(corruptRecord, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := NewMemoryStore(path); err == nil || !strings.Contains(err.Error(), "checksum mismatch") {
		t.Fatalf("expected a checksum mismatch error, got %v", err)
	}
}

func TestMemoryJournalAppendFailureRollsBackMutation(t *testing.T) {
	path := t.TempDir() + "/rollback.json"
	store, err := NewMemoryStore(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(path+".wal", 0o700); err != nil {
		t.Fatal(err)
	}
	if _, err := store.AddMemory("This write must roll back"); err == nil {
		t.Fatal("expected journal append to fail when the journal path is a directory")
	}
	if got := len(store.List()); got != 0 {
		t.Fatalf("failed journal append left %d in-memory records", got)
	}
}

func TestMemoryJournalCheckpointsAfterBoundedRecordCount(t *testing.T) {
	path := t.TempDir() + "/checkpoint.json"
	store, err := NewMemoryStore(path)
	if err != nil {
		t.Fatal(err)
	}
	for index := 0; index < memoryJournalCheckpointRecords; index++ {
		if _, err := store.AddMemory("A repeated test memory"); err != nil {
			t.Fatalf("add record %d: %v", index, err)
		}
	}

	journalInfo, err := os.Stat(path + ".wal")
	if err != nil {
		t.Fatal(err)
	}
	if journalInfo.Size() != 0 {
		t.Fatalf("journal should be compacted after %d records, got %d bytes", memoryJournalCheckpointRecords, journalInfo.Size())
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var snapshot []Memory
	if err := json.Unmarshal(data, &snapshot); err != nil {
		t.Fatal(err)
	}
	if len(snapshot) != memoryJournalCheckpointRecords {
		t.Fatalf("checkpoint contains %d records, want %d", len(snapshot), memoryJournalCheckpointRecords)
	}
	reloaded, err := NewMemoryStore(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(reloaded.List()) != memoryJournalCheckpointRecords {
		t.Fatalf("checkpoint reload has %d records", len(reloaded.List()))
	}
}

func TestMemoryStoreReplacesMemoryContent(t *testing.T) {
	path := t.TempDir() + "/replace.json"
	store, err := NewMemoryStore(path)
	if err != nil {
		t.Fatal(err)
	}
	created, err := store.AddMemoryWithMetadata("I prefer early meetings", "preferences", "high", nil)
	if err != nil {
		t.Fatal(err)
	}
	updated, err := store.ReplaceMemory(created.ID, "I prefer afternoon meetings")
	if err != nil {
		t.Fatal(err)
	}
	if updated.Content != "I prefer afternoon meetings" || updated.Category != "preferences" || updated.Importance != "high" {
		t.Fatalf("replacement did not preserve content and metadata: %+v", updated)
	}
	if strings.Contains(updated.Content, "early") {
		t.Fatalf("replacement unexpectedly appended old content: %q", updated.Content)
	}
	reloaded, err := NewMemoryStore(path)
	if err != nil {
		t.Fatal(err)
	}
	memories := reloaded.List()
	if len(memories) != 1 || memories[0].ID != created.ID || memories[0].Content != updated.Content {
		t.Fatalf("replacement did not persist: %+v", memories)
	}
}

func TestMemoryStoreDeletesNodesAndCleansRelations(t *testing.T) {
	path := t.TempDir() + "/relations.json"
	store, err := NewMemoryStore(path)
	if err != nil {
		t.Fatal(err)
	}
	first, err := store.AddMemoryWithMetadata("Work project deadline", "work", "high", nil)
	if err != nil {
		t.Fatal(err)
	}
	second, err := store.AddMemoryWithMetadata("Health appointment after work", "health", "medium", []string{first.ID})
	if err != nil {
		t.Fatal(err)
	}
	if len(store.List()[0].RelatedIDs) != 1 || store.List()[0].RelatedIDs[0] != second.ID {
		t.Fatal("expected related nodes to have reciprocal links")
	}

	deleted, err := store.DeleteMemories([]string{first.ID})
	if err != nil || len(deleted) != 1 || deleted[0].ID != first.ID {
		t.Fatalf("unexpected delete result: deleted=%+v err=%v", deleted, err)
	}
	reloaded, err := NewMemoryStore(path)
	if err != nil {
		t.Fatal(err)
	}
	remaining := reloaded.List()
	if len(remaining) != 1 || remaining[0].ID != second.ID || len(remaining[0].RelatedIDs) != 0 {
		t.Fatalf("deletion did not persist or left a dangling relation: %+v", remaining)
	}

	deleted, err = reloaded.DeleteAllMemories()
	if err != nil || len(deleted) != 1 {
		t.Fatalf("unexpected clear-all result: deleted=%+v err=%v", deleted, err)
	}
	reloaded, err = NewMemoryStore(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(reloaded.List()) != 0 {
		t.Fatal("clear-all was not persisted")
	}
}

func TestBulkDeleteCountMismatchLeavesMemoriesUntouched(t *testing.T) {
	store, err := NewMemoryStore("")
	if err != nil {
		t.Fatal(err)
	}
	created, err := store.AddMemory("A saved note")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.DeleteAllMemoriesIfCount(2); err == nil {
		t.Fatal("expected clear-all to reject a stale confirmed count")
	}
	if _, err := store.DeleteMemoriesIfCount([]string{created.ID}, 2); err == nil {
		t.Fatal("expected selected deletion to reject a stale confirmed count")
	}
	memories := store.List()
	if len(memories) != 1 || memories[0].ID != created.ID {
		t.Fatalf("stale bulk-delete confirmation changed memory state: %+v", memories)
	}
}

// TestMemorySearchMatchesMissingCharacterTypo guards the character n-gram lookup behavior.
func TestMemorySearchMatchesMissingCharacterTypo(t *testing.T) {
	store, err := NewMemoryStore("")
	if err != nil {
		t.Fatal(err)
	}
	created, err := store.AddMemory("guitar")
	if err != nil {
		t.Fatal(err)
	}

	match, score, found := store.FindRelevantMemory("guitra")
	if !found || match.ID != created.ID {
		t.Fatalf("expected typo-tolerant match, got found=%v score=%f memory=%+v", found, score, match)
	}
}

func TestMemorySearchRecallsPreferredNameFromNameQuestion(t *testing.T) {
	store, err := NewMemoryStore("")
	if err != nil {
		t.Fatal(err)
	}
	saved, err := store.AddMemoryWithMetadata("Please call me Colin", "preferences", "high", nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, query := range []string{"What's my name?", "What should you call me?", "Who am I?"} {
		matches := store.FindRelevantMemories(query, 5)
		found := false
		for _, match := range matches {
			if match.Memory.ID == saved.ID {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("name recall query %q did not retrieve the saved preference; matches=%+v", query, matches)
		}
	}
}

func TestIndexedSearchMatchesBruteForceCosineRanking(t *testing.T) {
	store, err := NewMemoryStore("")
	if err != nil {
		t.Fatal(err)
	}
	entries := []struct {
		content  string
		category string
	}{
		{"Please call me Colin", "preferences"},
		{"I prefer working in quiet rooms", "work"},
		{"My guitar practice is every Tuesday", "goals"},
		{"A dentist appointment is scheduled next month", "health"},
		{"The blue bicycle is stored at home", "home"},
		{"My favorite instrument is a guitar", "preferences"},
		{"Colin is working on a new project", "work"},
		{"Take a quiet walk after dinner", "health"},
	}
	for _, entry := range entries {
		if _, err := store.AddMemoryWithMetadata(entry.content, entry.category, "medium", nil); err != nil {
			t.Fatal(err)
		}
	}

	for _, query := range []string{"What should you call me?", "guitra practice", "quiet work", "dentist appointment", "blue bicycle", "no matching phrase"} {
		searchText := expandMemoryQuery(query)
		queryVector := localEmbedding(searchText)
		threshold := float32(0.18)
		if searchText != query {
			threshold = 0.1
		}
		want := make([]MemoryMatch, 0)
		for _, memory := range store.List() {
			if score := cosineSimilarity(queryVector, memory.Embedding); score >= threshold {
				want = append(want, MemoryMatch{Memory: memory, Score: score})
			}
		}
		sort.Slice(want, func(first, second int) bool { return want[first].Score > want[second].Score })
		got := store.FindRelevantMemories(query, len(entries))
		if len(got) != len(want) {
			t.Fatalf("query %q returned %d indexed results, brute force returned %d", query, len(got), len(want))
		}
		for index := range want {
			if got[index].Memory.ID != want[index].Memory.ID || got[index].Score != want[index].Score {
				t.Fatalf("query %q rank %d differs: indexed=%+v brute=%+v", query, index, got[index], want[index])
			}
		}
	}
}

func TestIndexedSearchPrunesUnrelatedMemoriesAndTracksUpdates(t *testing.T) {
	store, err := NewMemoryStore("")
	if err != nil {
		t.Fatal(err)
	}
	for index := 0; index < 128; index++ {
		content := fmt.Sprintf("Archive note sequence %03d contains ordinary details", index)
		if _, err := store.AddMemory(content); err != nil {
			t.Fatal(err)
		}
	}
	store.RLock()
	candidateCount := len(store.searchCandidatePositionsLocked("needlequartz"))
	store.RUnlock()
	if candidateCount >= len(store.List()) {
		t.Fatalf("candidate index did not prune unrelated records: candidates=%d total=%d", candidateCount, len(store.List()))
	}

	updated, err := store.ReplaceMemory(store.List()[0].ID, "needlequartz is the updated note")
	if err != nil {
		t.Fatal(err)
	}
	match, _, found := store.FindRelevantMemory("needlequartz")
	if !found || match.ID != updated.ID {
		t.Fatalf("updated memory was not reindexed: found=%v match=%+v", found, match)
	}
	if _, err := store.DeleteMemories([]string{updated.ID}); err != nil {
		t.Fatal(err)
	}
	store.RLock()
	for _, position := range store.searchCandidatePositionsLocked("needlequartz") {
		if store.memories[position].ID == updated.ID {
			store.RUnlock()
			t.Fatal("deleted memory remains in the search index")
		}
	}
	store.RUnlock()
}

// TestFallbackAction checks the inexpensive action path used when model JSON is unavailable.
func TestFallbackAction(t *testing.T) {
	tests := []struct {
		input    string
		hasMatch bool
		want     string
	}{
		{input: "What did I say about work?", hasMatch: true, want: "SEARCH"},
		{input: "Also, I finished the first task", hasMatch: true, want: "FOLLOW_UP"},
		{input: "Delete this memory", hasMatch: true, want: "DELETE"},
		{input: "Delete all my nodes", hasMatch: true, want: "DELETE_ALL"},
		{input: "Delete the node it had and all nodes", hasMatch: true, want: "DELETE_ALL"},
		{input: "I started learning guitar", hasMatch: false, want: "RESPOND"},
		{input: "Hello there", hasMatch: false, want: "RESPOND"},
		{input: "Thanks, that helps", hasMatch: false, want: "RESPOND"},
		{input: "Please call me Colin", hasMatch: false, want: "STORE"},
		{input: "Call me Colin.", hasMatch: false, want: "STORE"},
	}
	for _, test := range tests {
		if got := fallbackAction(test.input, test.hasMatch); got != test.want {
			t.Errorf("fallbackAction(%q, %v) = %q, want %q", test.input, test.hasMatch, got, test.want)
		}
	}
}

func TestResolveActionDoesNotTurnChatIntoStorage(t *testing.T) {
	for _, input := range []string{"Hello there", "Thanks, that helps", "Can you explain this?"} {
		if got := resolveAction("RESPOND", input, false); got != "RESPOND" {
			t.Errorf("resolveAction(RESPOND, %q, false) = %q", input, got)
		}
	}
	if got := resolveAction("STORE", "I prefer quiet rooms when I work.", false); got != "STORE" {
		t.Fatalf("explicit durable-memory decision was changed to %q", got)
	}
	if got := fallbackAction("Please call me Colin", false); got != "STORE" {
		t.Fatalf("personal name preference misclassified as %q", got)
	}
}

// TestResolveActionKeepsStatementsOutOfSearch prevents facts from being discarded as searches.
func TestResolveActionKeepsStatementsOutOfSearch(t *testing.T) {
	if got := resolveAction("SEARCH", "My favorite instrument is a guitar.", false); got != "RESPOND" {
		t.Fatalf("ordinary statement classified as %q; want RESPOND when model asks to search without recall intent", got)
	}
	if got := resolveAction("STORE", "What is my favorite instrument?", true); got != "SEARCH" {
		t.Fatalf("recall question classified as %q; want SEARCH", got)
	}
	for _, input := range []string{"Thanks, that helps.", "What is the capital of France?", "I'm tired right now", "Don't save this: I had a rough day."} {
		if got := resolveAction("STORE", input, false); got != "RESPOND" {
			t.Errorf("model STORE decision persisted non-memory input %q as %q", input, got)
		}
	}
	if got := resolveAction("STORE", "Please call me Colin.", false); got != "STORE" {
		t.Fatalf("explicit name preference changed to %q", got)
	}
}

func TestBulkDeleteRequiresExactCountPhrase(t *testing.T) {
	tests := []struct {
		action string
		count  int
		phrase string
		valid  bool
	}{
		{action: "DELETE_ALL", count: 6, phrase: "DELETE ALL 6", valid: true},
		{action: "DELETE_ALL", count: 6, phrase: "DELETE ALL", valid: false},
		{action: "DELETE_SELECTED", count: 3, phrase: "DELETE NODES 3", valid: true},
		{action: "DELETE_SELECTED", count: 3, phrase: "DELETE NODES 4", valid: false},
	}
	for _, test := range tests {
		err := validateBulkDeletePhrase(test.action, test.count, test.phrase)
		if (err == nil) != test.valid {
			t.Errorf("validateBulkDeletePhrase(%q, %d, %q) error=%v, valid=%v", test.action, test.count, test.phrase, err, test.valid)
		}
	}
}

// TestMemoryStoreRejectsWritesAtConfiguredLimits checks count and per-entry caps.
func TestMemoryStoreRejectsWritesAtConfiguredLimits(t *testing.T) {
	path := t.TempDir() + "/bounded.json"
	store, err := NewMemoryStoreWithLimits(path, MemoryLimits{
		MaxBytes:      1024 * 1024,
		MaxItems:      1,
		MaxEntryBytes: 16,
	})
	if err != nil {
		t.Fatal(err)
	}

	created, err := store.AddMemory("1234567890123456")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.AddMemory("second item"); err == nil {
		t.Fatal("expected the record-count cap to reject a second memory")
	}
	if _, err := store.UpdateMemory(created.ID, "additional content"); err == nil {
		t.Fatal("expected the entry-size cap to reject an oversized update")
	}
	if len(store.List()) != 1 || store.List()[0].Content != created.Content {
		t.Fatal("rejected writes changed the saved memory")
	}
}

// TestMemoryStoreRejectsDiskLimitWithoutWriting verifies failed saves leave no in-memory record.
func TestMemoryStoreRejectsDiskLimitWithoutWriting(t *testing.T) {
	path := t.TempDir() + "/too-small.json"
	store, err := NewMemoryStoreWithLimits(path, MemoryLimits{
		MaxBytes:      1,
		MaxItems:      1,
		MaxEntryBytes: 32,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.AddMemory("data"); err == nil {
		t.Fatal("expected serialized-file cap to reject the memory")
	}
	if len(store.List()) != 0 {
		t.Fatal("failed disk write left a memory in the in-memory store")
	}
}

// TestLimitTextBytesPreservesUTF8 ensures model prompts are never cut mid-character.
func TestLimitTextBytesPreservesUTF8(t *testing.T) {
	got := limitTextBytes("ééx", 3)
	if got != "é" {
		t.Fatalf("limitTextBytes returned %q; want one complete UTF-8 character", got)
	}
}

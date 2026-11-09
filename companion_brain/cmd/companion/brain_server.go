package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/gorilla/websocket"
)

// upgrader accepts WebSocket requests from the page's own origin.
var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		origin := r.Header.Get("Origin")
		if origin == "" {
			return true
		}
		parsed, err := url.Parse(origin)
		return err == nil && parsed.Host == r.Host
	},
}

// OllamaRequest carries a prompt and resource options to the local Ollama API.
type OllamaRequest struct {
	Model     string        `json:"model"`
	Prompt    string        `json:"prompt"`
	Stream    bool          `json:"stream"`
	Format    any           `json:"format,omitempty"`
	Options   OllamaOptions `json:"options"`
	KeepAlive string        `json:"keep_alive"`
}

// OllamaOptions bounds context size, generated tokens, CPU threads, and randomness.
type OllamaOptions struct {
	NumCtx      int     `json:"num_ctx"`
	NumPredict  int     `json:"num_predict"`
	NumThread   int     `json:"num_thread"`
	Temperature float32 `json:"temperature"`
}

// OllamaResponse is one response frame from Ollama's generate endpoint.
type OllamaResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

// OutgoingEvent structures specific system commands to alter the 3D frontend interface.
type OutgoingEvent struct {
	Type             string            `json:"type"`
	Content          string            `json:"content,omitempty"`
	ID               string            `json:"id,omitempty"`
	ProposalID       string            `json:"proposal_id,omitempty"`
	Action           string            `json:"action,omitempty"`
	Reason           string            `json:"reason,omitempty"`
	Score            float32           `json:"score,omitempty"`
	Timestamp        time.Time         `json:"timestamp,omitempty"`
	Category         string            `json:"category,omitempty"`
	Importance       string            `json:"importance,omitempty"`
	RelatedIDs       []string          `json:"related_ids,omitempty"`
	SearchCandidates []SearchCandidate `json:"search_candidates,omitempty"`
}

type SearchCandidate struct {
	ID    string  `json:"id"`
	Score float32 `json:"score"`
}

// IncomingMessage carries text plus the user's input mode from the browser.
type IncomingMessage struct {
	Text          string   `json:"text"`
	InputMode     string   `json:"input_mode"`
	ReviewMemory  bool     `json:"review_memory,omitempty"`
	ProposalID    string   `json:"proposal_id,omitempty"`
	Category      string   `json:"category,omitempty"`
	Importance    string   `json:"importance,omitempty"`
	RelatedIDs    []string `json:"related_ids,omitempty"`
	ConfirmAction string   `json:"confirm_action,omitempty"`
	ConfirmPhrase string   `json:"confirm_phrase,omitempty"`
	ID            string   `json:"id,omitempty"`
	IDs           []string `json:"ids,omitempty"`
}

// GatekeeperDecision combines the memory action with typo-corrected text.
type GatekeeperDecision struct {
	Action          string   `json:"action"`
	NormalizedInput string   `json:"normalized_input"`
	TargetID        string   `json:"target_id"`
	Category        string   `json:"category"`
	Importance      string   `json:"importance"`
	RelatedIDs      []string `json:"related_ids"`
}

// These defaults keep model requests and concurrent sessions within modest limits.
var ollamaClient = &http.Client{Timeout: 2 * time.Minute}
var ollamaOptions = OllamaOptions{NumCtx: 2048, NumPredict: 192, NumThread: 2, Temperature: 0.1}
var ollamaKeepAlive = "2m"
var maxInputBytes = 4 * 1024
var maxContextBytes = 2 * 1024
var clientSlots = make(chan struct{}, 8)

func ollamaURL() string {
	// Allow a custom local endpoint while keeping Ollama's standard URL as default.
	if value := os.Getenv("OLLAMA_URL"); value != "" {
		return strings.TrimRight(value, "/")
	}
	return "http://127.0.0.1:11434/api/generate"
}

func ollamaModel() string {
	if value := os.Getenv("OLLAMA_MODEL"); value != "" {
		return value
	}
	return "qwen2.5:3b"
}

// requestOllama handles both a single structured reply and newline-delimited token streams.
// When onToken is set, each generated fragment is forwarded immediately to the browser.
func requestOllama(ctx context.Context, prompt string, stream bool, format any, onToken func(string) error) (string, error) {
	payload := OllamaRequest{
		Model:     ollamaModel(),
		Prompt:    prompt,
		Stream:    stream,
		Format:    format,
		Options:   ollamaOptions,
		KeepAlive: ollamaKeepAlive,
	}
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("encode Ollama request: %w", err)
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, ollamaURL(), bytes.NewReader(jsonData))
	if err != nil {
		return "", fmt.Errorf("create Ollama request: %w", err)
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := ollamaClient.Do(request)
	if err != nil {
		return "", fmt.Errorf("connect to Ollama at %s: %w", ollamaURL(), err)
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return "", fmt.Errorf("Ollama returned %s", response.Status)
	}

	var output strings.Builder
	decoder := json.NewDecoder(response.Body)
	for {
		var result OllamaResponse
		if err := decoder.Decode(&result); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return output.String(), fmt.Errorf("decode Ollama response: %w", err)
		}
		output.WriteString(result.Response)
		if onToken != nil && result.Response != "" {
			if err := onToken(result.Response); err != nil {
				return output.String(), err
			}
		}
		if result.Done {
			break
		}
	}
	return output.String(), nil
}

// The schema makes Ollama return both a valid action and normalized input every time.
var gatekeeperResponseSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"action": map[string]any{
			"type": "string",
			"enum": []string{"RESPOND", "STORE", "SEARCH", "FOLLOW_UP", "DELETE", "DELETE_ALL"},
		},
		"normalized_input": map[string]any{"type": "string"},
		"target_id":        map[string]any{"type": "string"},
		"category": map[string]any{
			"type": "string",
			"enum": []string{"health", "work", "relationships", "goals", "preferences", "home", "other"},
		},
		"importance": map[string]any{
			"type": "string",
			"enum": []string{"low", "medium", "high"},
		},
		"related_ids": map[string]any{
			"type":  "array",
			"items": map[string]any{"type": "string"},
		},
	},
	"required":             []string{"action", "normalized_input", "target_id", "category", "importance", "related_ids"},
	"additionalProperties": false,
}

func boundedIntFromEnv(name string, fallback, minimum, maximum int64) (int64, error) {
	// Reject malformed or out-of-range settings at startup instead of silently ignoring caps.
	value := os.Getenv(name)
	if value == "" {
		return fallback, nil
	}
	parsed, err := strconv.ParseInt(value, 10, 64)
	if err != nil || parsed < minimum || parsed > maximum {
		return 0, fmt.Errorf("%s must be an integer from %d to %d", name, minimum, maximum)
	}
	return parsed, nil
}

func limitTextBytes(text string, maximum int) string {
	// Keep prompt truncation on a UTF-8 character boundary.
	if len(text) <= maximum {
		return text
	}
	end := maximum
	for end > 0 && !utf8.RuneStart(text[end]) {
		end--
	}
	return text[:end]
}

func loadRuntimeLimits() (MemoryLimits, error) {
	// Read storage, request, model, and client caps from environment variables.
	defaults := DefaultMemoryLimits()
	maxBytes, err := boundedIntFromEnv("MEMORY_MAX_BYTES", defaults.MaxBytes, 1024, 512*1024*1024)
	if err != nil {
		return MemoryLimits{}, err
	}
	maxItems, err := boundedIntFromEnv("MEMORY_MAX_ITEMS", int64(defaults.MaxItems), 1, 100000)
	if err != nil {
		return MemoryLimits{}, err
	}
	maxEntryBytes, err := boundedIntFromEnv("MEMORY_MAX_ENTRY_BYTES", int64(defaults.MaxEntryBytes), 1, 1024*1024)
	if err != nil {
		return MemoryLimits{}, err
	}
	inputBytes, err := boundedIntFromEnv("MAX_INPUT_BYTES", 4*1024, 128, 60*1024)
	if err != nil {
		return MemoryLimits{}, err
	}
	contextBytes, err := boundedIntFromEnv("MAX_CONTEXT_BYTES", 2*1024, 128, 32*1024)
	if err != nil {
		return MemoryLimits{}, err
	}
	contextTokens, err := boundedIntFromEnv("OLLAMA_NUM_CTX", 2048, 256, 8192)
	if err != nil {
		return MemoryLimits{}, err
	}
	outputTokens, err := boundedIntFromEnv("OLLAMA_NUM_PREDICT", 192, 16, 1024)
	if err != nil {
		return MemoryLimits{}, err
	}
	threads, err := boundedIntFromEnv("OLLAMA_NUM_THREAD", 2, 1, 64)
	if err != nil {
		return MemoryLimits{}, err
	}
	maxClients, err := boundedIntFromEnv("MAX_CLIENTS", 8, 1, 32)
	if err != nil {
		return MemoryLimits{}, err
	}
	maxInputBytes = int(inputBytes)
	maxContextBytes = int(contextBytes)
	ollamaOptions = OllamaOptions{NumCtx: int(contextTokens), NumPredict: int(outputTokens), NumThread: int(threads)}
	clientSlots = make(chan struct{}, int(maxClients))
	if value := os.Getenv("OLLAMA_KEEP_ALIVE"); value != "" {
		ollamaKeepAlive = value
	}
	return MemoryLimits{MaxBytes: maxBytes, MaxItems: int(maxItems), MaxEntryBytes: int(maxEntryBytes)}, nil
}

func decodeIncoming(data []byte) IncomingMessage {
	// Accept the current JSON payload and retain compatibility with plain-text clients.
	var message IncomingMessage
	if err := json.Unmarshal(data, &message); err == nil && (message.Text != "" || message.ConfirmAction != "") {
		if message.InputMode != "voice" {
			message.InputMode = "text"
		}
		return message
	}
	return IncomingMessage{Text: string(data), InputMode: "text"}
}

func classifyInput(ctx context.Context, input string, candidates []MemoryMatch) (GatekeeperDecision, error) {
	// Give the gatekeeper a short list of local candidates for action and relationship decisions.
	hasMatch := len(candidates) > 0
	var candidateText strings.Builder
	candidateLimit := maxContextBytes
	if len(candidates) > 0 {
		candidateLimit /= len(candidates)
	}
	validIDs := make(map[string]bool, len(candidates))
	for _, candidate := range candidates {
		validIDs[candidate.Memory.ID] = true
		fmt.Fprintf(&candidateText, "ID=%s; category=%s; importance=%s; content=%s\n",
			candidate.Memory.ID,
			candidate.Memory.Category,
			candidate.Memory.Importance,
			limitTextBytes(candidate.Memory.Content, candidateLimit),
		)
	}
	if !hasMatch {
		candidateText.WriteString("No matching memories found.\n")
	}
	prompt := fmt.Sprintf(
		"Process the user's input for a personal memory assistant. Return only the required JSON fields.\n"+
			"Choose RESPOND when uncertain. RESPOND answers, acknowledges, or chats without changing saved memory. STORE durable personal information likely to matter in future conversations: stable preferences, ongoing goals, important personal facts, or continuing situations. This includes name and identity preferences like 'call me Colin', 'my name is X', or 'please refer to me as X'. Do not store questions, greetings, thanks, ordinary conversation, one-time requests, or transient details.\n"+
			"Use SEARCH to recall saved information and FOLLOW_UP only to add a durable detail to an existing memory. Use DELETE for one clearly identified memory and DELETE_ALL only when the user explicitly asks to remove every memory.\n"+
			"Examples: 'Hello there!' => RESPOND; 'Can you explain why the sky is blue?' => RESPOND; 'Thanks, that helps.' => RESPOND; 'Please call me Colin.' => STORE with category preferences; 'My name is Sam.' => STORE with category preferences; 'I prefer quiet rooms when I work.' => STORE with category preferences; 'What did I say my work preference was?' => SEARCH. Apply the same distinction to equivalent wording.\n"+
			"For DELETE, set target_id to the matching candidate ID. For DELETE_ALL, leave target_id empty. Never invent IDs.\n"+
			"For STORE, set category and importance, and list only genuinely related candidate IDs in related_ids. Do not create links based only on weak similarity.\n"+
			"normalized_input should fix only clear spelling mistakes or accidental missing/extra characters. Use relevant memory to resolve an obvious contextual typo. Preserve the user's meaning and wording otherwise; never change names, numbers, dates, places, negation, or facts. If uncertain, copy the original input exactly.\n"+
			"Keep the corrected input concise and do not add facts.\n\n"+
			"Candidate memories:\n%sOriginal user input: %s\nJSON:", candidateText.String(), input,
	)
	result, err := requestOllama(ctx, prompt, false, gatekeeperResponseSchema, nil)
	if err != nil {
		return GatekeeperDecision{Action: fallbackAction(input, hasMatch), NormalizedInput: input, Category: "other", Importance: "medium"}, nil
	}
	var decision GatekeeperDecision
	if err := json.Unmarshal([]byte(result), &decision); err != nil {
		return GatekeeperDecision{Action: fallbackAction(input, hasMatch), NormalizedInput: input, Category: "other", Importance: "medium"}, nil
	}
	action := resolveAction(decision.Action, input, hasMatch)
	correctedInput := strings.TrimSpace(decision.NormalizedInput)
	if correctedInput == "" || len(correctedInput) > maxInputBytes {
		correctedInput = input
	}
	if action == "STORE" || action == "FOLLOW_UP" {
		correctedInput = input
	}
	targetID := ""
	if validIDs[decision.TargetID] {
		targetID = decision.TargetID
	} else if action == "DELETE" && hasMatch {
		targetID = candidates[0].Memory.ID
	}
	relatedIDs := make([]string, 0, len(decision.RelatedIDs))
	for _, id := range decision.RelatedIDs {
		if validIDs[id] && id != targetID {
			relatedIDs = appendUniqueID(relatedIDs, id)
		}
	}
	return GatekeeperDecision{
		Action:          action,
		NormalizedInput: correctedInput,
		TargetID:        targetID,
		Category:        normalizeCategory(decision.Category),
		Importance:      normalizeImportance(decision.Importance),
		RelatedIDs:      relatedIDs,
	}, nil
}

func resolveAction(modelAction, input string, hasMatch bool) string {
	// Keep explicit recall/deletion intent and prevent follow-ups without a match.
	action := strings.ToUpper(strings.TrimSpace(modelAction))
	fallback := fallbackAction(input, hasMatch)
	if action != "RESPOND" && action != "STORE" && action != "SEARCH" && action != "FOLLOW_UP" && action != "DELETE" && action != "DELETE_ALL" {
		action = fallback
	}
	if fallback == "SEARCH" || fallback == "DELETE" || fallback == "DELETE_ALL" || (action == "SEARCH" && fallback != "SEARCH") {
		action = fallback
	}
	if action == "FOLLOW_UP" && !hasMatch {
		action = "STORE"
	}
	if (action == "STORE" || action == "FOLLOW_UP") && shouldNotPersistInput(input) {
		return "RESPOND"
	}
	return action
}

func shouldNotPersistInput(input string) bool {
	lower := strings.ToLower(strings.TrimSpace(input))
	for _, phrase := range []string{"don't save", "do not save", "don't store", "do not store", "don't remember", "do not remember", "not a memory"} {
		if strings.Contains(lower, phrase) {
			return true
		}
	}
	if strings.Contains(lower, "?") && !hasExplicitDurablePrefix(lower) {
		return true
	}
	for _, phrase := range []string{"i'm tired", "i am tired", "i feel ", "i'm feeling ", "i am feeling ", "just venting", "for tonight", "for today", "one time", "one-time"} {
		if strings.Contains(lower, phrase) {
			return true
		}
	}
	switch strings.Trim(lower, ".! ") {
	case "hi", "hello", "hello there", "hey", "thanks", "thank you", "thanks, that helps", "okay", "ok", "got it", "good morning", "good night":
		return true
	}
	return false
}

func hasExplicitDurablePrefix(input string) bool {
	for _, prefix := range []string{"call me ", "please call me ", "my name is ", "i go by ", "please refer to me as ", "refer to me as ", "i prefer ", "my favorite is ", "my favourite is "} {
		if strings.HasPrefix(input, prefix) && len(strings.TrimSpace(strings.TrimPrefix(input, prefix))) > 0 {
			return true
		}
	}
	return false
}

func fallbackAction(input string, hasMatch bool) string {
	// Use lightweight cues when Ollama is unavailable or returns an invalid action.
	lower := strings.ToLower(strings.TrimSpace(input))
	hasDeleteIntent := false
	for _, verb := range []string{"delete", "remove", "erase", "forget", "clear", "wipe"} {
		if strings.Contains(lower, verb) {
			hasDeleteIntent = true
			break
		}
	}
	if hasDeleteIntent {
		for _, phrase := range []string{"all nodes", "every node", "all memories", "every memory", "all entries", "everything"} {
			if strings.Contains(lower, phrase) {
				return "DELETE_ALL"
			}
		}
	}
	for _, phrase := range []string{"delete all", "remove all", "clear all", "erase all", "forget everything", "wipe all", "clear the matrix"} {
		if strings.Contains(lower, phrase) {
			return "DELETE_ALL"
		}
	}
	for _, prefix := range []string{"delete ", "remove ", "erase ", "forget ", "clear "} {
		if strings.HasPrefix(lower, prefix) {
			return "DELETE"
		}
	}
	for _, prefix := range []string{"what did i", "what is my", "what's my", "when did i", "where did i", "do you remember", "tell me what", "find ", "search ", "remind me"} {
		if strings.HasPrefix(lower, prefix) {
			return "SEARCH"
		}
	}
	for _, phrase := range []string{"call me ", "my name is ", "please call me ", "please refer to me as ", "refer to me as ", "i go by "} {
		if strings.HasPrefix(lower, phrase) && len(strings.TrimSpace(strings.TrimPrefix(lower, phrase))) > 0 {
			return "STORE"
		}
	}
	if hasMatch {
		words := strings.Fields(strings.NewReplacer(".", " ", ",", " ", "?", " ", "!", " ").Replace(lower))
		wordSet := make(map[string]bool, len(words))
		for _, word := range words {
			wordSet[word] = true
		}
		for _, marker := range []string{"that", "it", "also", "update", "actually", "instead"} {
			if wordSet[marker] {
				return "FOLLOW_UP"
			}
		}
		if strings.Contains(lower, "follow up") {
			return "FOLLOW_UP"
		}
	}
	for _, marker := range []string{" delete ", " remove ", " erase ", " forget "} {
		if strings.Contains(" "+lower+" ", marker) {
			return "DELETE"
		}
	}
	return "RESPOND"
}

func writeEvent(conn *websocket.Conn, event OutgoingEvent) error {
	data, err := json.Marshal(event)
	if err != nil {
		return err
	}
	return conn.WriteMessage(websocket.TextMessage, data)
}

func applyConfirmedAction(conn *websocket.Conn, incoming IncomingMessage) error {
	var deleted []Memory
	var err error
	switch incoming.ConfirmAction {
	case "DELETE":
		deleted, err = GlobalMemoryStore.DeleteMemoriesIfCount([]string{incoming.ID}, 1)
	case "DELETE_SELECTED":
		if len(incoming.IDs) > 1 {
			if err := validateBulkDeletePhrase("DELETE_SELECTED", len(incoming.IDs), incoming.ConfirmPhrase); err != nil {
				return err
			}
		}
		deleted, err = GlobalMemoryStore.DeleteMemoriesIfCount(incoming.IDs, len(incoming.IDs))
	case "DELETE_ALL":
		currentCount := len(GlobalMemoryStore.List())
		if err := validateBulkDeletePhrase("DELETE_ALL", currentCount, incoming.ConfirmPhrase); err != nil {
			return err
		}
		deleted, err = GlobalMemoryStore.DeleteAllMemoriesIfCount(currentCount)
	case "APPROVE_STORE":
		memory, err := GlobalMemoryStore.AddMemoryWithMetadata(incoming.Text, incoming.Category, incoming.Importance, incoming.RelatedIDs)
		if err != nil {
			return err
		}
		return writeEvent(conn, OutgoingEvent{
			Type: "new_node", ID: memory.ID, ProposalID: incoming.ProposalID, Content: memory.Content, Action: "STORE",
			Reason: "Approved from the memory review queue.", Timestamp: memory.Timestamp,
			Category: memory.Category, Importance: memory.Importance, RelatedIDs: memory.RelatedIDs,
		})
	case "APPROVE_FOLLOW_UP":
		memory, err := GlobalMemoryStore.UpdateMemory(incoming.ID, incoming.Text)
		if err != nil {
			return err
		}
		return writeEvent(conn, OutgoingEvent{
			Type: "update_node", ID: memory.ID, ProposalID: incoming.ProposalID, Content: memory.Content, Action: "FOLLOW_UP",
			Reason: "Approved follow-up from the memory review queue.", Timestamp: memory.Timestamp,
			Category: memory.Category, Importance: memory.Importance, RelatedIDs: memory.RelatedIDs,
		})
	case "EDIT_MEMORY":
		memory, err := GlobalMemoryStore.ReplaceMemory(incoming.ID, incoming.Text)
		if err != nil {
			return err
		}
		return writeEvent(conn, OutgoingEvent{
			Type: "update_node", ID: memory.ID, Content: memory.Content, Action: "EDIT_MEMORY",
			Reason: "Edited by the user.", Timestamp: memory.Timestamp,
			Category: memory.Category, Importance: memory.Importance, RelatedIDs: memory.RelatedIDs,
		})
	default:
		return fmt.Errorf("unsupported confirmation action %q", incoming.ConfirmAction)
	}
	if err != nil {
		return err
	}
	if incoming.ConfirmAction == "DELETE_ALL" {
		return writeEvent(conn, OutgoingEvent{Type: "nodes_cleared", Content: fmt.Sprintf("%d", len(deleted))})
	}
	for _, memory := range deleted {
		if err := writeEvent(conn, OutgoingEvent{Type: "node_deleted", ID: memory.ID, Content: memory.Content}); err != nil {
			return err
		}
	}
	return nil
}

func validateBulkDeletePhrase(action string, count int, phrase string) error {
	expected := fmt.Sprintf("DELETE NODES %d", count)
	if action == "DELETE_ALL" {
		expected = fmt.Sprintf("DELETE ALL %d", count)
	}
	if phrase != expected {
		return fmt.Errorf("bulk deletion cancelled; type %q exactly to confirm", expected)
	}
	return nil
}

// handleCompanionCore restores saved nodes, applies the gatekeeper decision, then streams a reply.
func handleCompanionCore(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("WebSocket expansion failure:", err)
		return
	}
	defer conn.Close()
	conn.SetReadLimit(64 * 1024)
	select {
	case clientSlots <- struct{}{}:
		defer func() { <-clientSlots }()
	default:
		_ = writeEvent(conn, OutgoingEvent{Type: "error", Content: "The local companion is at its configured client limit. Try again shortly."})
		return
	}

	// Send existing memories first so a newly opened matrix matches the local store.
	for _, memory := range GlobalMemoryStore.List() {
		if err := writeEvent(conn, OutgoingEvent{
			Type: "restore_node", ID: memory.ID, Content: memory.Content, Timestamp: memory.Timestamp,
			Category: memory.Category, Importance: memory.Importance, RelatedIDs: memory.RelatedIDs,
		}); err != nil {
			return
		}
	}
	if err := writeEvent(conn, OutgoingEvent{Type: "restore_complete"}); err != nil {
		return
	}

	for {
		_, message, err := conn.ReadMessage()
		if err != nil {
			break
		}
		incoming := decodeIncoming(message)
		if incoming.ConfirmAction != "" {
			if err := applyConfirmedAction(conn, incoming); err != nil {
				_ = writeEvent(conn, OutgoingEvent{Type: "error", ProposalID: incoming.ProposalID, Content: err.Error()})
			}
			continue
		}
		userInput := strings.TrimSpace(incoming.Text)
		if userInput == "" {
			continue
		}
		if len(userInput) > maxInputBytes {
			_ = writeEvent(conn, OutgoingEvent{Type: "error", Content: fmt.Sprintf("Input is %d bytes; configured maximum is %d bytes.", len(userInput), maxInputBytes)})
			continue
		}

		// Retrieve a few typo-tolerant candidates so the model can select targets and links.
		candidates := GlobalMemoryStore.FindRelevantMemories(userInput, 5)
		hasMatch := len(candidates) > 0
		patterns := GlobalMemoryStore.AnalyzePatterns()
		ctx, cancel := context.WithTimeout(r.Context(), 90*time.Second)
		decision, err := classifyInput(ctx, userInput, candidates)
		if err != nil {
			cancel()
			_ = writeEvent(conn, OutgoingEvent{Type: "error", Content: err.Error()})
			continue
		}
		action := decision.Action
		normalizedInput := decision.NormalizedInput
		if normalizedInput != userInput {
			_ = writeEvent(conn, OutgoingEvent{Type: "input_corrected", Content: normalizedInput})
		}
		if action == "DELETE" {
			if decision.TargetID == "" {
				_ = writeEvent(conn, OutgoingEvent{Type: "error", Content: "I could not identify a saved node to delete."})
				cancel()
				continue
			}
			var target Memory
			for _, candidate := range candidates {
				if candidate.Memory.ID == decision.TargetID {
					target = candidate.Memory
					break
				}
			}
			_ = writeEvent(conn, OutgoingEvent{Type: "confirm_delete", ID: target.ID, Content: target.Content, Action: action})
			cancel()
			continue
		}
		if action == "DELETE_ALL" {
			count := len(GlobalMemoryStore.List())
			if count == 0 {
				_ = writeEvent(conn, OutgoingEvent{Type: "search_result", Content: "There are no saved nodes to delete.", Action: action})
			} else {
				_ = writeEvent(conn, OutgoingEvent{Type: "confirm_delete_all", Content: fmt.Sprintf("%d", count), Action: action})
			}
			cancel()
			continue
		}

		// Store only on explicit STORE; ordinary responses leave the memory file unchanged.
		var contextMemory string
		switch action {
		case "STORE":
			if incoming.ReviewMemory {
				_ = writeEvent(conn, OutgoingEvent{
					Type: "memory_proposal", Content: normalizedInput, Action: action,
					Reason:   "The local gatekeeper classified this as durable personal information.",
					Category: decision.Category, Importance: decision.Importance, RelatedIDs: decision.RelatedIDs,
				})
				break
			}
			memory, err := GlobalMemoryStore.AddMemoryWithMetadata(normalizedInput, decision.Category, decision.Importance, decision.RelatedIDs)
			if err != nil {
				_ = writeEvent(conn, OutgoingEvent{Type: "error", Content: err.Error()})
				cancel()
				continue
			}
			_ = writeEvent(conn, OutgoingEvent{
				Type: "new_node", ID: memory.ID, Content: memory.Content, Action: action,
				Reason:    "The local gatekeeper classified this as durable personal information.",
				Timestamp: memory.Timestamp, Category: memory.Category, Importance: memory.Importance, RelatedIDs: memory.RelatedIDs,
			})
		case "FOLLOW_UP":
			if incoming.ReviewMemory {
				contextMemory = candidates[0].Memory.Content
				_ = writeEvent(conn, OutgoingEvent{
					Type: "memory_proposal", ID: candidates[0].Memory.ID, Content: normalizedInput, Action: action,
					Reason:   "The local gatekeeper classified this as a possible update to a related memory.",
					Category: candidates[0].Memory.Category, Importance: candidates[0].Memory.Importance,
					RelatedIDs: candidates[0].Memory.RelatedIDs,
				})
				break
			}
			memory, err := GlobalMemoryStore.UpdateMemory(candidates[0].Memory.ID, normalizedInput)
			if err != nil {
				_ = writeEvent(conn, OutgoingEvent{Type: "error", Content: err.Error()})
				cancel()
				continue
			}
			contextMemory = memory.Content
			_ = writeEvent(conn, OutgoingEvent{
				Type: "update_node", ID: memory.ID, Content: memory.Content, Action: action,
				Timestamp: memory.Timestamp, Category: memory.Category, Importance: memory.Importance, RelatedIDs: memory.RelatedIDs,
			})
		case "SEARCH":
			searchCandidates := make([]SearchCandidate, 0, len(candidates))
			for _, candidate := range candidates {
				searchCandidates = append(searchCandidates, SearchCandidate{ID: candidate.Memory.ID, Score: candidate.Score})
			}
			_ = writeEvent(conn, OutgoingEvent{Type: "search_started", Content: normalizedInput, Action: action, SearchCandidates: searchCandidates})
			if hasMatch {
				closest := candidates[0].Memory
				contextMemory = closest.Content
				_ = writeEvent(conn, OutgoingEvent{Type: "search_result", ID: closest.ID, Content: closest.Content, Action: action, Score: candidates[0].Score, Reason: "Highest local semantic match for this recall query."})
			} else {
				_ = writeEvent(conn, OutgoingEvent{Type: "search_result", Content: "No relevant saved memory found.", Action: action})
			}
		case "RESPOND":
			_ = writeEvent(conn, OutgoingEvent{Type: "action_status", Content: "No memory saved for this message.", Action: action})
		}

		// Keep the response prompt small by including only the selected memory and current input.
		constructedPrompt := "You are a warm, brief, non-judgmental personal companion. Be practical and do not claim certainty you do not have.\n"
		if patterns.RoutineStatus != "Stable" {
			constructedPrompt += fmt.Sprintf("Behavioral context: %s. Keep guidance low-pressure.\n", patterns.RoutineStatus)
		}
		if contextMemory != "" {
			constructedPrompt += fmt.Sprintf("Relevant saved context: %s\n", limitTextBytes(contextMemory, maxContextBytes))
		}
		constructedPrompt += fmt.Sprintf("User: %s\nAssistant:", normalizedInput)
		_, err = requestOllama(ctx, constructedPrompt, true, nil, func(token string) error {
			return writeEvent(conn, OutgoingEvent{Type: "stream_text", Content: token})
		})
		cancel()
		if err != nil {
			_ = writeEvent(conn, OutgoingEvent{Type: "error", Content: err.Error()})
		}
		_ = writeEvent(conn, OutgoingEvent{Type: "response_done", Action: action})
	}
}

func main() {
	// Load guardrails and persistent memory before registering the HTTP and WebSocket routes.
	projectRoot := os.Getenv("COMPANION_ROOT")
	if projectRoot == "" {
		var err error
		projectRoot, err = os.Getwd()
		if err != nil {
			log.Fatal(err)
		}
	}
	limits, err := loadRuntimeLimits()
	if err != nil {
		log.Fatal(err)
	}
	memoryPath := os.Getenv("MEMORY_FILE")
	if memoryPath == "" {
		memoryPath = filepath.Join("data", "memory.json")
	}
	if !filepath.IsAbs(memoryPath) {
		memoryPath = filepath.Join(projectRoot, memoryPath)
	}
	GlobalMemoryStore, err = NewMemoryStoreWithLimits(memoryPath, limits)
	if err != nil {
		log.Fatal(err)
	}
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		http.ServeFile(w, r, filepath.Join(projectRoot, "web", "index.html"))
	})
	http.HandleFunc("/ws/stream", handleCompanionCore)
	http.HandleFunc("/api/transcribe", handleTranscribe)
	http.HandleFunc("/api/lab/simulate", handleLabSimulate)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	fmt.Printf("Companion Core running at http://localhost:%s...\n", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

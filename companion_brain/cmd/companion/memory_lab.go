package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"
)

type labMemoryInput struct {
	ID         string   `json:"id"`
	Content    string   `json:"content"`
	Category   string   `json:"category"`
	Importance string   `json:"importance"`
	RelatedIDs []string `json:"related_ids"`
}

type labSimulationRequest struct {
	Text       string           `json:"text"`
	Memories   []labMemoryInput `json:"memories"`
	UseContext bool             `json:"use_context"`
}

type labMemoryResult struct {
	ID      string  `json:"id"`
	Content string  `json:"content"`
	Score   float32 `json:"score"`
}

type labSimulationResponse struct {
	Action           string            `json:"action"`
	NormalizedInput  string            `json:"normalized_input"`
	Response         string            `json:"response"`
	RelevantMemories []labMemoryResult `json:"relevant_memories"`
}

func writeLabError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": message})
}

func handleLabSimulate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		writeLabError(w, http.StatusMethodNotAllowed, "Use POST to run a Memory Lab simulation.")
		return
	}
	if origin := r.Header.Get("Origin"); origin != "" {
		parsed, err := url.Parse(origin)
		if err != nil || !strings.EqualFold(parsed.Host, r.Host) {
			writeLabError(w, http.StatusForbidden, "Cross-origin Memory Lab requests are not allowed.")
			return
		}
	}

	r.Body = http.MaxBytesReader(w, r.Body, 64*1024)
	var input labSimulationRequest
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&input); err != nil {
		writeLabError(w, http.StatusBadRequest, "Invalid Memory Lab request.")
		return
	}
	if _, err := decoder.Token(); err != io.EOF {
		writeLabError(w, http.StatusBadRequest, "Memory Lab accepts one JSON object per request.")
		return
	}
	input.Text = strings.TrimSpace(input.Text)
	if input.Text == "" {
		writeLabError(w, http.StatusBadRequest, "Enter a test input.")
		return
	}
	if len(input.Text) > maxInputBytes {
		writeLabError(w, http.StatusRequestEntityTooLarge, fmt.Sprintf("Test input exceeds the %d-byte limit.", maxInputBytes))
		return
	}
	if len(input.Memories) > 8 {
		writeLabError(w, http.StatusBadRequest, "Choose no more than eight temporary memories.")
		return
	}

	queryEmbedding := localEmbedding(input.Text)
	candidates := make([]MemoryMatch, 0, len(input.Memories))
	validIDs := make(map[string]bool, len(input.Memories))
	for index, saved := range input.Memories {
		content := strings.TrimSpace(saved.Content)
		if content == "" || len(content) > DefaultMemoryLimits().MaxEntryBytes {
			writeLabError(w, http.StatusBadRequest, "Temporary memories must contain text within the configured entry limit.")
			return
		}
		id := strings.TrimSpace(saved.ID)
		if id == "" {
			id = fmt.Sprintf("lab-%d", index+1)
		}
		if validIDs[id] {
			writeLabError(w, http.StatusBadRequest, "Temporary memory IDs must be unique.")
			return
		}
		validIDs[id] = true
		memory := Memory{
			ID:         id,
			Content:    content,
			Category:   normalizeCategory(saved.Category),
			Importance: normalizeImportance(saved.Importance),
			RelatedIDs: append([]string(nil), saved.RelatedIDs...),
			Embedding:  localEmbedding(content),
		}
		candidates = append(candidates, MemoryMatch{Memory: memory, Score: cosineSimilarity(queryEmbedding, memory.Embedding)})
	}
	sort.SliceStable(candidates, func(i, j int) bool { return candidates[i].Score > candidates[j].Score })
	if len(candidates) > 5 {
		candidates = candidates[:5]
	}

	ctx, cancel := context.WithTimeout(r.Context(), 90*time.Second)
	defer cancel()
	decision, err := classifyInput(ctx, input.Text, candidates)
	if err != nil {
		writeLabError(w, http.StatusBadGateway, "Could not classify this sandbox input with the local model.")
		return
	}

	prompt := "You are a warm, brief personal companion in an isolated sandbox. This is only an experiment: never claim that a memory was saved, changed, or forgotten.\n"
	contextMemories := make([]labMemoryResult, 0, len(candidates))
	if input.UseContext {
		remaining := maxContextBytes
		for _, candidate := range candidates {
			content := limitTextBytes(candidate.Memory.Content, remaining)
			if content == "" {
				break
			}
			prompt += fmt.Sprintf("Temporary sandbox memory: %s\n", content)
			remaining -= len(content)
			contextMemories = append(contextMemories, labMemoryResult{
				ID: candidate.Memory.ID, Content: content, Score: candidate.Score,
			})
			if remaining <= 0 {
				break
			}
		}
	}
	prompt += fmt.Sprintf("Simulated memory action: %s\nUser: %s\nAssistant:", decision.Action, decision.NormalizedInput)
	answer, err := requestOllama(ctx, prompt, false, nil, nil)
	if err != nil {
		writeLabError(w, http.StatusBadGateway, "Could not generate a local sandbox response.")
		return
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	_ = json.NewEncoder(w).Encode(labSimulationResponse{
		Action: decision.Action, NormalizedInput: decision.NormalizedInput,
		Response: strings.TrimSpace(answer), RelevantMemories: contextMemories,
	})
}

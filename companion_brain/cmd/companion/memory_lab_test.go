package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gorilla/websocket"
)

func TestHandleLabSimulateRejectsUnsupportedMethod(t *testing.T) {
	request := httptest.NewRequest(http.MethodGet, "/api/lab/simulate", nil)
	response := httptest.NewRecorder()
	handleLabSimulate(response, request)
	if response.Code != http.StatusMethodNotAllowed {
		t.Fatalf("GET status = %d, want %d", response.Code, http.StatusMethodNotAllowed)
	}
}

func TestHandleLabSimulateRejectsCrossOrigin(t *testing.T) {
	request := httptest.NewRequest(http.MethodPost, "/api/lab/simulate", strings.NewReader(`{"text":"hello"}`))
	request.Host = "localhost:8080"
	request.Header.Set("Origin", "https://attacker.example")
	response := httptest.NewRecorder()
	handleLabSimulate(response, request)
	if response.Code != http.StatusForbidden {
		t.Fatalf("cross-origin status = %d, want %d", response.Code, http.StatusForbidden)
	}
}

func TestHandleLabSimulateDoesNotMutateMemoryStore(t *testing.T) {
	previousStore := GlobalMemoryStore
	store, err := NewMemoryStore(t.TempDir() + "/memories.json")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.AddMemoryWithMetadata("My favorite lunch is tacos", "preferences", "medium", nil); err != nil {
		t.Fatal(err)
	}
	GlobalMemoryStore = store
	defer func() { GlobalMemoryStore = previousStore }()

	ollama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request OllamaRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode model request: %v", err)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		responseText := "A taco is a good lunch."
		if request.Format != nil {
			responseText = `{"action":"RESPOND","normalized_input":"What is a good lunch?","target_id":"","category":"other","importance":"medium","related_ids":[]}`
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(OllamaResponse{Response: responseText, Done: true})
	}))
	defer ollama.Close()
	t.Setenv("OLLAMA_URL", ollama.URL)

	body := `{"text":"What is a good lunch?","use_context":true,"memories":[{"id":"lab-1","content":"My favorite lunch is tacos","category":"preferences","importance":"medium"}]}`
	request := httptest.NewRequest(http.MethodPost, "/api/lab/simulate", strings.NewReader(body))
	request.Host = "example.test"
	response := httptest.NewRecorder()
	before := store.List()
	handleLabSimulate(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("simulation status = %d, body=%s", response.Code, response.Body.String())
	}
	var result labSimulationResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Action != "RESPOND" || result.Response != "A taco is a good lunch." || len(result.RelevantMemories) != 1 {
		t.Fatalf("unexpected Lab response: %+v", result)
	}
	after := store.List()
	if len(after) != len(before) || after[0].Content != before[0].Content {
		t.Fatalf("Lab simulation mutated persistent memories: before=%+v after=%+v", before, after)
	}
}

func TestApplyConfirmedStoreApprovalPersistsOnlyApprovedContent(t *testing.T) {
	previousStore := GlobalMemoryStore
	store, err := NewMemoryStore(t.TempDir() + "/approved.json")
	if err != nil {
		t.Fatal(err)
	}
	GlobalMemoryStore = store
	defer func() { GlobalMemoryStore = previousStore }()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		connection, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Errorf("upgrade test socket: %v", err)
			return
		}
		defer connection.Close()
		for {
			var incoming IncomingMessage
			if err := connection.ReadJSON(&incoming); err != nil {
				return
			}
			if err := applyConfirmedAction(connection, incoming); err != nil {
				_ = writeEvent(connection, OutgoingEvent{Type: "error", ProposalID: incoming.ProposalID, Content: err.Error()})
			}
		}
	}))
	defer server.Close()

	connectionURL := "ws" + strings.TrimPrefix(server.URL, "http")
	connection, _, err := websocket.DefaultDialer.Dial(connectionURL, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer connection.Close()
	if err := connection.WriteJSON(IncomingMessage{
		ConfirmAction: "APPROVE_STORE", Text: "I prefer quiet rooms", Category: "preferences", Importance: "high",
	}); err != nil {
		t.Fatal(err)
	}
	var event OutgoingEvent
	if err := connection.ReadJSON(&event); err != nil {
		t.Fatal(err)
	}
	if event.Type != "new_node" || event.Content != "I prefer quiet rooms" {
		t.Fatalf("unexpected approval event: %+v", event)
	}
	memories := store.List()
	if len(memories) != 1 || memories[0].Content != "I prefer quiet rooms" {
		t.Fatalf("approved memory was not persisted: %+v", memories)
	}
	if err := connection.WriteJSON(IncomingMessage{ConfirmAction: "EDIT_MEMORY", ID: event.ID, Text: "I prefer quiet rooms after work"}); err != nil {
		t.Fatal(err)
	}
	if err := connection.ReadJSON(&event); err != nil {
		t.Fatal(err)
	}
	if event.Type != "update_node" || event.Content != "I prefer quiet rooms after work" {
		t.Fatalf("unexpected edit event: %+v", event)
	}
	memories = store.List()
	if len(memories) != 1 || memories[0].Content != event.Content {
		t.Fatalf("edited memory was not persisted: %+v", memories)
	}
}

func TestHandleCompanionCoreQueuesStoreWithoutWriting(t *testing.T) {
	previousStore := GlobalMemoryStore
	store, err := NewMemoryStore(t.TempDir() + "/review.json")
	if err != nil {
		t.Fatal(err)
	}
	GlobalMemoryStore = store
	defer func() { GlobalMemoryStore = previousStore }()

	ollama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request OllamaRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode model request: %v", err)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		responseText := "Understood."
		if request.Format != nil {
			responseText = `{"action":"STORE","normalized_input":"My favorite kind of tea is peppermint.","target_id":"","category":"preferences","importance":"medium","related_ids":[]}`
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(OllamaResponse{Response: responseText, Done: true})
	}))
	defer ollama.Close()
	t.Setenv("OLLAMA_URL", ollama.URL)

	server := httptest.NewServer(http.HandlerFunc(handleCompanionCore))
	defer server.Close()
	connectionURL := "ws" + strings.TrimPrefix(server.URL, "http")
	connection, _, err := websocket.DefaultDialer.Dial(connectionURL, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer connection.Close()
	var restored OutgoingEvent
	if err := connection.ReadJSON(&restored); err != nil {
		t.Fatal(err)
	}
	if restored.Type != "restore_complete" {
		t.Fatalf("expected restore completion before client actions, got %+v", restored)
	}
	if err := connection.WriteJSON(IncomingMessage{Text: "My favorite kind of tea is peppermint.", ReviewMemory: true}); err != nil {
		t.Fatal(err)
	}
	var event OutgoingEvent
	if err := connection.ReadJSON(&event); err != nil {
		t.Fatal(err)
	}
	if event.Type != "memory_proposal" || event.Action != "STORE" || event.Content == "" {
		t.Fatalf("expected a pending memory proposal, got %+v", event)
	}
	if got := len(store.List()); got != 0 {
		t.Fatalf("review mode wrote %d memories before approval", got)
	}
}

func TestHandleCompanionCoreRecallsPreferredName(t *testing.T) {
	previousStore := GlobalMemoryStore
	store, err := NewMemoryStore(t.TempDir() + "/recall.json")
	if err != nil {
		t.Fatal(err)
	}
	saved, err := store.AddMemoryWithMetadata("Please call me Colin", "preferences", "high", nil)
	if err != nil {
		t.Fatal(err)
	}
	GlobalMemoryStore = store
	defer func() { GlobalMemoryStore = previousStore }()

	ollama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request OllamaRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode model request: %v", err)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		responseText := "Your name is Colin."
		if request.Format != nil {
			responseText = `{"action":"SEARCH","normalized_input":"What's my name?","target_id":"","category":"preferences","importance":"medium","related_ids":[]}`
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(OllamaResponse{Response: responseText, Done: true})
	}))
	defer ollama.Close()
	t.Setenv("OLLAMA_URL", ollama.URL)

	server := httptest.NewServer(http.HandlerFunc(handleCompanionCore))
	defer server.Close()
	connectionURL := "ws" + strings.TrimPrefix(server.URL, "http")
	connection, _, err := websocket.DefaultDialer.Dial(connectionURL, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer connection.Close()
	if err := connection.WriteJSON(IncomingMessage{Text: "What's my name?"}); err != nil {
		t.Fatal(err)
	}
	var foundStarted, foundResult, foundResponse bool
	for index := 0; index < 8 && !(foundResult && foundResponse); index++ {
		var event OutgoingEvent
		if err := connection.ReadJSON(&event); err != nil {
			t.Fatal(err)
		}
		switch event.Type {
		case "search_started":
			foundStarted = len(event.SearchCandidates) == 1 && event.SearchCandidates[0].ID == saved.ID
		case "search_result":
			foundResult = event.ID == saved.ID && event.Score > 0
		case "stream_text":
			foundResponse = strings.Contains(event.Content, "Colin")
		}
	}
	if !foundStarted || !foundResult || !foundResponse {
		t.Fatalf("name recall incomplete: started=%v result=%v response=%v", foundStarted, foundResult, foundResponse)
	}
}

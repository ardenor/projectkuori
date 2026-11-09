package main

import (
	"bytes"
	"encoding/binary"
	"math"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
)

func makeAudioUpload(t *testing.T, origin string, speech bool, steadyTone ...bool) *http.Request {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	file, err := writer.CreateFormFile("audio", "recording.wav")
	if err != nil {
		t.Fatal(err)
	}
	wavHeader := make([]byte, 44+16000*2)
	copy(wavHeader, "RIFF")
	binary.LittleEndian.PutUint32(wavHeader[4:8], uint32(len(wavHeader)-8))
	copy(wavHeader[8:], "WAVE")
	copy(wavHeader[12:], "fmt ")
	binary.LittleEndian.PutUint32(wavHeader[16:20], 16)
	binary.LittleEndian.PutUint16(wavHeader[20:22], 1)
	binary.LittleEndian.PutUint16(wavHeader[22:24], 1)
	binary.LittleEndian.PutUint32(wavHeader[24:28], 16000)
	binary.LittleEndian.PutUint32(wavHeader[28:32], 32000)
	binary.LittleEndian.PutUint16(wavHeader[32:34], 2)
	binary.LittleEndian.PutUint16(wavHeader[34:36], 16)
	copy(wavHeader[36:], "data")
	binary.LittleEndian.PutUint32(wavHeader[40:44], uint32(len(wavHeader)-44))
	if speech {
		for sample := 0; sample < 16000; sample++ {
			timeSeconds := float64(sample) / 16000
			envelope := 1.0
			if len(steadyTone) == 0 || !steadyTone[0] {
				envelope = 0.2 + 0.8*math.Abs(math.Sin(2*math.Pi*2.7*timeSeconds))
			}
			value := int16(math.Sin(float64(sample)*2*math.Pi*220/16000) * envelope * 5000)
			binary.LittleEndian.PutUint16(wavHeader[44+sample*2:], uint16(value))
		}
	}
	if _, err := file.Write(wavHeader); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, "/api/transcribe", &body)
	request.Header.Set("Content-Type", writer.FormDataContentType())
	if origin != "" {
		request.Header.Set("Origin", origin)
	}
	return request
}

func TestHandleTranscribeRejectsUnsupportedMethod(t *testing.T) {
	request := httptest.NewRequest(http.MethodGet, "/api/transcribe", nil)
	response := httptest.NewRecorder()
	handleTranscribe(response, request)
	if response.Code != http.StatusMethodNotAllowed {
		t.Fatalf("GET status = %d, want %d", response.Code, http.StatusMethodNotAllowed)
	}
}

func TestHandleTranscribeRejectsCrossOrigin(t *testing.T) {
	request := makeAudioUpload(t, "https://attacker.example", true)
	request.Host = "localhost:8080"
	response := httptest.NewRecorder()
	handleTranscribe(response, request)
	if response.Code != http.StatusForbidden {
		t.Fatalf("cross-origin status = %d, want %d", response.Code, http.StatusForbidden)
	}
}

func TestHandleTranscribeExplainsMissingModel(t *testing.T) {
	t.Setenv("WHISPER_MODEL", filepath.Join(t.TempDir(), "missing-model.bin"))
	request := makeAudioUpload(t, "http://example.test", true)
	request.Host = "example.test"
	response := httptest.NewRecorder()
	handleTranscribe(response, request)
	if response.Code != http.StatusServiceUnavailable {
		t.Fatalf("missing-model status = %d, want %d", response.Code, http.StatusServiceUnavailable)
	}
	if !strings.Contains(response.Body.String(), "./scripts/setup_voice.sh") {
		t.Fatalf("missing-model response lacks setup instruction: %s", response.Body.String())
	}
}

func TestHandleTranscribeRejectsSilence(t *testing.T) {
	request := makeAudioUpload(t, "http://example.test", false)
	request.Host = "example.test"
	response := httptest.NewRecorder()
	handleTranscribe(response, request)
	if response.Code != http.StatusUnprocessableEntity {
		t.Fatalf("silent-audio status = %d, want %d", response.Code, http.StatusUnprocessableEntity)
	}
	if !strings.Contains(response.Body.String(), "No speech was detected") {
		t.Fatalf("silent-audio response lacks actionable guidance: %s", response.Body.String())
	}
}

func TestHandleTranscribeRejectsSteadyTone(t *testing.T) {
	t.Setenv("WHISPER_MODEL", filepath.Join(t.TempDir(), "missing-model.bin"))
	request := makeAudioUpload(t, "http://example.test", true, true)
	request.Host = "example.test"
	response := httptest.NewRecorder()
	handleTranscribe(response, request)
	if response.Code != http.StatusUnprocessableEntity {
		t.Fatalf("steady-tone status = %d, want %d", response.Code, http.StatusUnprocessableEntity)
	}
	if !strings.Contains(response.Body.String(), "No changing speech") {
		t.Fatalf("steady-tone response lacks no-speech guidance: %s", response.Body.String())
	}
}

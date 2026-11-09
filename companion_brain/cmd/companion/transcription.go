package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const maxTranscriptionUploadBytes = 16 << 20

var transcriptionSlots = make(chan struct{}, 1)

type transcriptionResponse struct {
	Text  string `json:"text,omitempty"`
	Error string `json:"error,omitempty"`
}

type whisperTranscript struct {
	Segments []struct {
		Text         string  `json:"text"`
		NoSpeechProb float64 `json:"no_speech_prob"`
	} `json:"transcription"`
}

func whisperModelPath() (string, error) {
	if path := os.Getenv("WHISPER_MODEL"); path != "" {
		return path, nil
	}
	cacheDir, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("locate user cache directory: %w", err)
	}
	return filepath.Join(cacheDir, "companion-core", "whisper", "ggml-small.en.bin"), nil
}

func writeTranscriptionResponse(w http.ResponseWriter, status int, response transcriptionResponse) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(response)
}

func validatePCM16WAV(data []byte) error {
	if len(data) < 46 || string(data[:4]) != "RIFF" || string(data[8:12]) != "WAVE" || string(data[12:16]) != "fmt " || string(data[36:40]) != "data" {
		return errors.New("recording must be a PCM WAV file")
	}
	if binary.LittleEndian.Uint16(data[20:22]) != 1 ||
		binary.LittleEndian.Uint16(data[22:24]) != 1 ||
		binary.LittleEndian.Uint32(data[24:28]) != 16000 ||
		binary.LittleEndian.Uint16(data[34:36]) != 16 {
		return errors.New("recording must be 16 kHz mono PCM audio")
	}
	if (len(data)-44)%2 != 0 {
		return errors.New("recording contains an incomplete PCM sample")
	}

	sampleCount := (len(data) - 44) / 2
	var squareSum float64
	peak := 0
	windowFrames := 1600
	windowLevels := make([]float64, 0, (sampleCount+windowFrames-1)/windowFrames)
	windowCrossings := make([]float64, 0, cap(windowLevels))
	windowSquareSum := float64(0)
	windowSamples := 0
	windowZeroCrossings := 0
	previousSample := 0
	for offset := 44; offset < len(data); offset += 2 {
		sample := int(binary.LittleEndian.Uint16(data[offset : offset+2]))
		if sample >= 1<<15 {
			sample -= 1 << 16
		}
		absolute := int(math.Abs(float64(sample)))
		if absolute > peak {
			peak = absolute
		}
		squareSum += float64(sample * sample)
		windowSquareSum += float64(sample * sample)
		if windowSamples > 0 && ((sample < 0 && previousSample >= 0) || (sample >= 0 && previousSample < 0)) {
			windowZeroCrossings++
		}
		previousSample = sample
		windowSamples++
		if windowSamples == windowFrames {
			windowLevels = append(windowLevels, math.Sqrt(windowSquareSum/float64(windowSamples)))
			windowCrossings = append(windowCrossings, float64(windowZeroCrossings)/float64(windowSamples))
			windowSquareSum = 0
			windowSamples = 0
			windowZeroCrossings = 0
		}
	}
	if windowSamples > windowFrames/2 {
		windowLevels = append(windowLevels, math.Sqrt(windowSquareSum/float64(windowSamples)))
		windowCrossings = append(windowCrossings, float64(windowZeroCrossings)/float64(windowSamples))
	}
	if sampleCount == 0 || peak < 300 || math.Sqrt(squareSum/float64(sampleCount)) < 120 {
		return errors.New("No speech was detected. Speak closer to the microphone and try again.")
	}
	if len(windowLevels) >= 6 {
		var levelMean, crossingMean float64
		for index, level := range windowLevels {
			levelMean += level
			crossingMean += windowCrossings[index]
		}
		levelMean /= float64(len(windowLevels))
		crossingMean /= float64(len(windowCrossings))
		var levelVariance, crossingVariance float64
		for index, level := range windowLevels {
			levelVariance += (level - levelMean) * (level - levelMean)
			crossingVariance += (windowCrossings[index] - crossingMean) * (windowCrossings[index] - crossingMean)
		}
		levelVariation := math.Sqrt(levelVariance/float64(len(windowLevels))) / math.Max(1, levelMean)
		crossingVariation := math.Sqrt(crossingVariance / float64(len(windowCrossings)))
		if levelVariation < 0.025 && crossingVariation < 0.003 {
			return errors.New("No changing speech was detected. Speak a short phrase and try again.")
		}
	}
	return nil
}

func handleTranscribe(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		writeTranscriptionResponse(w, http.StatusMethodNotAllowed, transcriptionResponse{Error: "Use POST to transcribe audio."})
		return
	}
	if origin := r.Header.Get("Origin"); origin != "" {
		parsed, err := url.Parse(origin)
		if err != nil || !strings.EqualFold(parsed.Host, r.Host) {
			writeTranscriptionResponse(w, http.StatusForbidden, transcriptionResponse{Error: "Cross-origin transcription requests are not allowed."})
			return
		}
	}

	r.Body = http.MaxBytesReader(w, r.Body, maxTranscriptionUploadBytes)
	if err := r.ParseMultipartForm(1 << 20); err != nil {
		var maxBytesError *http.MaxBytesError
		if errors.As(err, &maxBytesError) {
			writeTranscriptionResponse(w, http.StatusRequestEntityTooLarge, transcriptionResponse{Error: "Audio recording is too large."})
		} else {
			writeTranscriptionResponse(w, http.StatusBadRequest, transcriptionResponse{Error: "Upload a WAV recording in the audio field."})
		}
		return
	}
	audioFile, _, err := r.FormFile("audio")
	if err != nil {
		writeTranscriptionResponse(w, http.StatusBadRequest, transcriptionResponse{Error: "Upload a WAV recording in the audio field."})
		return
	}
	defer audioFile.Close()
	audioData, err := io.ReadAll(io.LimitReader(audioFile, maxTranscriptionUploadBytes+1))
	if err != nil {
		writeTranscriptionResponse(w, http.StatusBadRequest, transcriptionResponse{Error: "Could not read the audio recording."})
		return
	}
	if len(audioData) > maxTranscriptionUploadBytes {
		writeTranscriptionResponse(w, http.StatusRequestEntityTooLarge, transcriptionResponse{Error: "Audio recording is too large."})
		return
	}
	if err := validatePCM16WAV(audioData); err != nil {
		status := http.StatusBadRequest
		if strings.HasPrefix(err.Error(), "No ") {
			status = http.StatusUnprocessableEntity
		}
		writeTranscriptionResponse(w, status, transcriptionResponse{Error: err.Error()})
		return
	}

	modelPath, err := whisperModelPath()
	if err != nil {
		writeTranscriptionResponse(w, http.StatusInternalServerError, transcriptionResponse{Error: "Could not locate the local speech model directory."})
		return
	}
	if info, err := os.Stat(modelPath); err != nil || !info.Mode().IsRegular() {
		writeTranscriptionResponse(w, http.StatusServiceUnavailable, transcriptionResponse{Error: "Local speech model is not installed. Run ./scripts/setup_voice.sh."})
		return
	}
	whisperCLI := os.Getenv("WHISPER_CLI")
	if whisperCLI == "" {
		whisperCLI = "whisper-cli"
	}
	whisperPath, err := exec.LookPath(whisperCLI)
	if err != nil {
		writeTranscriptionResponse(w, http.StatusServiceUnavailable, transcriptionResponse{Error: "Local speech engine is not installed. Run ./scripts/setup_voice.sh."})
		return
	}

	select {
	case transcriptionSlots <- struct{}{}:
		defer func() { <-transcriptionSlots }()
	default:
		writeTranscriptionResponse(w, http.StatusTooManyRequests, transcriptionResponse{Error: "Local speech recognition is busy. Try again shortly."})
		return
	}

	tempDir, err := os.MkdirTemp("", "companion-transcription-*")
	if err != nil {
		writeTranscriptionResponse(w, http.StatusInternalServerError, transcriptionResponse{Error: "Could not prepare the local transcription request."})
		return
	}
	defer os.RemoveAll(tempDir)
	audioPath := filepath.Join(tempDir, "recording.wav")
	if err := os.WriteFile(audioPath, audioData, 0600); err != nil {
		writeTranscriptionResponse(w, http.StatusInternalServerError, transcriptionResponse{Error: "Could not prepare the local audio recording."})
		return
	}
	outputPrefix := filepath.Join(tempDir, "transcript")
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Minute)
	defer cancel()
	command := exec.CommandContext(ctx, whisperPath,
		"-m", modelPath,
		"-f", audioPath,
		"-l", "en",
		"-bs", "5",
		"-bo", "5",
		"-nt",
		"-otxt",
		"-ojf",
		"-of", outputPrefix,
		"-np",
	)
	if _, err := command.CombinedOutput(); err != nil {
		log.Printf("local speech transcription failed: %v", err)
		writeTranscriptionResponse(w, http.StatusBadGateway, transcriptionResponse{Error: "Local transcription failed. Check the Whisper installation and model."})
		return
	}
	transcript, err := os.ReadFile(outputPrefix + ".txt")
	if err != nil {
		writeTranscriptionResponse(w, http.StatusBadGateway, transcriptionResponse{Error: "The local speech engine returned no transcript."})
		return
	}
	text := strings.TrimSpace(string(transcript))
	if text == "" {
		writeTranscriptionResponse(w, http.StatusUnprocessableEntity, transcriptionResponse{Error: "No speech was detected. Try again and speak clearly."})
		return
	}
	jsonTranscript, err := os.ReadFile(outputPrefix + ".json")
	if err != nil {
		writeTranscriptionResponse(w, http.StatusBadGateway, transcriptionResponse{Error: "The local speech engine returned no confidence data."})
		return
	}
	var confidence whisperTranscript
	if err := json.Unmarshal(jsonTranscript, &confidence); err != nil {
		writeTranscriptionResponse(w, http.StatusBadGateway, transcriptionResponse{Error: "The local speech engine returned invalid confidence data."})
		return
	}
	speechDetected := false
	for _, segment := range confidence.Segments {
		if strings.TrimSpace(segment.Text) != "" && segment.NoSpeechProb < 0.6 {
			speechDetected = true
			break
		}
	}
	if !speechDetected {
		writeTranscriptionResponse(w, http.StatusUnprocessableEntity, transcriptionResponse{Error: "No clear speech was detected. Speak closer to the microphone and try again."})
		return
	}
	writeTranscriptionResponse(w, http.StatusOK, transcriptionResponse{Text: text})
}

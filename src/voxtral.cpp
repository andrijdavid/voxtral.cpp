#include "voxtral.h"
#include "gguf.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif
#ifdef GGML_USE_BLAS
#include "ggml-blas.h"
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>

// ============================================================================
// Internal constants
// ============================================================================

static constexpr float VOXTRAL_PI = 3.14159265358979323846f;
static constexpr int32_t VOXTRAL_N_FFT       = VOXTRAL_WINDOW_SIZE;         // 400
static constexpr int32_t VOXTRAL_N_FREQ      = VOXTRAL_N_FFT / 2 + 1;      // 201
static constexpr int32_t VOXTRAL_ENC_CHUNK_MEL     = 3000;  // mel frames per encoder chunk
static constexpr int32_t VOXTRAL_ENC_CHUNK_OVERLAP  = 750;  // overlap in encoder-token space (= window)
static constexpr int32_t VOXTRAL_MAX_ENC_CHUNK      = 2000; // max enc tokens per single chunk

// ============================================================================
// Logging helper
// ============================================================================

#define LOG(ctx_ptr, lvl, ...) \
    do { \
        if ((ctx_ptr)->logger && static_cast<int>(lvl) <= static_cast<int>((ctx_ptr)->log_level)) { \
            char _buf[2048]; \
            snprintf(_buf, sizeof(_buf), __VA_ARGS__); \
            (ctx_ptr)->logger(lvl, std::string(_buf)); \
        } \
    } while (0)

#define LOG_INFO(ctx_ptr, ...)  LOG(ctx_ptr, voxtral_log_level::info,  __VA_ARGS__)
#define LOG_WARN(ctx_ptr, ...)  LOG(ctx_ptr, voxtral_log_level::warn,  __VA_ARGS__)
#define LOG_ERR(ctx_ptr, ...)   LOG(ctx_ptr, voxtral_log_level::error, __VA_ARGS__)
#define LOG_DBG(ctx_ptr, ...)   LOG(ctx_ptr, voxtral_log_level::debug, __VA_ARGS__)

// ============================================================================
// Weight structures (internal)
// ============================================================================

struct voxtral_encoder_layer {
    ggml_tensor * attn_norm_weight;  // [enc_dim]
    ggml_tensor * attn_q_weight;     // [enc_heads*enc_head_dim, enc_dim]
    ggml_tensor * attn_q_bias;       // [enc_heads*enc_head_dim]
    ggml_tensor * attn_k_weight;     // [enc_kv_heads*enc_head_dim, enc_dim]
    ggml_tensor * attn_v_weight;     // [enc_kv_heads*enc_head_dim, enc_dim]
    ggml_tensor * attn_v_bias;       // [enc_kv_heads*enc_head_dim]
    ggml_tensor * attn_o_weight;     // [enc_dim, enc_heads*enc_head_dim]
    ggml_tensor * attn_o_bias;       // [enc_dim]
    ggml_tensor * ffn_norm_weight;   // [enc_dim]
    ggml_tensor * ffn_w1_weight;     // [enc_hidden, enc_dim]
    ggml_tensor * ffn_w2_weight;     // [enc_dim, enc_hidden]
    ggml_tensor * ffn_w2_bias;       // [enc_dim]
    ggml_tensor * ffn_w3_weight;     // [enc_hidden, enc_dim]
};

struct voxtral_decoder_layer {
    ggml_tensor * attn_norm_weight;  // [dec_dim]
    ggml_tensor * attn_q_weight;     // [dec_heads*dec_head_dim, dec_dim]
    ggml_tensor * attn_k_weight;     // [dec_kv_heads*dec_head_dim, dec_dim]
    ggml_tensor * attn_v_weight;     // [dec_kv_heads*dec_head_dim, dec_dim]
    ggml_tensor * attn_o_weight;     // [dec_dim, dec_heads*dec_head_dim]
    ggml_tensor * ffn_norm_weight;   // [dec_dim]
    ggml_tensor * ffn_w1_weight;     // [dec_hidden, dec_dim]
    ggml_tensor * ffn_w2_weight;     // [dec_dim, dec_hidden]
    ggml_tensor * ffn_w3_weight;     // [dec_hidden, dec_dim]
    ggml_tensor * ada0_weight;       // [ada_dim, dec_dim]
    ggml_tensor * ada2_weight;       // [dec_dim, ada_dim]
};

// ============================================================================
// Model structure
// ============================================================================

struct voxtral_model {
    // Encoder conv stem
    ggml_tensor * enc_conv0_weight;  // [enc_dim, num_mel_bins, 3]
    ggml_tensor * enc_conv0_bias;    // [enc_dim]
    ggml_tensor * enc_conv1_weight;  // [enc_dim, enc_dim, 3]
    ggml_tensor * enc_conv1_bias;    // [enc_dim]
    std::vector<voxtral_encoder_layer> enc_layers;
    ggml_tensor * enc_norm_weight;   // [enc_dim]

    // Adapter
    ggml_tensor * adapter_0_weight;  // [dec_dim, enc_dim*downsample]
    ggml_tensor * adapter_2_weight;  // [dec_dim, dec_dim]

    // Decoder
    ggml_tensor * tok_embeddings_weight; // [vocab_size, dec_dim]
    std::vector<voxtral_decoder_layer> dec_layers;
    ggml_tensor * dec_norm_weight;   // [dec_dim]

    // Mel filters (stored in GGUF)
    ggml_tensor * mel_filters;       // [n_freq, n_mel] = [201, 128]

    // Tokenizer (Tekken vocab)
    int32_t tokenizer_num_special_tokens = 1000;
    std::unordered_set<int32_t> tokenizer_special_ranks;
    std::vector<std::string> tokenizer_vocab_b64;
    mutable std::unordered_map<int32_t, std::string> tokenizer_bytes_cache;

    // Owning contexts
    ggml_context * ctx_gguf   = nullptr;
    gguf_context * gguf_ctx   = nullptr;
    ggml_backend_buffer_t buf_weights = nullptr;
    ggml_backend_t         backend_weights = nullptr;
    bool                   weights_on_gpu = false;
    voxtral_gpu_backend    gpu_type = voxtral_gpu_backend::none;
};

// ============================================================================
// Context structure
// ============================================================================

struct voxtral_context {
    voxtral_model        * model     = nullptr;
    voxtral_log_level      log_level = voxtral_log_level::info;
    voxtral_log_callback   logger    = nullptr;
    int32_t                n_threads = 4;

    // Backend
    ggml_backend_t         backend      = nullptr;
    ggml_backend_t         backend_cpu  = nullptr;
    ggml_backend_t         blas_backend = nullptr;
    voxtral_gpu_backend    gpu_type     = voxtral_gpu_backend::none;

    // Persistent device tensors (allocated once)
    ggml_context       * ctx_persistent = nullptr;
    ggml_backend_buffer_t buf_persistent = nullptr;

    // Per-chunk encoder output (fixed size, reused each chunk)
    ggml_tensor * encoder_chunk_output = nullptr;  // [enc_dim, MAX_ENC_CHUNK]
    ggml_tensor * decoder_logits  = nullptr;  // [vocab_size]

    // KV cache: [kv_heads*head_dim, dec_window, dec_layers]
    ggml_tensor * kv_self_k       = nullptr;
    ggml_tensor * kv_self_v       = nullptr;

    // Full accumulated encoder output (dynamic, allocated per utterance ON DEVICE)
    ggml_context       * ctx_enc_full = nullptr;
    ggml_backend_buffer_t buf_enc_full = nullptr;
    ggml_tensor        * encoder_output = nullptr;  // [enc_dim, total_enc_tokens]
    int32_t total_enc_tokens = 0;

    // Dynamic decoder memory (allocated per utterance ON DEVICE)
    ggml_context       * ctx_dec_mem = nullptr;
    ggml_backend_buffer_t buf_dec_mem = nullptr;
    ggml_tensor        * decoder_memory = nullptr;  // [dec_dim, dec_seq]

    // Actual sizes (set per utterance)
    int32_t enc_seq_len  = 0;  // after conv, before left-trunc
    int32_t enc_seq_used = 0;  // after left-trunc (multiple of downsample_factor)
    int32_t dec_seq_len  = 0;  // adapter output length

    // KV ring buffer state
    int32_t kv_used      = 0;  // tokens currently in KV cache

    // Schedulers
    ggml_backend_sched_t sched_encoder  = nullptr;
    ggml_backend_sched_t sched_adapter  = nullptr;
    ggml_backend_sched_t sched_dec_pre  = nullptr;
    ggml_backend_sched_t sched_dec_step = nullptr;

    // CPU scratch
    std::vector<float> hann_window;     // [window_size]
    std::vector<float> mel_filters_cpu; // [n_freq * n_mel]
    std::vector<float> time_emb_cpu;    // [dec_dim]
};

// ============================================================================
// Mel filterbank computation (Slaney-style, matches Python reference)
// ============================================================================

static float hertz_to_mel(float freq_hz) {
    constexpr float min_log_hertz = 1000.0f;
    constexpr float min_log_mel   = 15.0f;
    const float logstep       = 27.0f / logf(6.4f);
    float mels = 3.0f * freq_hz / 200.0f;
    if (freq_hz >= min_log_hertz) {
        mels = min_log_mel + logf(freq_hz / min_log_hertz) * logstep;
    }
    return mels;
}

static float mel_to_hertz(float mels) {
    constexpr float min_log_hertz = 1000.0f;
    constexpr float min_log_mel   = 15.0f;
    const float logstep       = logf(6.4f) / 27.0f;
    float freq = 200.0f * mels / 3.0f;
    if (mels >= min_log_mel) {
        freq = min_log_hertz * expf(logstep * (mels - min_log_mel));
    }
    return freq;
}

static void compute_mel_filters_slaney(std::vector<float> & filters) {
    // Output: filters[k * n_mel + m] for k in [0..n_freq), m in [0..n_mel)
    // Matches Python compute_mel_filters() exactly
    constexpr int32_t n_freq = VOXTRAL_N_FREQ;  // 201
    constexpr int32_t n_mel  = VOXTRAL_NUM_MEL_BINS;  // 128

    filters.resize(n_freq * n_mel, 0.0f);

    // FFT frequencies: linspace(0, sr/2, n_freq)
    std::vector<float> fft_freqs(n_freq);
    for (int32_t i = 0; i < n_freq; i++) {
        fft_freqs[i] = (float)(VOXTRAL_SAMPLE_RATE / 2) * (float)i / (float)(n_freq - 1);
    }

    // Mel frequencies: linspace(mel(0), mel(8000), n_mel+2)
    const float mel_min = hertz_to_mel(0.0f);
    const float mel_max = hertz_to_mel(8000.0f);

    std::vector<float> mel_pts(n_mel + 2);
    for (int32_t i = 0; i < n_mel + 2; i++) {
        mel_pts[i] = mel_min + (mel_max - mel_min) * (float)i / (float)(n_mel + 1);
    }

    std::vector<float> filter_freqs(n_mel + 2);
    for (int32_t i = 0; i < n_mel + 2; i++) {
        filter_freqs[i] = mel_to_hertz(mel_pts[i]);
    }

    // Build triangular filters (matching Python slopes approach)
    for (int32_t m = 0; m < n_mel; m++) {
        const float f_left   = filter_freqs[m];
        const float f_center = filter_freqs[m + 1];
        const float f_right  = filter_freqs[m + 2];
        const float enorm    = 2.0f / (f_right - f_left);

        for (int32_t k = 0; k < n_freq; k++) {
            const float f = fft_freqs[k];
            float down_slope = -(f - f_center) / (f_center - f_left);   // -slopes[:, :-2] / filter_diff[:-1]
            float up_slope   =  (f_right - f)  / (f_right - f_center);  // slopes[:, 2:] / filter_diff[1:]

            float val = std::max(0.0f, std::min(down_slope, up_slope));
            filters[k * n_mel + m] = val * enorm;
        }
    }
}

// ============================================================================
// Time embedding (sinusoidal, matches Python compute_time_embedding)
// ============================================================================

static void compute_time_embedding(std::vector<float> & out, float t, int32_t dim) {
    // Python: inv_freq = exp(-log(10000) * arange(half) / half)
    //         emb = t * inv_freq;  return cat([cos(emb), sin(emb)])
    out.resize(dim);
    const int32_t half = dim / 2;
    for (int32_t i = 0; i < half; i++) {
        const float inv_freq = expf(-logf(10000.0f) * (float)i / (float)half);
        const float angle = t * inv_freq;
        out[i]        = cosf(angle);   // cos first half
        out[i + half] = sinf(angle);   // sin second half
    }
}

static double elapsed_ms(const std::chrono::steady_clock::time_point & t0) {
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ============================================================================
// Token decode helpers (Tekken vocab from GGUF metadata)
// ============================================================================

static std::vector<uint8_t> base64_decode(const std::string & in) {
    static const std::array<int8_t, 256> table = [] {
        std::array<int8_t, 256> t{};
        t.fill(-1);
        for (int c = 'A'; c <= 'Z'; ++c) t[static_cast<size_t>(c)] = static_cast<int8_t>(c - 'A');
        for (int c = 'a'; c <= 'z'; ++c) t[static_cast<size_t>(c)] = static_cast<int8_t>(26 + (c - 'a'));
        for (int c = '0'; c <= '9'; ++c) t[static_cast<size_t>(c)] = static_cast<int8_t>(52 + (c - '0'));
        t[static_cast<size_t>('+')] = 62;
        t[static_cast<size_t>('/')] = 63;
        return t;
    }();

    std::vector<uint8_t> out;
    out.reserve((in.size() * 3) / 4 + 4);

    uint32_t acc = 0;
    int bits = 0;

    for (char ch : in) {
        if (ch == '=') {
            break;
        }

        const uint8_t uch = static_cast<uint8_t>(ch);
        const int8_t val = table[uch];
        if (val < 0) {
            continue;
        }

        acc = (acc << 6) | static_cast<uint32_t>(val);
        bits += 6;

        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((acc >> bits) & 0xFF));
        }
    }

    return out;
}

static const std::string & token_bytes_for_id(const voxtral_model & model, int32_t token_id) {
    auto it_cached = model.tokenizer_bytes_cache.find(token_id);
    if (it_cached != model.tokenizer_bytes_cache.end()) {
        return it_cached->second;
    }

    std::string decoded;
    if (token_id >= 0 &&
        token_id >= model.tokenizer_num_special_tokens &&
        model.tokenizer_special_ranks.find(token_id) == model.tokenizer_special_ranks.end()) {
        const int64_t vocab_id = static_cast<int64_t>(token_id) -
                                 static_cast<int64_t>(model.tokenizer_num_special_tokens);
        if (vocab_id >= 0 && vocab_id < static_cast<int64_t>(model.tokenizer_vocab_b64.size())) {
            const std::vector<uint8_t> bytes =
                base64_decode(model.tokenizer_vocab_b64[static_cast<size_t>(vocab_id)]);
            decoded.assign(reinterpret_cast<const char *>(bytes.data()), bytes.size());
        }
    }

    auto [it_new, _] = model.tokenizer_bytes_cache.emplace(token_id, std::move(decoded));
    return it_new->second;
}

static std::string decode_tokens(const voxtral_model & model, const std::vector<int32_t> & tokens) {
    if (model.tokenizer_vocab_b64.empty()) {
        return {};
    }

    std::string out;
    out.reserve(tokens.size() * 3);

    for (int32_t token : tokens) {
        if (token < model.tokenizer_num_special_tokens) {
            continue;
        }
        if (model.tokenizer_special_ranks.find(token) != model.tokenizer_special_ranks.end()) {
            continue;
        }

        const std::string & token_bytes = token_bytes_for_id(model, token);
        out.append(token_bytes);
    }

    return out;
}

// ============================================================================
// Reflect padding helper (matches PyTorch pad(mode="reflect"))
// ============================================================================

static inline int32_t reflect_index(int32_t idx, int32_t len) {
    if (len <= 1) {
        return 0;
    }
    while (idx < 0 || idx >= len) {
        if (idx < 0) {
            idx = -idx;
        } else {
            idx = 2 * len - 2 - idx;
        }
    }
    return idx;
}

// ============================================================================
// WAV file loading (16-bit PCM or 32-bit float, mono/stereo)
// ============================================================================

static bool load_wav_file(const std::string & path, std::vector<float> & audio_out) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) return false;

    // RIFF header
    char riff[4]; fin.read(riff, 4);
    if (memcmp(riff, "RIFF", 4) != 0) return false;

    uint32_t chunk_size; fin.read(reinterpret_cast<char*>(&chunk_size), 4);
    char wave[4]; fin.read(wave, 4);
    if (memcmp(wave, "WAVE", 4) != 0) return false;

    uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0, data_size = 0;
    bool found_fmt = false, found_data = false;

    while (fin.good() && !(found_fmt && found_data)) {
        char sub_id[4]; fin.read(sub_id, 4);
        uint32_t sub_size; fin.read(reinterpret_cast<char*>(&sub_size), 4);
        if (!fin.good()) break;

        if (memcmp(sub_id, "fmt ", 4) == 0) {
            fin.read(reinterpret_cast<char*>(&audio_format),    2);
            fin.read(reinterpret_cast<char*>(&num_channels),    2);
            fin.read(reinterpret_cast<char*>(&sample_rate),     4);
            uint32_t byte_rate; fin.read(reinterpret_cast<char*>(&byte_rate), 4);
            uint16_t block_align; fin.read(reinterpret_cast<char*>(&block_align), 2);
            fin.read(reinterpret_cast<char*>(&bits_per_sample), 2);
            if (sub_size > 16) fin.seekg(sub_size - 16, std::ios::cur);
            found_fmt = true;
        } else if (memcmp(sub_id, "data", 4) == 0) {
            data_size = sub_size;
            found_data = true;
        } else {
            fin.seekg(sub_size, std::ios::cur);
        }
    }

    if (!found_fmt || !found_data) return false;
    if (audio_format != 1 && audio_format != 3) return false; // 1=PCM, 3=IEEE float

    const int32_t n_samples_total = data_size / (bits_per_sample / 8);
    const int32_t n_samples = n_samples_total / num_channels;

    if (audio_format == 1 && bits_per_sample == 16) {
        std::vector<int16_t> raw(n_samples_total);
        fin.read(reinterpret_cast<char*>(raw.data()), data_size);
        audio_out.resize(n_samples);
        for (int32_t i = 0; i < n_samples; i++) {
            float sum = 0.0f;
            for (int32_t c = 0; c < num_channels; c++) {
                sum += (float)raw[i * num_channels + c] / 32768.0f;
            }
            audio_out[i] = sum / num_channels;
        }
    } else if (audio_format == 3 && bits_per_sample == 32) {
        std::vector<float> raw(n_samples_total);
        fin.read(reinterpret_cast<char*>(raw.data()), data_size);
        audio_out.resize(n_samples);
        for (int32_t i = 0; i < n_samples; i++) {
            float sum = 0.0f;
            for (int32_t c = 0; c < num_channels; c++) {
                sum += raw[i * num_channels + c];
            }
            audio_out[i] = sum / num_channels;
        }
    } else {
        return false;
    }

    return true;
}

// ============================================================================
// Mel spectrogram computation (CPU, matches Python compute_mel_spectrogram)
// ============================================================================

struct stft_plan {
    int32_t n_fft = 0;
    int32_t n_bins = 0;
    std::vector<float> cos_table;
    std::vector<float> sin_table;
};

static const stft_plan & get_stft_plan() {
    static stft_plan plan = []() {
        stft_plan p;
        p.n_fft  = VOXTRAL_N_FFT;
        p.n_bins = VOXTRAL_N_FREQ;
        p.cos_table.resize((size_t) p.n_bins * (size_t) p.n_fft);
        p.sin_table.resize((size_t) p.n_bins * (size_t) p.n_fft);
        for (int32_t k = 0; k < p.n_bins; ++k) {
            for (int32_t n = 0; n < p.n_fft; ++n) {
                const float angle = 2.0f * VOXTRAL_PI * (float) k * (float) n / (float) p.n_fft;
                const size_t idx = (size_t) k * (size_t) p.n_fft + (size_t) n;
                p.cos_table[idx] = cosf(angle);
                p.sin_table[idx] = sinf(angle);
            }
        }
        return p;
    }();
    return plan;
}

static void compute_mel_spectrogram(
    const float * audio,
    int32_t       n_samples,
    const float * mel_filters,   // [n_freq * n_mel]
    const float * hann_window,   // [window_size]
    float       * mel_out,       // [n_mel, n_frames]  (pre-allocated)
    int32_t     * out_n_frames)
{
    // torch.stft with window_size, hop_length, return_complex=True
    // produces (n_freq, n_stft_frames) where n_stft_frames = n_samples/hop + 1
    // Then magnitudes = stft[..., :-1].abs()**2  -> drops last frame
    const int32_t n_stft_frames = n_samples / VOXTRAL_HOP_LENGTH + 1;
    const int32_t n_frames = n_stft_frames - 1;  // drop last frame (matching Python [:-1])
    *out_n_frames = n_frames;

    constexpr int32_t n_freq = VOXTRAL_N_FREQ;
    constexpr int32_t n_mel  = VOXTRAL_NUM_MEL_BINS;
    constexpr int32_t n_fft  = VOXTRAL_N_FFT;
    constexpr int32_t hop    = VOXTRAL_HOP_LENGTH;
    constexpr int32_t pad    = n_fft / 2;

    if (n_frames <= 0) {
        return;
    }

    const stft_plan & plan = get_stft_plan();

    // Reflect padding once (equivalent to center=True, pad_mode="reflect")
    const int32_t centered_len = n_samples + 2 * pad;
    std::vector<float> centered((size_t) centered_len, 0.0f);
    if (n_samples > 0) {
        for (int32_t i = 0; i < centered_len; ++i) {
            const int32_t src = i - pad;
            const int32_t ridx = (src >= 0 && src < n_samples) ? src : reflect_index(src, n_samples);
            centered[(size_t) i] = audio[(size_t) ridx];
        }
    }

    // Pre-allocate per-call buffers
    std::vector<float> windowed((size_t) n_fft);
    std::vector<float> power((size_t) n_freq);
    std::vector<float> mel_accum((size_t) n_mel);

    for (int32_t frame = 0; frame < n_frames; ++frame) {
        const int32_t start = frame * hop;
        const float * frame_ptr = centered.data() + (size_t) start;

        for (int32_t i = 0; i < n_fft; ++i) {
            windowed[(size_t) i] = frame_ptr[(size_t) i] * hann_window[(size_t) i];
        }

        // DFT with precomputed sin/cos tables
        for (int32_t k = 0; k < n_freq; ++k) {
            const float * cos_row = plan.cos_table.data() + (size_t) k * (size_t) n_fft;
            const float * sin_row = plan.sin_table.data() + (size_t) k * (size_t) n_fft;
            float re = 0.0f;
            float im = 0.0f;

            int32_t i = 0;
            for (; i + 3 < n_fft; i += 4) {
                const float x0 = windowed[(size_t) i + 0];
                const float x1 = windowed[(size_t) i + 1];
                const float x2 = windowed[(size_t) i + 2];
                const float x3 = windowed[(size_t) i + 3];

                re += x0 * cos_row[i + 0] + x1 * cos_row[i + 1] + x2 * cos_row[i + 2] + x3 * cos_row[i + 3];
                im -= x0 * sin_row[i + 0] + x1 * sin_row[i + 1] + x2 * sin_row[i + 2] + x3 * sin_row[i + 3];
            }
            for (; i < n_fft; ++i) {
                const float x = windowed[(size_t) i];
                re += x * cos_row[i];
                im -= x * sin_row[i];
            }

            power[(size_t) k] = re * re + im * im;
        }

        // Apply mel filterbank (k-major for cache-friendly access)
        std::fill(mel_accum.begin(), mel_accum.end(), 0.0f);
        for (int32_t k = 0; k < n_freq; ++k) {
            const float * w = mel_filters + (size_t) k * (size_t) n_mel;
            const float  pk = power[(size_t) k];
            for (int32_t m = 0; m < n_mel; ++m) {
                mel_accum[(size_t) m] += w[m] * pk;
            }
        }

        for (int32_t m = 0; m < n_mel; ++m) {
            float val = mel_accum[(size_t) m];
            val = std::max(val, 1e-10f);
            val = log10f(val);
            val = std::max(val, VOXTRAL_GLOBAL_LOG_MEL_MAX - 8.0f);
            val = (val + 4.0f) / 4.0f;
            mel_out[(size_t) m * (size_t) n_frames + (size_t) frame] = val;
        }
    }
}

// ============================================================================
// GGUF tensor loading helper
// ============================================================================

static ggml_tensor * get_tensor(ggml_context * ctx, const char * name) {
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) {
        fprintf(stderr, "voxtral: tensor '%s' not found in GGUF\n", name);
    }
    return t;
}

// ============================================================================
// Model loading
// ============================================================================

voxtral_model * voxtral_model_load_from_file(
    const std::string    & path,
    voxtral_log_callback   logger,
    voxtral_gpu_backend    gpu)
{
    auto log_info = [&](const std::string & msg) {
        if (logger) logger(voxtral_log_level::info, msg);
    };

    const auto t_load_start = std::chrono::steady_clock::now();
    log_info("loading model from " + path);

    ggml_context * ctx_meta = nullptr;
    gguf_init_params gguf_params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &ctx_meta,
    };

    gguf_context * gguf_ctx = gguf_init_from_file(path.c_str(), gguf_params);
    if (!gguf_ctx) {
        fprintf(stderr, "voxtral: failed to open GGUF file: %s\n", path.c_str());
        return nullptr;
    }

    voxtral_model * model = new voxtral_model();
    model->gguf_ctx  = gguf_ctx;
    model->ctx_gguf  = ctx_meta;

    // Allocate a backend buffer for all the weights
    ggml_backend_t weights_backend = nullptr;
    voxtral_gpu_backend resolved_gpu = voxtral_gpu_backend::none;

    auto try_cuda = [&]() -> bool {
#ifdef GGML_USE_CUDA
        weights_backend = ggml_backend_cuda_init(0);
        if (weights_backend) { resolved_gpu = voxtral_gpu_backend::cuda; return true; }
        log_info("CUDA backend init failed");
#endif
        return false;
    };

    auto try_metal = [&]() -> bool {
#ifdef GGML_USE_METAL
        weights_backend = ggml_backend_metal_init();
        if (weights_backend) { resolved_gpu = voxtral_gpu_backend::metal; return true; }
        log_info("Metal backend init failed");
#endif
        return false;
    };

    auto try_vulkan = [&]() -> bool {
#ifdef GGML_USE_VULKAN
        weights_backend = ggml_backend_vk_init(0);
        if (weights_backend) { resolved_gpu = voxtral_gpu_backend::vulkan; return true; }
        log_info("Vulkan backend init failed");
#endif
        return false;
    };

    switch (gpu) {
        case voxtral_gpu_backend::cuda:
            if (!try_cuda()) {
                log_info("CUDA not available in this build, falling back to CPU");
            }
            break;
        case voxtral_gpu_backend::metal:
            if (!try_metal()) {
                log_info("Metal not available in this build, falling back to CPU");
            }
            break;
        case voxtral_gpu_backend::vulkan:
            if (!try_vulkan()) {
                log_info("Vulkan not available in this build, falling back to CPU");
            }
            break;
        case voxtral_gpu_backend::auto_detect:
            if (!try_cuda() && !try_metal() && !try_vulkan()) {
                log_info("no GPU backend available, using CPU");
            }
            break;
        case voxtral_gpu_backend::none:
        default:
            break;
    }

    if (!weights_backend) {
        weights_backend = ggml_backend_cpu_init();
    }

    model->backend_weights = weights_backend;
    model->weights_on_gpu = (resolved_gpu != voxtral_gpu_backend::none);
    model->gpu_type = resolved_gpu;
    model->buf_weights = ggml_backend_alloc_ctx_tensors(ctx_meta, weights_backend);

    if (!model->buf_weights) {
        fprintf(stderr, "voxtral: failed to allocate weight buffer\n");
        if (model->backend_weights) {
            ggml_backend_free(model->backend_weights);
            model->backend_weights = nullptr;
        }
        gguf_free(gguf_ctx);
        ggml_free(ctx_meta);
        delete model;
        return nullptr;
    }

    // Load tensor data from file into buffer
    {
        FILE * fp = fopen(path.c_str(), "rb");
        if (!fp) {
            fprintf(stderr, "voxtral: failed to open file for reading weights\n");
            voxtral_model_free(model);
            return nullptr;
        }

        const int n_tensors = gguf_get_n_tensors(gguf_ctx);
        for (int i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(gguf_ctx, i);
            ggml_tensor * t = ggml_get_tensor(ctx_meta, name);
            if (!t) continue;

            const size_t offset = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, i);
            const size_t nbytes = ggml_nbytes(t);

            std::vector<uint8_t> tmp(nbytes);
            fseek(fp, (long)offset, SEEK_SET);
            if (fread(tmp.data(), 1, nbytes, fp) != nbytes) {
                fprintf(stderr, "voxtral: failed to read tensor '%s'\n", name);
                fclose(fp);
                voxtral_model_free(model);
                return nullptr;
            }
            ggml_backend_tensor_set(t, tmp.data(), 0, nbytes);
        }
        fclose(fp);
    }

    // Map weight tensors
    model->enc_conv0_weight = get_tensor(ctx_meta, "enc.conv0.weight");
    model->enc_conv0_bias   = get_tensor(ctx_meta, "enc.conv0.bias");
    model->enc_conv1_weight = get_tensor(ctx_meta, "enc.conv1.weight");
    model->enc_conv1_bias   = get_tensor(ctx_meta, "enc.conv1.bias");
    model->enc_norm_weight  = get_tensor(ctx_meta, "enc.norm.weight");

    model->enc_layers.resize(VOXTRAL_ENC_LAYERS);
    for (int32_t i = 0; i < VOXTRAL_ENC_LAYERS; i++) {
        char nm[256];
        auto & L = model->enc_layers[i];
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_norm.weight",i); L.attn_norm_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_q.weight",i);    L.attn_q_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_q.bias",i);      L.attn_q_bias   = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_k.weight",i);    L.attn_k_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_v.weight",i);    L.attn_v_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_v.bias",i);      L.attn_v_bias   = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_o.weight",i);    L.attn_o_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_o.bias",i);      L.attn_o_bias   = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.ffn_norm.weight",i);  L.ffn_norm_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.ffn_w1.weight",i);    L.ffn_w1_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.ffn_w2.weight",i);    L.ffn_w2_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.ffn_w2.bias",i);      L.ffn_w2_bias   = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.ffn_w3.weight",i);    L.ffn_w3_weight = get_tensor(ctx_meta,nm);
    }

    model->adapter_0_weight = get_tensor(ctx_meta, "adapter.0.weight");
    model->adapter_2_weight = get_tensor(ctx_meta, "adapter.2.weight");

    model->tok_embeddings_weight = get_tensor(ctx_meta, "tok_embeddings.weight");
    model->dec_norm_weight       = get_tensor(ctx_meta, "norm.weight");

    model->dec_layers.resize(VOXTRAL_DEC_LAYERS);
    for (int32_t i = 0; i < VOXTRAL_DEC_LAYERS; i++) {
        char nm[256];
        auto & L = model->dec_layers[i];
        snprintf(nm,sizeof(nm),"dec.blk.%d.attn_norm.weight",i); L.attn_norm_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.attn_q.weight",i);    L.attn_q_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.attn_k.weight",i);    L.attn_k_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.attn_v.weight",i);    L.attn_v_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.attn_o.weight",i);    L.attn_o_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.ffn_norm.weight",i);  L.ffn_norm_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.ffn_w1.weight",i);    L.ffn_w1_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.ffn_w2.weight",i);    L.ffn_w2_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.ffn_w3.weight",i);    L.ffn_w3_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.ada0.weight",i);      L.ada0_weight   = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.ada2.weight",i);      L.ada2_weight   = get_tensor(ctx_meta,nm);
    }

    model->mel_filters = get_tensor(ctx_meta, "audio.mel_filters");

    // Tokenizer metadata (Tekken)
    {
        const int64_t key_num_special = gguf_find_key(gguf_ctx, "voxtral.tokenizer.num_special_tokens");
        if (key_num_special >= 0) {
            model->tokenizer_num_special_tokens = gguf_get_val_i32(gguf_ctx, key_num_special);
        }

        const int64_t key_special = gguf_find_key(gguf_ctx, "voxtral.tokenizer.special_token_ranks");
        if (key_special >= 0 && gguf_get_kv_type(gguf_ctx, key_special) == GGUF_TYPE_ARRAY) {
            if (gguf_get_arr_type(gguf_ctx, key_special) == GGUF_TYPE_INT32) {
                const size_t n = gguf_get_arr_n(gguf_ctx, key_special);
                const int32_t * data = (const int32_t *) gguf_get_arr_data(gguf_ctx, key_special);
                if (data) {
                    for (size_t i = 0; i < n; ++i) {
                        model->tokenizer_special_ranks.insert(data[i]);
                    }
                }
            }
        }

        const int64_t key_vocab = gguf_find_key(gguf_ctx, "voxtral.tokenizer.vocab_token_bytes_b64");
        if (key_vocab >= 0 && gguf_get_kv_type(gguf_ctx, key_vocab) == GGUF_TYPE_ARRAY) {
            if (gguf_get_arr_type(gguf_ctx, key_vocab) == GGUF_TYPE_STRING) {
                const size_t n = gguf_get_arr_n(gguf_ctx, key_vocab);
                model->tokenizer_vocab_b64.reserve(n);
                for (size_t i = 0; i < n; ++i) {
                    const char * s = gguf_get_arr_str(gguf_ctx, key_vocab, i);
                    model->tokenizer_vocab_b64.emplace_back(s ? s : "");
                }
            }
        }
    }

    log_info("model loaded: enc_layers=" + std::to_string(VOXTRAL_ENC_LAYERS) +
             " dec_layers=" + std::to_string(VOXTRAL_DEC_LAYERS) +
             " vocab=" + std::to_string(VOXTRAL_VOCAB_SIZE));

    if (model->buf_weights) {
        const double sz_mb = (double) ggml_backend_buffer_get_size(model->buf_weights) / 1e6;
        log_info("model weights: " + std::to_string(sz_mb) + " MB");
    }
    log_info("encoder: dim=" + std::to_string(VOXTRAL_ENC_DIM) +
             " heads=" + std::to_string(VOXTRAL_ENC_HEADS) +
             " head_dim=" + std::to_string(VOXTRAL_ENC_HEAD_DIM) +
             " hidden=" + std::to_string(VOXTRAL_ENC_HIDDEN));
    log_info("decoder: dim=" + std::to_string(VOXTRAL_DEC_DIM) +
             " heads=" + std::to_string(VOXTRAL_DEC_HEADS) +
             " head_dim=" + std::to_string(VOXTRAL_DEC_HEAD_DIM) +
             " hidden=" + std::to_string(VOXTRAL_DEC_HIDDEN) +
             " kv_heads=" + std::to_string(VOXTRAL_DEC_KV_HEADS));

    {
        char buf[128];
        snprintf(buf, sizeof(buf), "model load time: %.2f ms", elapsed_ms(t_load_start));
        log_info(std::string(buf));
    }

    return model;
}

void voxtral_model_free(voxtral_model * model) {
    if (!model) return;
    if (model->buf_weights) ggml_backend_buffer_free(model->buf_weights);
    if (model->backend_weights) ggml_backend_free(model->backend_weights);
    if (model->ctx_gguf)    ggml_free(model->ctx_gguf);
    if (model->gguf_ctx)    gguf_free(model->gguf_ctx);
    delete model;
}

// ============================================================================
// Context initialization
// ============================================================================

voxtral_context * voxtral_init_from_model(
    voxtral_model              * model,
    const voxtral_context_params & params)
{
    voxtral_context * ctx = new voxtral_context();
    ctx->model     = model;
    ctx->log_level = params.log_level;
    ctx->logger    = params.logger;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    // Select GPU backend — inherit from model if params say none
    voxtral_gpu_backend gpu = params.gpu;
    if (gpu == voxtral_gpu_backend::none && model && model->weights_on_gpu) {
        gpu = model->gpu_type;
    }
    ctx->gpu_type = voxtral_gpu_backend::none;

    auto try_cuda_ctx = [&]() -> bool {
#ifdef GGML_USE_CUDA
        ctx->backend = ggml_backend_cuda_init(0);
        if (ctx->backend) { ctx->gpu_type = voxtral_gpu_backend::cuda; return true; }
        LOG_WARN(ctx, "CUDA backend init failed");
#endif
        return false;
    };
    auto try_metal_ctx = [&]() -> bool {
#ifdef GGML_USE_METAL
        ctx->backend = ggml_backend_metal_init();
        if (ctx->backend) { ctx->gpu_type = voxtral_gpu_backend::metal; return true; }
        LOG_WARN(ctx, "Metal backend init failed");
#endif
        return false;
    };
    auto try_vulkan_ctx = [&]() -> bool {
#ifdef GGML_USE_VULKAN
        ctx->backend = ggml_backend_vk_init(0);
        if (ctx->backend) { ctx->gpu_type = voxtral_gpu_backend::vulkan; return true; }
        LOG_WARN(ctx, "Vulkan backend init failed");
#endif
        return false;
    };

    switch (gpu) {
        case voxtral_gpu_backend::cuda:    try_cuda_ctx();   break;
        case voxtral_gpu_backend::metal:   try_metal_ctx();  break;
        case voxtral_gpu_backend::vulkan:  try_vulkan_ctx(); break;
        case voxtral_gpu_backend::auto_detect:
            if (!try_cuda_ctx() && !try_metal_ctx() && !try_vulkan_ctx()) {
                LOG_INFO(ctx, "no GPU backend available, using CPU");
            }
            break;
        case voxtral_gpu_backend::none:
        default:
            break;
    }

    bool has_gpu = (ctx->gpu_type != voxtral_gpu_backend::none);

    if (!ctx->backend) {
        ctx->backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
        LOG_INFO(ctx, "backend: CPU with %d threads", ctx->n_threads);
    } else {
        ctx->backend_cpu = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
        const char * gpu_name = "GPU";
        if (ctx->gpu_type == voxtral_gpu_backend::cuda)   gpu_name = "CUDA";
        if (ctx->gpu_type == voxtral_gpu_backend::metal)  gpu_name = "METAL";
        if (ctx->gpu_type == voxtral_gpu_backend::vulkan) gpu_name = "VULKAN";
        LOG_INFO(ctx, "backend: %s (CPU fallback %d threads)", gpu_name, ctx->n_threads);
    }

    // Try to init BLAS backend for accelerated CPU matmuls
#ifdef GGML_USE_BLAS
    ctx->blas_backend = ggml_backend_blas_init();
    if (ctx->blas_backend) {
        ggml_backend_blas_set_n_threads(ctx->blas_backend, ctx->n_threads);
        LOG_INFO(ctx, "BLAS backend enabled with %d threads", ctx->n_threads);
    }
#endif

    // Allocate persistent tensors: encoder chunk output, decoder logits, KV cache
    {
        constexpr size_t n_tensors = 4;
        ggml_init_params p = {
            /*.mem_size  =*/ ggml_tensor_overhead() * n_tensors,
            /*.mem_buffer=*/ nullptr,
            /*.no_alloc  =*/ true,
        };
        ctx->ctx_persistent = ggml_init(p);

        // encoder_chunk_output: [enc_dim, MAX_ENC_CHUNK] (reused per chunk)
        ctx->encoder_chunk_output = ggml_new_tensor_2d(ctx->ctx_persistent, GGML_TYPE_F32,
            VOXTRAL_ENC_DIM, VOXTRAL_MAX_ENC_CHUNK);
        ggml_set_name(ctx->encoder_chunk_output, "encoder_chunk_output");

        // decoder_logits: [vocab_size]
        ctx->decoder_logits = ggml_new_tensor_1d(ctx->ctx_persistent, GGML_TYPE_F32,
            VOXTRAL_VOCAB_SIZE);
        ggml_set_name(ctx->decoder_logits, "decoder_logits");

        // KV cache: [kv_dim, dec_window, dec_layers]
        const int32_t kv_dim = VOXTRAL_DEC_KV_HEADS * VOXTRAL_DEC_HEAD_DIM;  // 1024
        ctx->kv_self_k = ggml_new_tensor_3d(ctx->ctx_persistent, GGML_TYPE_F32,
            kv_dim, VOXTRAL_DEC_WINDOW, VOXTRAL_DEC_LAYERS);
        ggml_set_name(ctx->kv_self_k, "kv_self_k");

        ctx->kv_self_v = ggml_new_tensor_3d(ctx->ctx_persistent, GGML_TYPE_F32,
            kv_dim, VOXTRAL_DEC_WINDOW, VOXTRAL_DEC_LAYERS);
        ggml_set_name(ctx->kv_self_v, "kv_self_v");

        ctx->buf_persistent = ggml_backend_alloc_ctx_tensors(ctx->ctx_persistent, ctx->backend);
        if (!ctx->buf_persistent) {
            fprintf(stderr, "voxtral: failed to allocate persistent buffer\n");
            voxtral_free(ctx);
            return nullptr;
        }

        // Zero persistent buffer (KV cache etc.)
        ggml_backend_buffer_clear(ctx->buf_persistent, 0);
    }

    {
        const double chunk_mb = (double) ggml_nbytes(ctx->encoder_chunk_output) / 1e6;
        const double kv_mb  = (double) (ggml_nbytes(ctx->kv_self_k) + ggml_nbytes(ctx->kv_self_v)) / 1e6;
        LOG_INFO(ctx, "buffers: encoder_chunk=%.2f MB kv_cache=%.2f MB",
            chunk_mb, kv_mb);
    }

    // Schedulers — ggml requires the last backend to be CPU.
    // With GPU:    [GPU, BLAS?, CPU]
    // Without GPU: [BLAS?, CPU]  (ctx->backend IS the CPU backend)
    ggml_backend_t backends[4];
    int n_backends = 0;
    if (has_gpu) {
        backends[n_backends++] = ctx->backend;           // GPU first
    }
    if (ctx->blas_backend) {
        backends[n_backends++] = ctx->blas_backend;      // BLAS before CPU
    }
    // CPU must be last
    ggml_backend_t cpu_be = has_gpu ? ctx->backend_cpu : ctx->backend;
    backends[n_backends++] = cpu_be;
    const bool op_offload = has_gpu;

    ctx->sched_encoder  = ggml_backend_sched_new(backends, nullptr, n_backends, GGML_DEFAULT_GRAPH_SIZE, false, op_offload);
    ctx->sched_adapter  = ggml_backend_sched_new(backends, nullptr, n_backends, GGML_DEFAULT_GRAPH_SIZE, false, op_offload);
    ctx->sched_dec_pre  = ggml_backend_sched_new(backends, nullptr, n_backends, GGML_DEFAULT_GRAPH_SIZE, false, op_offload);
    ctx->sched_dec_step = ggml_backend_sched_new(backends, nullptr, n_backends, GGML_DEFAULT_GRAPH_SIZE, false, op_offload);

    // Hann window
    ctx->hann_window.resize(VOXTRAL_WINDOW_SIZE);
    for (int32_t i = 0; i < VOXTRAL_WINDOW_SIZE; i++) {
        // Match torch.hann_window(W, periodic=True)
        ctx->hann_window[i] = 0.5f * (1.0f - cosf(2.0f * VOXTRAL_PI * (float)i / (float)(VOXTRAL_WINDOW_SIZE)));
    }

    // Mel filters (compute on CPU if not available from model, else load from GGUF)
    if (model->mel_filters) {
        constexpr int32_t n = VOXTRAL_N_FREQ * VOXTRAL_NUM_MEL_BINS;
        ctx->mel_filters_cpu.resize(n);
        ggml_backend_tensor_get(model->mel_filters, ctx->mel_filters_cpu.data(), 0, n * sizeof(float));
    } else {
        compute_mel_filters_slaney(ctx->mel_filters_cpu);
    }

    // Time embedding for t = N_DELAY_TOKENS
    compute_time_embedding(ctx->time_emb_cpu, (float)VOXTRAL_N_DELAY_TOKENS, VOXTRAL_DEC_DIM);

    LOG_INFO(ctx, "context initialized");
    return ctx;
}

void voxtral_free(voxtral_context * ctx) {
    if (!ctx) return;
    if (ctx->sched_encoder)  ggml_backend_sched_free(ctx->sched_encoder);
    if (ctx->sched_adapter)  ggml_backend_sched_free(ctx->sched_adapter);
    if (ctx->sched_dec_pre)  ggml_backend_sched_free(ctx->sched_dec_pre);
    if (ctx->sched_dec_step) ggml_backend_sched_free(ctx->sched_dec_step);
    if (ctx->buf_enc_full)   ggml_backend_buffer_free(ctx->buf_enc_full);
    if (ctx->ctx_enc_full)   ggml_free(ctx->ctx_enc_full);
    if (ctx->buf_dec_mem)    ggml_backend_buffer_free(ctx->buf_dec_mem);
    if (ctx->ctx_dec_mem)    ggml_free(ctx->ctx_dec_mem);
    if (ctx->buf_persistent) ggml_backend_buffer_free(ctx->buf_persistent);
    if (ctx->ctx_persistent) ggml_free(ctx->ctx_persistent);
    if (ctx->blas_backend)   ggml_backend_free(ctx->blas_backend);
    if (ctx->backend_cpu)    ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend)        ggml_backend_free(ctx->backend);
    delete ctx;
}

// ============================================================================
// KV cache helpers
// ============================================================================

static void clear_kv_cache(voxtral_context * ctx) {
    if (!ctx || !ctx->kv_self_k || !ctx->kv_self_v) {
        return;
    }
    const bool k_host = ctx->kv_self_k->buffer != nullptr &&
                        ggml_backend_buffer_is_host(ctx->kv_self_k->buffer);
    const bool v_host = ctx->kv_self_v->buffer != nullptr &&
                        ggml_backend_buffer_is_host(ctx->kv_self_v->buffer);

    // On non-host buffers (CUDA/Vulkan/Metal), raw memset on ggml_get_data()
    // may access device pointers and segfault. Clear through backend ops instead.
    if (!k_host || !v_host) {
        if (ctx->buf_persistent) {
            ggml_backend_buffer_clear(ctx->buf_persistent, 0);
        } else {
            static thread_local std::vector<uint8_t> zeros(1 << 20, 0);

            for (ggml_tensor * t : {ctx->kv_self_k, ctx->kv_self_v}) {
                const size_t nbytes = ggml_nbytes(t);
                for (size_t off = 0; off < nbytes; off += zeros.size()) {
                    const size_t n = std::min(zeros.size(), nbytes - off);
                    ggml_backend_tensor_set(t, zeros.data(), off, n);
                }
            }
        }
        ctx->kv_used = 0;
        return;
    }

    void * k_data = ggml_get_data(ctx->kv_self_k);
    void * v_data = ggml_get_data(ctx->kv_self_v);
    if (k_data) {
        memset(k_data, 0, ggml_nbytes(ctx->kv_self_k));
    }
    if (v_data) {
        memset(v_data, 0, ggml_nbytes(ctx->kv_self_v));
    }
    ctx->kv_used = 0;
}

static void kv_cache_shift_left(voxtral_context * ctx, int32_t shift) {
    if (!ctx || shift <= 0 || !ctx->kv_self_k || !ctx->kv_self_v) {
        return;
    }
    const bool k_host = ctx->kv_self_k->buffer != nullptr &&
                        ggml_backend_buffer_is_host(ctx->kv_self_k->buffer);
    const bool v_host = ctx->kv_self_v->buffer != nullptr &&
                        ggml_backend_buffer_is_host(ctx->kv_self_v->buffer);
    if (!k_host || !v_host) {
        clear_kv_cache(ctx);
        return;
    }

    const int32_t window = VOXTRAL_DEC_WINDOW;
    if (shift >= window) {
        clear_kv_cache(ctx);
        return;
    }

    uint8_t * k_data = (uint8_t *) ggml_get_data(ctx->kv_self_k);
    uint8_t * v_data = (uint8_t *) ggml_get_data(ctx->kv_self_v);
    if (!k_data || !v_data) {
        return;
    }

    const size_t row_bytes = ctx->kv_self_k->nb[1];
    const size_t layer_stride = ctx->kv_self_k->nb[2];

    for (int32_t l = 0; l < VOXTRAL_DEC_LAYERS; ++l) {
        uint8_t * k_base = k_data + (size_t) l * layer_stride;
        uint8_t * v_base = v_data + (size_t) l * layer_stride;

        memmove(k_base, k_base + (size_t) shift * row_bytes, (size_t) (window - shift) * row_bytes);
        memmove(v_base, v_base + (size_t) shift * row_bytes, (size_t) (window - shift) * row_bytes);

        memset(k_base + (size_t) (window - shift) * row_bytes, 0, (size_t) shift * row_bytes);
        memset(v_base + (size_t) (window - shift) * row_bytes, 0, (size_t) shift * row_bytes);
    }
}

// ============================================================================
// Graph Building: Encoder
// ============================================================================


struct causal_conv1d_dims {
    int32_t pad_left = 0;
    int32_t pad_right = 0;
    int32_t padded_len = 0;
    int32_t out_len = 0;
};

causal_conv1d_dims compute_causal_conv1d_dims(int32_t in_len, int32_t kernel_size, int32_t stride) {
    causal_conv1d_dims out{};
    if (in_len <= 0 || kernel_size <= 0 || stride <= 0) {
        return out;
    }

    const int32_t padding_total = kernel_size - stride;
    const float n_frames = (static_cast<float>(in_len - kernel_size + padding_total) / static_cast<float>(stride)) + 1.0f;
    const int32_t target_length =
        (static_cast<int32_t>(std::ceil(n_frames)) - 1) * stride + (kernel_size - padding_total);
    const int32_t extra_padding = target_length - in_len;

    out.pad_left = padding_total;
    out.pad_right = std::max<int32_t>(0, extra_padding);
    out.padded_len = in_len + out.pad_left + out.pad_right;
    out.out_len = (out.padded_len - kernel_size) / stride + 1;
    return out;
}

// Compute the number of encoder tokens from mel frames (accounting for conv and truncation)
static int32_t mel_frames_to_enc_tokens(int32_t n_frames) {
    auto d0 = compute_causal_conv1d_dims(n_frames, 3, 1);  // conv0
    auto d1 = compute_causal_conv1d_dims(d0.out_len, 3, 2); // conv1 (stride 2)
    int32_t trunc = d1.out_len % VOXTRAL_DOWNSAMPLE_FACTOR;
    return d1.out_len - trunc;
}

// Pre-compute total encoder tokens for a given mel frame count (for buffer allocation)
static int32_t compute_total_enc_tokens(int32_t total_mel_frames) {
    const int32_t mel_stride = VOXTRAL_ENC_CHUNK_MEL - VOXTRAL_ENC_CHUNK_OVERLAP * 2;
    int32_t total = 0;
    int32_t mel_offset = 0;
    bool first = true;

    while (mel_offset < total_mel_frames) {
        int32_t chunk_mel = std::min(VOXTRAL_ENC_CHUNK_MEL, total_mel_frames - mel_offset);
        int32_t chunk_tokens = mel_frames_to_enc_tokens(chunk_mel);
        int32_t skip = first ? 0 : VOXTRAL_ENC_CHUNK_OVERLAP;
        int32_t stride = chunk_tokens - skip;
        if (stride <= 0) break;
        total += stride;
        mel_offset += mel_stride;
        first = false;
    }
    return total;
}

// Allocate per-utterance encoder output buffer on device
static bool alloc_encoder_output(voxtral_context * ctx, int32_t n_tokens) {
    // Free previous allocation
    if (ctx->buf_enc_full) { ggml_backend_buffer_free(ctx->buf_enc_full); ctx->buf_enc_full = nullptr; }
    if (ctx->ctx_enc_full) { ggml_free(ctx->ctx_enc_full); ctx->ctx_enc_full = nullptr; }
    ctx->encoder_output = nullptr;

    ggml_init_params p = {
        /*.mem_size  =*/ ggml_tensor_overhead(),
        /*.mem_buffer=*/ nullptr,
        /*.no_alloc  =*/ true,
    };
    ctx->ctx_enc_full = ggml_init(p);
    ctx->encoder_output = ggml_new_tensor_2d(ctx->ctx_enc_full, GGML_TYPE_F32,
        VOXTRAL_ENC_DIM, n_tokens);
    ggml_set_name(ctx->encoder_output, "encoder_output");
    ctx->buf_enc_full = ggml_backend_alloc_ctx_tensors(ctx->ctx_enc_full, ctx->backend);
    if (!ctx->buf_enc_full) return false;

    ctx->total_enc_tokens = n_tokens;
    return true;
}

// Allocate per-utterance decoder memory buffer on device
static bool alloc_decoder_memory(voxtral_context * ctx, int32_t dec_seq) {
    if (ctx->buf_dec_mem) { ggml_backend_buffer_free(ctx->buf_dec_mem); ctx->buf_dec_mem = nullptr; }
    if (ctx->ctx_dec_mem) { ggml_free(ctx->ctx_dec_mem); ctx->ctx_dec_mem = nullptr; }
    ctx->decoder_memory = nullptr;

    ggml_init_params p = {
        /*.mem_size  =*/ ggml_tensor_overhead(),
        /*.mem_buffer=*/ nullptr,
        /*.no_alloc  =*/ true,
    };
    ctx->ctx_dec_mem = ggml_init(p);
    ctx->decoder_memory = ggml_new_tensor_2d(ctx->ctx_dec_mem, GGML_TYPE_F32,
        VOXTRAL_DEC_DIM, dec_seq);
    ggml_set_name(ctx->decoder_memory, "decoder_memory");
    ctx->buf_dec_mem = ggml_backend_alloc_ctx_tensors(ctx->ctx_dec_mem, ctx->backend);
    if (!ctx->buf_dec_mem) return false;

    ctx->dec_seq_len = dec_seq;
    return true;
}

ggml_tensor * causal_conv1d_graph(
    ggml_context * ctx0,
    ggml_tensor * x,
    int32_t in_len,
    ggml_tensor * weight,
    ggml_tensor * bias,
    int32_t out_channels,
    int32_t kernel_size,
    int32_t stride,
    int32_t & out_len) {
    out_len = 0;
    if (ctx0 == nullptr || x == nullptr || weight == nullptr || kernel_size <= 0 || stride <= 0) {
        return nullptr;
    }
    if (in_len <= 0 || out_channels <= 0) {
        return nullptr;
    }

    const auto dims = compute_causal_conv1d_dims(in_len, kernel_size, stride);
    if (dims.out_len <= 0) {
        return nullptr;
    }

    ggml_tensor * x_pad = ggml_pad_ext(ctx0, x, dims.pad_left, dims.pad_right, 0, 0, 0, 0, 0, 0);
    if (x_pad == nullptr) {
        return nullptr;
    }

    ggml_tensor * y = ggml_conv_1d(ctx0, weight, x_pad, stride, 0, 1);
    if (y == nullptr) {
        return nullptr;
    }

    if (bias != nullptr) {
        y = ggml_add(ctx0, y, ggml_reshape_3d(ctx0, bias, 1, out_channels, 1));
    }

    out_len = dims.out_len;
    return y;
}


void print_tensor_info(struct ggml_tensor * tensor) {
    printf("Tensor name: %s\n", tensor->name);
    printf("Tensor type: %s\n", ggml_type_name(tensor->type));
    printf("Number of dimensions: %d\n", ggml_n_dims(tensor));
    printf("Total elements: %" PRId64 "\n", ggml_nelements(tensor));
    printf("Shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n",
           tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
}

static void log_tensor_info(voxtral_context * ctx, const char * tag, struct ggml_tensor * t) {
    if (t == nullptr) {
        LOG_DBG(ctx, "%s: <null>", tag);
        return;
    }
    LOG_DBG(ctx, "%s: type=%s ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] nb=[%zu,%zu,%zu,%zu] n_dims=%d nbytes=%zu",
        tag,
        ggml_type_name(t->type),
        t->ne[0], t->ne[1], t->ne[2], t->ne[3],
        (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3],
        ggml_n_dims(t),
        (size_t) ggml_nbytes(t));
}

static void log_graph_info(voxtral_context * ctx, const char * name, struct ggml_cgraph * gf) {
    if (gf == nullptr) {
        return;
    }
    const int size  = ggml_graph_size(gf);
    const int nodes = ggml_graph_n_nodes(gf);
    LOG_INFO(ctx, "%s graph: size=%d nodes=%d", name, size, nodes);
}

// Build encoder graph that writes output into ctx->encoder_chunk_output
static ggml_cgraph * build_encoder_graph(
    voxtral_context * ctx,
    ggml_context * gctx,
    const float * mel_data,   // [n_mel, n_frames] on CPU
    int32_t n_frames,
    int32_t * out_seq_len)    // output: encoder tokens produced by this chunk
{
    LOG_DBG(ctx, "Building encoder graph");
    voxtral_model * model = ctx->model;

    ggml_cgraph * gf = ggml_new_graph_custom(gctx, GGML_DEFAULT_GRAPH_SIZE * 4, false);

    // ggml_conv_1d expects input as [length, in_channels, batch]
    // mel_data is [n_mel, n_frames] on CPU; we transpose on upload.
    ggml_tensor * mel_input = ggml_new_tensor_3d(
        gctx, GGML_TYPE_F32, n_frames, VOXTRAL_NUM_MEL_BINS, 1);
    ggml_set_name(mel_input, "mel_input");

    // We need to set data after sched_alloc, mark as input
    ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, mel_input, ctx->backend);

    // Conv stem: mel is [n_frames, n_mel, 1], weights are [k, in_ch, out_ch]
    log_tensor_info(ctx, "enc.conv0.weight", model->enc_conv0_weight);
    log_tensor_info(ctx, "enc.conv1.weight", model->enc_conv1_weight);
    log_tensor_info(ctx, "mel_input", mel_input);

    int32_t conv0_len = 0;
    ggml_tensor * conv0_out = causal_conv1d_graph(
        gctx, mel_input, n_frames,
        model->enc_conv0_weight, model->enc_conv0_bias,
        VOXTRAL_ENC_DIM, 3, 1, conv0_len);
    if (conv0_out == nullptr) {
        LOG_ERR(ctx, "conv0_out is null");
        return gf;
    }
    log_tensor_info(ctx, "conv0_out(pre_act)", conv0_out);
    conv0_out = ggml_gelu_erf(gctx, conv0_out);

    int32_t conv_out_len = 0;
    ggml_tensor * conv1_out = causal_conv1d_graph(
        gctx, conv0_out, conv0_len,
        model->enc_conv1_weight, model->enc_conv1_bias,
        VOXTRAL_ENC_DIM, 3, 2, conv_out_len);
    if (conv1_out == nullptr) {
        LOG_ERR(ctx, "conv1_out is null");
        return gf;
    }
    log_tensor_info(ctx, "conv1_out(pre_act)", conv1_out);
    conv1_out = ggml_gelu_erf(gctx, conv1_out);
    log_tensor_info(ctx, "conv1_out", conv1_out);

    // Transpose for transformer: [enc_dim, seq] -> [enc_dim, seq] (already correct for ggml)
    // In ggml, tensor is [ne0=enc_dim, ne1=seq], which means each "row" (token) has enc_dim elements
    // This is what we need for mul_mat: ggml_mul_mat(weight[out,in], x[in,seq]) -> [out,seq]

    // Left-truncate to multiple of downsample_factor (matching Python)
    const int32_t trunc = conv_out_len % VOXTRAL_DOWNSAMPLE_FACTOR;
    ggml_tensor * x_len_first = conv1_out;
    int32_t seq_len = conv_out_len;
    if (trunc > 0) {
        // Skip first 'trunc' frames along length dimension (ne0)
        x_len_first = ggml_view_3d(gctx, conv1_out,
            conv_out_len - trunc, VOXTRAL_ENC_DIM, 1,
            conv1_out->nb[1], conv1_out->nb[2],
            (size_t) trunc * conv1_out->nb[0]); // [len, enc_dim, 1]
        seq_len = conv_out_len - trunc;
    }
    LOG_DBG(ctx, "encoder conv: in_frames=%d conv0_len=%d conv1_len=%d trunc=%d seq_len=%d",
        n_frames, conv0_len, conv_out_len, trunc, seq_len);

    // Transpose to [enc_dim, seq_len] for transformer blocks
    ggml_tensor * x = ggml_permute(gctx, x_len_first, 1, 0, 2, 3); // [enc_dim, seq_len, 1]
    x = ggml_cont(gctx, x);
    x = ggml_reshape_2d(gctx, x, VOXTRAL_ENC_DIM, seq_len);
    log_tensor_info(ctx, "encoder_x", x);

    // Position tensor for RoPE: [seq_len] int32
    ggml_tensor * enc_positions = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, seq_len);
    ggml_set_name(enc_positions, "enc_positions");
    ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, enc_positions, ctx->backend);

    // Encoder attention mask (sliding causal window)
    ggml_tensor * enc_attn_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, seq_len, seq_len);
    ggml_set_name(enc_attn_mask, "enc_attn_mask");
    ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, enc_attn_mask, ctx->backend);

    // Cast mask to F16 for flash attention
    ggml_tensor * enc_attn_mask_f16 = ggml_cast(gctx, enc_attn_mask, GGML_TYPE_F16);

    // Transformer layers
    for (int32_t i = 0; i < VOXTRAL_ENC_LAYERS; i++) {
        auto & L = model->enc_layers[i];

        // Pre-attention RMS norm
        ggml_tensor * residual = x; // [enc_dim, seq_len]
        ggml_tensor * x_norm = ggml_rms_norm(gctx, x, VOXTRAL_ENC_NORM_EPS); // [enc_dim, seq_len]
        x_norm = ggml_mul(gctx, x_norm, L.attn_norm_weight); // [enc_dim, seq_len]

        // Q, K, V projections
        ggml_tensor * q = ggml_mul_mat(gctx, L.attn_q_weight, x_norm); // [enc_heads*head_dim, seq_len]
        q = ggml_add(gctx, q, L.attn_q_bias); // [enc_heads*head_dim, seq_len]

        ggml_tensor * k = ggml_mul_mat(gctx, L.attn_k_weight, x_norm); // [enc_kv_heads*head_dim, seq_len]
        // k has no bias in encoder

        ggml_tensor * v = ggml_mul_mat(gctx, L.attn_v_weight, x_norm); // [enc_kv_heads*head_dim, seq_len]
        v = ggml_add(gctx, v, L.attn_v_bias); // [enc_kv_heads*head_dim, seq_len]

        // Reshape for RoPE: [head_dim, n_heads, seq_len]
        q = ggml_reshape_3d(gctx, q, VOXTRAL_ENC_HEAD_DIM, VOXTRAL_ENC_HEADS, seq_len);
        k = ggml_reshape_3d(gctx, k, VOXTRAL_ENC_HEAD_DIM, VOXTRAL_ENC_KV_HEADS, seq_len);

        // Apply RoPE (interleaved, mode=0)
        // ggml_rope_ext expects: a=[head_dim, n_heads, seq], b=[seq] positions
        q = ggml_rope_ext(gctx, q, enc_positions, nullptr,
            VOXTRAL_ENC_HEAD_DIM, 0, 0,
            VOXTRAL_ENC_ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f); // [head_dim, n_heads, seq_len]
        k = ggml_rope_ext(gctx, k, enc_positions, nullptr,
            VOXTRAL_ENC_HEAD_DIM, 0, 0,
            VOXTRAL_ENC_ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f); // [head_dim, n_kv_heads, seq_len]

        // Flash attention
        // Q: [head_dim, n_heads, seq_len] -> [head_dim, seq_len, n_heads]
        q = ggml_permute(gctx, q, 0, 2, 1, 3); // [head_dim, seq_len, n_heads]
        k = ggml_permute(gctx, k, 0, 2, 1, 3); // [head_dim, seq_len, n_kv_heads]

        // V: [enc_kv_heads*head_dim, seq_len] -> [head_dim, n_kv_heads, seq_len] -> [head_dim, seq_len, n_kv_heads]
        v = ggml_reshape_3d(gctx, v, VOXTRAL_ENC_HEAD_DIM, VOXTRAL_ENC_KV_HEADS, seq_len);
        v = ggml_permute(gctx, v, 0, 2, 1, 3); // [head_dim, seq_len, n_kv_heads]

        const float scale = 1.0f / sqrtf((float)VOXTRAL_ENC_HEAD_DIM);

        // ggml_flash_attn_ext fuses Q@K^T, scale, mask, softmax, @V
        ggml_tensor * attn_out = ggml_flash_attn_ext(gctx, q, k, v, enc_attn_mask_f16, scale, 0.0f, 0.0f);
        // Output: [head_dim, n_heads, seq_len] (already permuted by flash_attn_ext)
        attn_out = ggml_cont(gctx, attn_out);
        attn_out = ggml_reshape_2d(gctx, attn_out, VOXTRAL_ENC_HEADS * VOXTRAL_ENC_HEAD_DIM, seq_len); // [n_heads*head_dim, seq_len]

        // Output projection + residual
        ggml_tensor * attn_proj = ggml_mul_mat(gctx, L.attn_o_weight, attn_out); // [enc_dim, seq_len]
        attn_proj = ggml_add(gctx, attn_proj, L.attn_o_bias); // [enc_dim, seq_len]
        x = ggml_add(gctx, residual, attn_proj); // [enc_dim, seq_len]

        // FFN
        residual = x; // [enc_dim, seq_len]
        x_norm = ggml_rms_norm(gctx, x, VOXTRAL_ENC_NORM_EPS); // [enc_dim, seq_len]
        x_norm = ggml_mul(gctx, x_norm, L.ffn_norm_weight); // [enc_dim, seq_len]

        // SwiGLU: silu(w1(x)) * w3(x), then w2
        ggml_tensor * gate = ggml_mul_mat(gctx, L.ffn_w1_weight, x_norm); // [enc_hidden, seq_len]
        gate = ggml_silu(gctx, gate); // [enc_hidden, seq_len]
        ggml_tensor * up = ggml_mul_mat(gctx, L.ffn_w3_weight, x_norm); // [enc_hidden, seq_len]
        ggml_tensor * ffn_out = ggml_mul(gctx, gate, up); // [enc_hidden, seq_len]
        ffn_out = ggml_mul_mat(gctx, L.ffn_w2_weight, ffn_out); // [enc_dim, seq_len]
        ffn_out = ggml_add(gctx, ffn_out, L.ffn_w2_bias); // [enc_dim, seq_len]

        x = ggml_add(gctx, residual, ffn_out); // [enc_dim, seq_len]
    }

    // Final norm
    x = ggml_rms_norm(gctx, x, VOXTRAL_ENC_NORM_EPS); // [enc_dim, seq_len]
    x = ggml_mul(gctx, x, model->enc_norm_weight); // [enc_dim, seq_len]

    // Copy result to encoder_chunk_output (per-chunk buffer, reused each chunk)
    ggml_tensor * enc_out_view = ggml_view_2d(gctx, ctx->encoder_chunk_output,
        VOXTRAL_ENC_DIM, seq_len,
        ctx->encoder_chunk_output->nb[1], 0); // [enc_dim, seq_len]
    ggml_tensor * cpy = ggml_cpy(gctx, x, enc_out_view);
    ggml_build_forward_expand(gf, cpy);

    if (out_seq_len) *out_seq_len = seq_len;

    return gf;
}

// ============================================================================
// Graph Building: Adapter
// ============================================================================

static ggml_cgraph * build_adapter_graph(
    voxtral_context * ctx,
    ggml_context * gctx)
{
    voxtral_model * model = ctx->model;
    const int32_t enc_seq = ctx->enc_seq_used;
    const int32_t dec_seq = enc_seq / VOXTRAL_DOWNSAMPLE_FACTOR;

    ggml_cgraph * gf = ggml_new_graph(gctx);

    // Read encoder_output: [enc_dim, enc_seq]
    ggml_tensor * enc_out = ggml_view_2d(gctx, ctx->encoder_output,
        VOXTRAL_ENC_DIM, enc_seq,
        ctx->encoder_output->nb[1], 0); // [enc_dim, enc_seq]

    // Reshape for downsample: [enc_dim, enc_seq] -> [enc_dim * 4, enc_seq/4]
    ggml_tensor * x = ggml_reshape_2d(gctx, enc_out,
        VOXTRAL_ENC_DIM * VOXTRAL_DOWNSAMPLE_FACTOR, dec_seq); // [enc_dim*4, dec_seq]

    // Linear 0: [enc_dim*4, dec_seq] -> [dec_dim, dec_seq]
    x = ggml_mul_mat(gctx, model->adapter_0_weight, x); // [dec_dim, dec_seq]
    x = ggml_gelu_erf(gctx, x); // [dec_dim, dec_seq]

    // Linear 2: [dec_dim, dec_seq] -> [dec_dim, dec_seq]
    x = ggml_mul_mat(gctx, model->adapter_2_weight, x); // [dec_dim, dec_seq]

    // Copy to persistent decoder_memory
    ggml_tensor * dec_mem_view = ggml_view_2d(gctx, ctx->decoder_memory,
        VOXTRAL_DEC_DIM, dec_seq,
        ctx->decoder_memory->nb[1], 0); // [dec_dim, dec_seq]
    ggml_tensor * cpy = ggml_cpy(gctx, x, dec_mem_view);
    ggml_build_forward_expand(gf, cpy);

    ctx->dec_seq_len = dec_seq;

    return gf;
}

// ============================================================================
// Graph Building: Decoder (common layer forward)
// ============================================================================

// Build one decoder layer. Returns updated hidden state.
// For prefill: n_tokens > 1, positions = [0..n_tokens-1]
// For step: n_tokens = 1
static ggml_tensor * build_decoder_layer(
    voxtral_context     * ctx,
    ggml_context * gctx,
    ggml_cgraph  * gf,
    ggml_tensor  * x,          // [dec_dim, n_tokens]
    ggml_tensor  * positions,  // [n_tokens] int32
    ggml_tensor  * time_emb,   // [dec_dim]
    int32_t layer_idx,
    int32_t n_tokens,
    int32_t kv_offset,                // starting position in KV cache
    ggml_tensor  * attn_mask)  // [n_kv, n_tokens] or nullptr
{
    voxtral_model * model = ctx->model;
    auto & L = model->dec_layers[layer_idx];

    const int32_t kv_dim = VOXTRAL_DEC_KV_HEADS * VOXTRAL_DEC_HEAD_DIM; // 1024

    // Pre-attention RMS norm
    ggml_tensor * residual = x; // [dec_dim, n_tokens]
    ggml_tensor * x_norm = ggml_rms_norm(gctx, x, VOXTRAL_DEC_NORM_EPS); // [dec_dim, n_tokens]
    x_norm = ggml_mul(gctx, x_norm, L.attn_norm_weight); // [dec_dim, n_tokens]

    // Q, K, V (no bias in decoder)
    ggml_tensor * q = ggml_mul_mat(gctx, L.attn_q_weight, x_norm); // [dec_heads*head_dim, n_tokens]
    ggml_tensor * k = ggml_mul_mat(gctx, L.attn_k_weight, x_norm); // [kv_dim, n_tokens]
    ggml_tensor * v = ggml_mul_mat(gctx, L.attn_v_weight, x_norm); // [kv_dim, n_tokens]

    // Reshape for RoPE: [head_dim, n_heads, n_tokens]
    q = ggml_reshape_3d(gctx, q, VOXTRAL_DEC_HEAD_DIM, VOXTRAL_DEC_HEADS, n_tokens);
    k = ggml_reshape_3d(gctx, k, VOXTRAL_DEC_HEAD_DIM, VOXTRAL_DEC_KV_HEADS, n_tokens);

    // RoPE (interleaved, mode=0)
    q = ggml_rope_ext(gctx, q, positions, nullptr,
        VOXTRAL_DEC_HEAD_DIM, 0, 0,
        VOXTRAL_DEC_ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(gctx, k, positions, nullptr,
        VOXTRAL_DEC_HEAD_DIM, 0, 0,
        VOXTRAL_DEC_ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // Flatten Q back: [head_dim, n_heads, n_tokens] -> [n_heads*head_dim, n_tokens]
    q = ggml_cont(gctx, ggml_reshape_2d(gctx, q, VOXTRAL_DEC_HEADS * VOXTRAL_DEC_HEAD_DIM, n_tokens));
    k = ggml_cont(gctx, ggml_reshape_2d(gctx, k, kv_dim, n_tokens));

    // Store K, V in KV cache at positions [kv_offset .. kv_offset+n_tokens-1]
    // KV cache layout: [kv_dim, dec_window, dec_layers]
    // Layer slice: offset = layer_idx * kv_dim * dec_window * sizeof(float)
    {
        ggml_tensor * k_cache_slice = ggml_view_2d(gctx, ctx->kv_self_k,
            kv_dim, n_tokens,
            ctx->kv_self_k->nb[1],
            layer_idx * ctx->kv_self_k->nb[2] + (size_t)kv_offset * ctx->kv_self_k->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(gctx, k, k_cache_slice));

        ggml_tensor * v_cache_slice = ggml_view_2d(gctx, ctx->kv_self_v,
            kv_dim, n_tokens,
            ctx->kv_self_v->nb[1],
            layer_idx * ctx->kv_self_v->nb[2] + (size_t)kv_offset * ctx->kv_self_v->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(gctx, v, v_cache_slice));
    }

    // Read full KV from cache: [kv_dim, n_kv] where n_kv = kv_offset + n_tokens
    const int32_t n_kv = kv_offset + n_tokens;
    ggml_tensor * k_full = ggml_view_2d(gctx, ctx->kv_self_k,
        kv_dim, n_kv,
        ctx->kv_self_k->nb[1],
        layer_idx * ctx->kv_self_k->nb[2]); // [kv_dim, n_kv]
    ggml_tensor * v_full = ggml_view_2d(gctx, ctx->kv_self_v,
        kv_dim, n_kv,
        ctx->kv_self_v->nb[1],
        layer_idx * ctx->kv_self_v->nb[2]); // [kv_dim, n_kv]

    // Flash attention with GQA
    // Q: [n_heads*head_dim, n_tokens] -> [head_dim, n_heads, n_tokens] -> [head_dim, n_tokens, n_heads]
    ggml_tensor * q3 = ggml_reshape_3d(gctx, q, VOXTRAL_DEC_HEAD_DIM, VOXTRAL_DEC_HEADS, n_tokens);
    q3 = ggml_permute(gctx, q3, 0, 2, 1, 3); // [head_dim, n_tokens, n_heads]

    // K: [kv_dim, n_kv] -> [head_dim, n_kv_heads, n_kv] -> [head_dim, n_kv, n_kv_heads]
    ggml_tensor * k3 = ggml_reshape_3d(gctx, k_full, VOXTRAL_DEC_HEAD_DIM, VOXTRAL_DEC_KV_HEADS, n_kv);
    k3 = ggml_permute(gctx, k3, 0, 2, 1, 3); // [head_dim, n_kv, n_kv_heads]

    // V: [kv_dim, n_kv] -> [head_dim, n_kv_heads, n_kv] -> [head_dim, n_kv, n_kv_heads]
    ggml_tensor * v3 = ggml_reshape_3d(gctx, v_full, VOXTRAL_DEC_HEAD_DIM, VOXTRAL_DEC_KV_HEADS, n_kv);
    v3 = ggml_permute(gctx, v3, 0, 2, 1, 3); // [head_dim, n_kv, n_kv_heads]

    const float scale = 1.0f / sqrtf((float)VOXTRAL_DEC_HEAD_DIM);

    // ggml_flash_attn_ext fuses Q@K^T, scale, mask, softmax, @V in one op
    // GQA broadcast is built-in (n_heads % n_kv_heads == 0)
    // Mask is cast to F16 inside the graph if provided
    ggml_tensor * attn_mask_f16 = attn_mask ? ggml_cast(gctx, attn_mask, GGML_TYPE_F16) : nullptr;
    ggml_tensor * attn_out = ggml_flash_attn_ext(gctx, q3, k3, v3, attn_mask_f16, scale, 0.0f, 0.0f);
    // Output: [head_dim, n_heads, n_tokens] (already permuted by flash_attn_ext)
    attn_out = ggml_cont(gctx, attn_out);
    attn_out = ggml_reshape_2d(gctx, attn_out, VOXTRAL_DEC_HEADS * VOXTRAL_DEC_HEAD_DIM, n_tokens);

    // Output projection + residual
    ggml_tensor * attn_proj = ggml_mul_mat(gctx, L.attn_o_weight, attn_out); // [dec_dim, n_tokens]
    x = ggml_add(gctx, residual, attn_proj); // [dec_dim, n_tokens]

    // Pre-FFN RMS norm
    residual = x; // [dec_dim, n_tokens]
    ggml_tensor * h_norm = ggml_rms_norm(gctx, x, VOXTRAL_DEC_NORM_EPS); // [dec_dim, n_tokens]
    h_norm = ggml_mul(gctx, h_norm, L.ffn_norm_weight); // [dec_dim, n_tokens]

    // Ada time conditioning: h_norm = h_norm * (1 + ada_mlp(time_emb))
    // = h_norm + h_norm * ada_scale
    // ada_mlp: Linear(3072->32) -> GELU -> Linear(32->3072)
    {
        ggml_tensor * ada_hidden = ggml_mul_mat(gctx, L.ada0_weight, time_emb); // [ada_dim]
        ada_hidden = ggml_gelu_erf(gctx, ada_hidden); // [ada_dim]
        ggml_tensor * ada_scale = ggml_mul_mat(gctx, L.ada2_weight, ada_hidden); // [dec_dim]

        // h_norm * (1 + ada_scale) = h_norm + h_norm * ada_scale
        ggml_tensor * scaled = ggml_mul(gctx, h_norm, ada_scale); // [dec_dim, n_tokens]
        h_norm = ggml_add(gctx, h_norm, scaled); // [dec_dim, n_tokens]
    }

    // SwiGLU FFN
    ggml_tensor * gate = ggml_mul_mat(gctx, L.ffn_w1_weight, h_norm); // [dec_hidden, n_tokens]
    gate = ggml_silu(gctx, gate); // [dec_hidden, n_tokens]
    ggml_tensor * up = ggml_mul_mat(gctx, L.ffn_w3_weight, h_norm); // [dec_hidden, n_tokens]
    ggml_tensor * ffn_out = ggml_mul(gctx, gate, up); // [dec_hidden, n_tokens]
    ffn_out = ggml_mul_mat(gctx, L.ffn_w2_weight, ffn_out); // [dec_dim, n_tokens]

    x = ggml_add(gctx, residual, ffn_out); // [dec_dim, n_tokens]

    return x;
}

// ============================================================================
// Graph Building: Decoder Prefill
// ============================================================================

static ggml_cgraph * build_decoder_prefill_graph(
    voxtral_context     * ctx,
    ggml_context * gctx,
    int32_t               n_tokens)  // number of prompt tokens
{
    voxtral_model * model = ctx->model;
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, GGML_DEFAULT_GRAPH_SIZE * 4, false);

    // Token IDs input: [n_tokens] int32
    ggml_tensor * token_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(token_ids, "token_ids");
    ggml_backend_sched_set_tensor_backend(ctx->sched_dec_pre, token_ids, ctx->backend);

    // Position indices: [n_tokens] int32
    ggml_tensor * positions = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(positions, "positions");
    ggml_backend_sched_set_tensor_backend(ctx->sched_dec_pre, positions, ctx->backend);

    // Time embedding: [dec_dim]
    ggml_tensor * time_emb = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, VOXTRAL_DEC_DIM);
    ggml_set_name(time_emb, "time_emb");
    ggml_backend_sched_set_tensor_backend(ctx->sched_dec_pre, time_emb, ctx->backend);

    // Token embeddings: [dec_dim, n_tokens]
    ggml_tensor * tok_emb = ggml_get_rows(gctx, model->tok_embeddings_weight, token_ids); // [dec_dim, n_tokens]

    // Audio embeddings from decoder_memory: [dec_dim, n_tokens]
    ggml_tensor * audio_emb = ggml_view_2d(gctx, ctx->decoder_memory,
        VOXTRAL_DEC_DIM, n_tokens,
        ctx->decoder_memory->nb[1], 0); // [dec_dim, n_tokens]

    // Combined input: tok_emb + audio_emb
    ggml_tensor * x = ggml_add(gctx, tok_emb, audio_emb); // [dec_dim, n_tokens]

    // Causal mask for prefill: [n_tokens, n_tokens] additive mask
    // -inf for positions that should not attend
    ggml_tensor * causal_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_tokens, n_tokens);
    ggml_set_name(causal_mask, "causal_mask");
    ggml_backend_sched_set_tensor_backend(ctx->sched_dec_pre, causal_mask, ctx->backend);

    // Decoder layers
    for (int32_t i = 0; i < VOXTRAL_DEC_LAYERS; i++) {
        x = build_decoder_layer(ctx, gctx, gf, x, positions, time_emb,
            i, n_tokens, /*kv_offset=*/0, causal_mask);
    }

    // Final norm
    x = ggml_rms_norm(gctx, x, VOXTRAL_DEC_NORM_EPS); // [dec_dim, n_tokens]
    x = ggml_mul(gctx, x, model->dec_norm_weight); // [dec_dim, n_tokens]

    // Logits for last token only: extract last token -> matmul with embeddings
    ggml_tensor * last_hidden = ggml_view_1d(gctx, x, VOXTRAL_DEC_DIM,
        (n_tokens - 1) * x->nb[1]); // [dec_dim]

    ggml_tensor * logits = ggml_mul_mat(gctx, model->tok_embeddings_weight, last_hidden); // [vocab_size]

    // Copy logits to persistent
    ggml_build_forward_expand(gf, ggml_cpy(gctx, logits, ctx->decoder_logits));

    return gf;
}

// ============================================================================
// Graph Building: Decoder Step (single token)
// ============================================================================

static ggml_cgraph * build_decoder_step_graph(
    voxtral_context     * ctx,
    ggml_context * gctx,
    int32_t               position,    // absolute position
    int32_t               audio_pos)   // position in audio embeddings (may differ)
{
    voxtral_model * model = ctx->model;
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, GGML_DEFAULT_GRAPH_SIZE * 4, false);

    const int32_t kv_used = ctx->kv_used;  // tokens already in KV cache

    // Token ID input: [1] int32
    ggml_tensor * token_id = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    ggml_set_name(token_id, "token_id");
    ggml_backend_sched_set_tensor_backend(ctx->sched_dec_step, token_id, ctx->backend);

    // Position: [1] int32
    ggml_tensor * pos_tensor = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    ggml_set_name(pos_tensor, "position");
    ggml_backend_sched_set_tensor_backend(ctx->sched_dec_step, pos_tensor, ctx->backend);

    // Time embedding: [dec_dim]
    ggml_tensor * time_emb = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, VOXTRAL_DEC_DIM);
    ggml_set_name(time_emb, "time_emb");
    ggml_backend_sched_set_tensor_backend(ctx->sched_dec_step, time_emb, ctx->backend);

    // Token embedding: [dec_dim, 1]
    ggml_tensor * tok_emb = ggml_get_rows(gctx, model->tok_embeddings_weight, token_id); // [dec_dim, 1]

    // Audio embedding from decoder_memory at audio_pos
    ggml_tensor * audio_emb = ggml_view_2d(gctx, ctx->decoder_memory,
        VOXTRAL_DEC_DIM, 1,
        ctx->decoder_memory->nb[1],
        (size_t)audio_pos * ctx->decoder_memory->nb[1]); // [dec_dim, 1]

    ggml_tensor * x = ggml_add(gctx, tok_emb, audio_emb); // [dec_dim, 1]

    // Decoder layers (no mask needed for single token - all KV positions are valid)
    for (int32_t i = 0; i < VOXTRAL_DEC_LAYERS; i++) {
        x = build_decoder_layer(ctx, gctx, gf, x, pos_tensor, time_emb,
            i, 1, /*kv_offset=*/kv_used, /*attn_mask=*/nullptr);
    }

    // Final norm
    x = ggml_rms_norm(gctx, x, VOXTRAL_DEC_NORM_EPS); // [dec_dim, 1]
    x = ggml_mul(gctx, x, model->dec_norm_weight); // [dec_dim, 1]

    // Logits
    ggml_tensor * x_flat = ggml_reshape_1d(gctx, x, VOXTRAL_DEC_DIM); // [dec_dim]
    ggml_tensor * logits = ggml_mul_mat(gctx, model->tok_embeddings_weight, x_flat); // [vocab_size]

    // Copy to persistent
    ggml_build_forward_expand(gf, ggml_cpy(gctx, logits, ctx->decoder_logits));

    return gf;
}

// ============================================================================
// Helper: set named input tensors in a graph
// ============================================================================

static ggml_tensor * find_tensor_in_graph(ggml_cgraph * gf, const char * name) {
    return ggml_graph_get_tensor(gf, name);
}

// ============================================================================
// Run Encoder
// ============================================================================

// Run a single encoder chunk: build graph, set inputs, compute, return seq_len
static bool run_encoder_chunk(
    voxtral_context * ctx,
    const float * chunk_mel_data,  // [n_mel, chunk_mel_frames]
    int32_t chunk_mel_frames,
    int32_t rope_pos_offset,
    int32_t * out_seq_len)
{
    const size_t meta_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 4 +
                             ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE * 4, false);
    std::vector<uint8_t> meta_buf(meta_size);

    ggml_init_params p = {
        /*.mem_size  =*/ meta_size,
        /*.mem_buffer=*/ meta_buf.data(),
        /*.no_alloc  =*/ true,
    };
    ggml_context * gctx = ggml_init(p);

    int32_t chunk_seq_len = 0;
    ggml_cgraph * gf = build_encoder_graph(ctx, gctx, chunk_mel_data, chunk_mel_frames, &chunk_seq_len);

    ggml_backend_sched_reset(ctx->sched_encoder);
    if (!ggml_backend_sched_alloc_graph(ctx->sched_encoder, gf)) {
        LOG_ERR(ctx, "encoder chunk: failed to allocate graph");
        ggml_free(gctx);
        return false;
    }

    // Set mel input
    ggml_tensor * mel_t = find_tensor_in_graph(gf, "mel_input");
    if (mel_t) {
        const int64_t expected_ne0 = chunk_mel_frames;
        const int64_t expected_ne1 = VOXTRAL_NUM_MEL_BINS;
        if (mel_t->ne[0] == expected_ne0 && mel_t->ne[1] == expected_ne1) {
            ggml_backend_tensor_set(mel_t, chunk_mel_data, 0,
                (size_t) VOXTRAL_NUM_MEL_BINS * chunk_mel_frames * sizeof(float));
        } else if (mel_t->ne[0] == expected_ne1 && mel_t->ne[1] == expected_ne0) {
            std::vector<float> mel_tbuf((size_t) chunk_mel_frames * VOXTRAL_NUM_MEL_BINS);
            for (int32_t m = 0; m < VOXTRAL_NUM_MEL_BINS; ++m) {
                const float * src = chunk_mel_data + (size_t) m * chunk_mel_frames;
                for (int32_t f = 0; f < chunk_mel_frames; ++f) {
                    mel_tbuf[(size_t) m + (size_t) VOXTRAL_NUM_MEL_BINS * f] = src[f];
                }
            }
            ggml_backend_tensor_set(mel_t, mel_tbuf.data(), 0,
                (size_t) VOXTRAL_NUM_MEL_BINS * chunk_mel_frames * sizeof(float));
        } else {
            ggml_backend_tensor_set(mel_t, chunk_mel_data, 0,
                (size_t) VOXTRAL_NUM_MEL_BINS * chunk_mel_frames * sizeof(float));
        }
    }

    // Set positions with RoPE offset for absolute positions across chunks
    ggml_tensor * pos_t = find_tensor_in_graph(gf, "enc_positions");
    if (pos_t) {
        std::vector<int32_t> pos(chunk_seq_len);
        std::iota(pos.begin(), pos.end(), rope_pos_offset);
        ggml_backend_tensor_set(pos_t, pos.data(), 0, chunk_seq_len * sizeof(int32_t));
    }

    // Set encoder sliding causal mask (local to chunk)
    ggml_tensor * mask_t = find_tensor_in_graph(gf, "enc_attn_mask");
    if (mask_t) {
        std::vector<float> mask((size_t) chunk_seq_len * chunk_seq_len);
        for (int32_t q = 0; q < chunk_seq_len; ++q) {
            const int32_t min_kv = std::max<int32_t>(0, q - (VOXTRAL_ENC_WINDOW - 1));
            for (int32_t kv = 0; kv < chunk_seq_len; ++kv) {
                const bool allow = (kv <= q) && (kv >= min_kv);
                mask[(size_t) q * chunk_seq_len + kv] = allow ? 0.0f : -INFINITY;
            }
        }
        ggml_backend_tensor_set(mask_t, mask.data(), 0, mask.size() * sizeof(float));
    }

    // Compute
    ggml_backend_sched_graph_compute(ctx->sched_encoder, gf);
    ggml_backend_sched_reset(ctx->sched_encoder);
    ggml_free(gctx);

    if (out_seq_len) *out_seq_len = chunk_seq_len;
    return true;
}

// Process mel spectrogram in overlapping chunks, accumulating encoder output on device
static bool run_encoder_chunked(voxtral_context * ctx, const float * mel_data, int32_t total_mel_frames) {
    const int32_t mel_overlap = VOXTRAL_ENC_CHUNK_OVERLAP * 2;  // mel frames of overlap (1500)
    const int32_t mel_stride = VOXTRAL_ENC_CHUNK_MEL - mel_overlap;  // 1500

    // Pre-compute total encoder tokens for allocation
    int32_t alloc_total = compute_total_enc_tokens(total_mel_frames);
    if (alloc_total <= 0) {
        LOG_ERR(ctx, "encoder: audio too short to produce encoder tokens");
        return false;
    }

    // Allocate encoder_output on device
    if (!alloc_encoder_output(ctx, alloc_total)) {
        LOG_ERR(ctx, "encoder: failed to allocate encoder output (%d tokens, %.2f MB)",
                alloc_total, (double) alloc_total * VOXTRAL_ENC_DIM * sizeof(float) / 1e6);
        return false;
    }

    LOG_INFO(ctx, "encoder chunked: %d mel frames, %d alloc enc tokens, mel_stride=%d",
             total_mel_frames, alloc_total, mel_stride);

    int32_t mel_offset = 0;
    int32_t enc_write_offset = 0;
    int32_t chunk_idx = 0;

    while (mel_offset < total_mel_frames) {
        int32_t chunk_mel_frames = std::min(VOXTRAL_ENC_CHUNK_MEL, total_mel_frames - mel_offset);

        // Pre-check: will this chunk contribute any new tokens?
        // This avoids building and running the full encoder graph for nothing.
        int32_t skip = (chunk_idx > 0) ? VOXTRAL_ENC_CHUNK_OVERLAP : 0;
        {
            int32_t expected_tokens = mel_frames_to_enc_tokens(chunk_mel_frames);
            if (expected_tokens - skip <= 0) {
                LOG_DBG(ctx, "encoder chunk %d: skipped (expected %d tokens, skip=%d)",
                        chunk_idx, expected_tokens, skip);
                break;
            }
        }

        // For single-chunk case (entire mel fits), use mel_data directly to avoid copy
        const float * chunk_mel_ptr = nullptr;
        std::vector<float> chunk_mel_buf;
        if (mel_offset == 0 && chunk_mel_frames == total_mel_frames) {
            // Single chunk — mel_data is already in [n_mel, total_frames] layout
            chunk_mel_ptr = mel_data;
        } else {
            // Multi-chunk — extract sub-range of frames for this chunk
            chunk_mel_buf.resize((size_t) VOXTRAL_NUM_MEL_BINS * chunk_mel_frames);
            for (int32_t m = 0; m < VOXTRAL_NUM_MEL_BINS; m++) {
                memcpy(chunk_mel_buf.data() + (size_t) m * chunk_mel_frames,
                       mel_data + (size_t) m * total_mel_frames + mel_offset,
                       chunk_mel_frames * sizeof(float));
            }
            chunk_mel_ptr = chunk_mel_buf.data();
        }

        int32_t rope_offset = enc_write_offset - skip;

        // Run encoder for this chunk
        int32_t chunk_seq_len = 0;
        if (!run_encoder_chunk(ctx, chunk_mel_ptr, chunk_mel_frames, rope_offset, &chunk_seq_len)) {
            LOG_ERR(ctx, "encoder chunk %d: failed", chunk_idx);
            return false;
        }

        int32_t stride = chunk_seq_len - skip;
        if (stride <= 0) {
            LOG_DBG(ctx, "encoder chunk %d: no new tokens (seq_len=%d, skip=%d), stopping",
                    chunk_idx, chunk_seq_len, skip);
            break;
        }

        // Clamp stride to not overflow pre-allocated buffer
        if (enc_write_offset + stride > alloc_total) {
            stride = alloc_total - enc_write_offset;
            if (stride <= 0) break;
        }

        LOG_INFO(ctx, "encoder chunk %d: mel[%d..%d) enc_tokens=%d skip=%d stride=%d rope_offset=%d",
                 chunk_idx, mel_offset, mel_offset + chunk_mel_frames,
                 chunk_seq_len, skip, stride, rope_offset);

        // Copy stride portion from encoder_chunk_output to encoder_output
        // Goes through CPU (device->CPU->device), but stride data is small (~3.8 MB max)
        {
            const size_t elem_bytes = VOXTRAL_ENC_DIM * sizeof(float);
            const size_t src_offset = (size_t) skip * elem_bytes;
            const size_t dst_offset = (size_t) enc_write_offset * elem_bytes;
            const size_t copy_bytes = (size_t) stride * elem_bytes;

            std::vector<uint8_t> tmp(copy_bytes);
            ggml_backend_tensor_get(ctx->encoder_chunk_output, tmp.data(), src_offset, copy_bytes);
            ggml_backend_tensor_set(ctx->encoder_output, tmp.data(), dst_offset, copy_bytes);
        }

        enc_write_offset += stride;
        mel_offset += mel_stride;
        chunk_idx++;
    }

    // Trim to multiple of downsample factor for adapter compatibility
    ctx->enc_seq_used = (enc_write_offset / VOXTRAL_DOWNSAMPLE_FACTOR) * VOXTRAL_DOWNSAMPLE_FACTOR;
    ctx->total_enc_tokens = ctx->enc_seq_used;

    LOG_INFO(ctx, "encoder done: %d chunks, enc_seq_used=%d (raw=%d)",
             chunk_idx, ctx->enc_seq_used, enc_write_offset);
    return true;
}

// ============================================================================
// Run Adapter
// ============================================================================

static bool run_adapter(voxtral_context * ctx) {
    const int32_t enc_seq = ctx->enc_seq_used;
    const int32_t dec_seq = enc_seq / VOXTRAL_DOWNSAMPLE_FACTOR;

    LOG_INFO(ctx, "running adapter: enc_seq=%d -> dec_seq=%d", enc_seq, dec_seq);

    // Allocate decoder_memory for this utterance
    if (!alloc_decoder_memory(ctx, dec_seq)) {
        LOG_ERR(ctx, "adapter: failed to allocate decoder memory (%d tokens, %.2f MB)",
                dec_seq, (double) dec_seq * VOXTRAL_DEC_DIM * sizeof(float) / 1e6);
        return false;
    }

    const size_t meta_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE +
                             ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, false);
    std::vector<uint8_t> meta_buf(meta_size);

    ggml_init_params p = {
        /*.mem_size  =*/ meta_size,
        /*.mem_buffer=*/ meta_buf.data(),
        /*.no_alloc  =*/ true,
    };
    ggml_context * gctx = ggml_init(p);

    ggml_cgraph * gf = build_adapter_graph(ctx, gctx);
    log_graph_info(ctx, "adapter", gf);

    ggml_backend_sched_reset(ctx->sched_adapter);
    if (!ggml_backend_sched_alloc_graph(ctx->sched_adapter, gf)) {
        LOG_ERR(ctx, "adapter: failed to allocate graph");
        ggml_free(gctx);
        return false;
    }

    ggml_backend_sched_graph_compute(ctx->sched_adapter, gf);
    ggml_backend_sched_reset(ctx->sched_adapter);
    ggml_free(gctx);

    LOG_INFO(ctx, "adapter done: dec_seq_len=%d (%.2f MB on device)",
             ctx->dec_seq_len,
             (double) ggml_nbytes(ctx->decoder_memory) / 1e6);
    return true;
}

// ============================================================================
// Run Decoder Prefill
// ============================================================================

static bool run_decoder_prefill(
    voxtral_context * ctx,
    const int32_t   * token_ids,
    int32_t           n_tokens,
    float           * logits_out)  // [vocab_size]
{
    LOG_INFO(ctx, "decoder prefill: %d tokens", n_tokens);

    if (n_tokens > VOXTRAL_DEC_WINDOW) {
        LOG_ERR(ctx, "decoder prefill: n_tokens=%d exceeds DEC_WINDOW=%d", n_tokens, VOXTRAL_DEC_WINDOW);
        return false;
    }

    const size_t meta_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 4 +
                             ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE * 4, false);
    std::vector<uint8_t> meta_buf(meta_size);

    ggml_init_params p = {
        /*.mem_size  =*/ meta_size,
        /*.mem_buffer=*/ meta_buf.data(),
        /*.no_alloc  =*/ true,
    };
    ggml_context * gctx = ggml_init(p);

    ggml_cgraph * gf = build_decoder_prefill_graph(ctx, gctx, n_tokens);
    log_graph_info(ctx, "decoder prefill", gf);

    ggml_backend_sched_reset(ctx->sched_dec_pre);
    if (!ggml_backend_sched_alloc_graph(ctx->sched_dec_pre, gf)) {
        LOG_ERR(ctx, "decoder prefill: failed to allocate graph");
        ggml_free(gctx);
        return false;
    }

    // Set inputs
    ggml_tensor * tok_t = find_tensor_in_graph(gf, "token_ids");
    if (tok_t) {
        ggml_backend_tensor_set(tok_t, token_ids, 0, n_tokens * sizeof(int32_t));
    }

    ggml_tensor * pos_t = find_tensor_in_graph(gf, "positions");
    if (pos_t) {
        std::vector<int32_t> pos(n_tokens);
        std::iota(pos.begin(), pos.end(), 0);
        ggml_backend_tensor_set(pos_t, pos.data(), 0, n_tokens * sizeof(int32_t));
    }

    ggml_tensor * time_t = find_tensor_in_graph(gf, "time_emb");
    if (time_t) {
        ggml_backend_tensor_set(time_t, ctx->time_emb_cpu.data(), 0, VOXTRAL_DEC_DIM * sizeof(float));
    }

    // Set causal mask: lower-triangular (0 for allowed, -inf for masked)
    ggml_tensor * mask_t = find_tensor_in_graph(gf, "causal_mask");
    if (mask_t) {
        std::vector<float> mask(n_tokens * n_tokens);
        for (int32_t i = 0; i < n_tokens; i++) {
            for (int32_t j = 0; j < n_tokens; j++) {
                mask[i * n_tokens + j] = (j <= i) ? 0.0f : -INFINITY;
            }
        }
        ggml_backend_tensor_set(mask_t, mask.data(), 0, mask.size() * sizeof(float));
    }

    // Compute
    ggml_backend_sched_graph_compute(ctx->sched_dec_pre, gf);

    // Read logits
    ggml_backend_tensor_get(ctx->decoder_logits, logits_out, 0, VOXTRAL_VOCAB_SIZE * sizeof(float));

    ctx->kv_used = std::min(n_tokens, VOXTRAL_DEC_WINDOW);

    ggml_backend_sched_reset(ctx->sched_dec_pre);
    ggml_free(gctx);

    LOG_INFO(ctx, "decoder prefill done");
    return true;
}

// ============================================================================
// Run Decoder Step
// ============================================================================

static bool run_decoder_step(
    voxtral_context * ctx,
    int32_t           token_id,
    int32_t           position,     // absolute position in decoder sequence
    int32_t           audio_pos,    // position in adapter output for audio embedding
    float           * logits_out)   // [vocab_size]
{
    if (ctx->kv_used >= VOXTRAL_DEC_WINDOW) {
        kv_cache_shift_left(ctx, 1);
        ctx->kv_used = VOXTRAL_DEC_WINDOW - 1;
    }

    // Use thread-local buffer to avoid per-step heap allocation
    static thread_local std::vector<uint8_t> step_meta_buf;
    const size_t meta_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 4 +
                             ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE * 4, false);
    if (step_meta_buf.size() < meta_size) {
        step_meta_buf.resize(meta_size);
    }

    ggml_init_params p = {
        /*.mem_size  =*/ meta_size,
        /*.mem_buffer=*/ step_meta_buf.data(),
        /*.no_alloc  =*/ true,
    };
    ggml_context * gctx = ggml_init(p);

    ggml_cgraph * gf = build_decoder_step_graph(ctx, gctx, position, audio_pos);

    ggml_backend_sched_reset(ctx->sched_dec_step);
    if (!ggml_backend_sched_alloc_graph(ctx->sched_dec_step, gf)) {
        LOG_ERR(ctx, "decoder step: failed to allocate graph");
        ggml_free(gctx);
        return false;
    }

    // Set inputs
    ggml_tensor * tok_t = find_tensor_in_graph(gf, "token_id");
    if (tok_t) {
        ggml_backend_tensor_set(tok_t, &token_id, 0, sizeof(int32_t));
    }

    ggml_tensor * pos_t = find_tensor_in_graph(gf, "position");
    if (pos_t) {
        ggml_backend_tensor_set(pos_t, &position, 0, sizeof(int32_t));
    }

    ggml_tensor * time_t = find_tensor_in_graph(gf, "time_emb");
    if (time_t) {
        ggml_backend_tensor_set(time_t, ctx->time_emb_cpu.data(), 0, VOXTRAL_DEC_DIM * sizeof(float));
    }

    // Compute
    ggml_backend_sched_graph_compute(ctx->sched_dec_step, gf);

    // Read logits
    ggml_backend_tensor_get(ctx->decoder_logits, logits_out, 0, VOXTRAL_VOCAB_SIZE * sizeof(float));

    ctx->kv_used += 1;

    ggml_backend_sched_reset(ctx->sched_dec_step);
    ggml_free(gctx);

    return true;
}

// ============================================================================
// High-level: Transcribe
// ============================================================================

static bool voxtral_transcribe_from_audio(
    voxtral_context & ctx,
    const float     * audio,
    int32_t           n_samples,
    int32_t           max_tokens,
    voxtral_result  & result,
    bool              log_audio)
{
    result.text.clear();
    result.tokens.clear();
    result.first_step_logits.clear();

    if (audio == nullptr || n_samples <= 0) {
        LOG_ERR(&ctx, "audio input is empty");
        return false;
    }

    if (log_audio) {
        LOG_INFO(&ctx, "audio input: %d samples (%.1f s)", n_samples,
            (float)n_samples / VOXTRAL_SAMPLE_RATE);
    }

    // 2. Streaming padding (matching Python pad_audio_streaming)
    constexpr int32_t mult_of   = VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK;   // 1280
    const int32_t n_raw     = n_samples;
    const int32_t align_pad = (mult_of - (n_raw % mult_of)) % mult_of;
    const int32_t right_pad = align_pad + VOXTRAL_N_RIGHT_PAD_TOKENS * mult_of;
    constexpr int32_t left_pad  = VOXTRAL_N_LEFT_PAD_TOKENS * mult_of;

    std::vector<float> padded(left_pad + n_raw + right_pad, 0.0f);
    memcpy(padded.data() + left_pad, audio, n_raw * sizeof(float));

    LOG_INFO(&ctx, "padded audio: %d samples (left=%d, right=%d)", (int)padded.size(), left_pad, right_pad);

    // 3. Compute mel spectrogram
    int32_t n_frames = 0;
    const int32_t max_frames = (int32_t)padded.size() / VOXTRAL_HOP_LENGTH + 1;
    std::vector<float> mel_data(VOXTRAL_NUM_MEL_BINS * max_frames);

    compute_mel_spectrogram(
        padded.data(), (int32_t)padded.size(),
        ctx.mel_filters_cpu.data(),
        ctx.hann_window.data(),
        mel_data.data(), &n_frames);

    LOG_INFO(&ctx, "mel spectrogram: %d frames", n_frames);

    // Truncate to even number of frames (for conv stride=2)
    if (n_frames % 2 != 0) {
        // Drop first frame (matching Python mel[:, 1:])
        // Shift mel data left by 1 frame
        for (int32_t m = 0; m < VOXTRAL_NUM_MEL_BINS; m++) {
            memmove(mel_data.data() + m * (n_frames - 1),
                    mel_data.data() + m * n_frames + 1,
                    (n_frames - 1) * sizeof(float));
        }
        n_frames -= 1;
        LOG_INFO(&ctx, "mel truncated to %d frames (even)", n_frames);
    }

    // 4. Run encoder (chunked for arbitrarily long audio)
    auto t_encoder = std::chrono::steady_clock::now();
    if (!run_encoder_chunked(&ctx, mel_data.data(), n_frames)) {
        return false;
    }
    LOG_INFO(&ctx, "encoder time: %.1f ms", elapsed_ms(t_encoder));

    // 5. Run adapter
    auto t_adapter = std::chrono::steady_clock::now();
    if (!run_adapter(&ctx)) {
        return false;
    }
    LOG_INFO(&ctx, "adapter time: %.1f ms", elapsed_ms(t_adapter));

    const int32_t n_audio = ctx.dec_seq_len;

    // 6. Build prompt tokens: [BOS] + [STREAMING_PAD] * (N_LEFT_PAD_TOKENS + N_DELAY_TOKENS)
    std::vector<int32_t> prompt_ids;
    prompt_ids.push_back(VOXTRAL_TOKEN_BOS);
    for (int32_t i = 0; i < VOXTRAL_N_LEFT_PAD_TOKENS + VOXTRAL_N_DELAY_TOKENS; i++) {
        prompt_ids.push_back(VOXTRAL_TOKEN_STREAMING_PAD);
    }
    const int32_t L = (int32_t)prompt_ids.size();  // 39

    LOG_INFO(&ctx, "prompt: %d tokens, audio_tokens: %d", L, n_audio);

    if (L > n_audio) {
        LOG_ERR(&ctx, "prompt length %d exceeds audio tokens %d", L, n_audio);
        return false;
    }

    // 7. Reset KV cache
    clear_kv_cache(&ctx);

    // 8. Decoder prefill
    auto t_prefill = std::chrono::steady_clock::now();
    std::vector<float> logits(VOXTRAL_VOCAB_SIZE);
    if (L > 1) {
        if (!run_decoder_prefill(&ctx, prompt_ids.data(), L - 1, logits.data())) {
            return false;
        }
    }

    // 8b. One step with last prefix token (matches Python prefill + forward_one)
    if (!run_decoder_step(&ctx, prompt_ids[L - 1], L - 1, L - 1, logits.data())) {
        return false;
    }
    LOG_INFO(&ctx, "prefill time: %.1f ms", elapsed_ms(t_prefill));

    // First token from prefill
    int32_t token = (int32_t)(std::max_element(logits.begin(), logits.end()) - logits.begin());

    // Store first step logits
    result.first_step_logits = logits;
    result.tokens.push_back(token);

    LOG_INFO(&ctx, "first token: %d", token);

    // 9. Autoregressive decoding
    auto t_decode = std::chrono::steady_clock::now();
    int32_t consecutive_pad = 0;
    bool seen_text = false;
    for (int32_t pos = L; pos < n_audio && (int32_t)result.tokens.size() < max_tokens; pos++) {
        if (token == VOXTRAL_TOKEN_EOS) break;

        if (!run_decoder_step(&ctx, token, pos, pos, logits.data())) {
            return false;
        }

        // Greedy argmax
        token = (int32_t)(std::max_element(logits.begin(), logits.end()) - logits.begin());

        result.tokens.push_back(token);

        // Early stopping: if we've seen real text and then get enough
        // consecutive streaming pad tokens, the transcript is complete.
        if (token == VOXTRAL_TOKEN_STREAMING_PAD) {
            consecutive_pad++;
        } else {
            consecutive_pad = 0;
            if (token >= ctx.model->tokenizer_num_special_tokens) {
                seen_text = true;
            }
        }
        if (seen_text && consecutive_pad >= VOXTRAL_N_RIGHT_PAD_TOKENS) {
            LOG_INFO(&ctx, "early stop: %d consecutive pad tokens after text", consecutive_pad);
            break;
        }
    }
    LOG_INFO(&ctx, "decode time: %.1f ms (%d steps, %.1f ms/step)",
        elapsed_ms(t_decode), (int)result.tokens.size() - 1,
        result.tokens.size() > 1 ? elapsed_ms(t_decode) / (result.tokens.size() - 1) : 0.0);

    // Remove trailing EOS
    if (!result.tokens.empty() && result.tokens.back() == VOXTRAL_TOKEN_EOS) {
        result.tokens.pop_back();
    }

    LOG_INFO(&ctx, "generated %d tokens", (int)result.tokens.size());

    // 10. Decode tokens to text (Tekken vocab from GGUF metadata)
    result.text = decode_tokens(*ctx.model, result.tokens);

    return true;
}

bool voxtral_transcribe_audio(
    voxtral_context   & ctx,
    const std::vector<float> & audio,
    int32_t             max_tokens,
    voxtral_result    & result)
{
    return voxtral_transcribe_from_audio(
        ctx, audio.data(), (int32_t) audio.size(), max_tokens, result, true);
}

bool voxtral_transcribe_file(
    voxtral_context   & ctx,
    const std::string & audio_path,
    int32_t             max_tokens,
    voxtral_result    & result)
{
    std::vector<float> audio;
    if (!load_wav_file(audio_path, audio)) {
        LOG_ERR(&ctx, "failed to load WAV: %s", audio_path.c_str());
        return false;
    }
    LOG_INFO(&ctx, "audio loaded: %d samples (%.1f s)", (int)audio.size(),
        (float)audio.size() / VOXTRAL_SAMPLE_RATE);

    return voxtral_transcribe_from_audio(
        ctx, audio.data(), (int32_t) audio.size(), max_tokens, result, false);
}

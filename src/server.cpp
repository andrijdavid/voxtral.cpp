#include "voxtral.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <signal.h>
#include <sstream>
#include <string>
#include <vector>

#include "httplib.h"

namespace {

// ────────────────────────────────────────────────────────────────────
// JSON helpers (no external dependency)
// ────────────────────────────────────────────────────────────────────

std::string json_escape(const std::string & s) {
    std::ostringstream os;
    for (unsigned char c : s) {
        switch (c) {
            case '"':  os << "\\\""; break;
            case '\\': os << "\\\\"; break;
            case '\b': os << "\\b";  break;
            case '\f': os << "\\f";  break;
            case '\n': os << "\\n";  break;
            case '\r': os << "\\r";  break;
            case '\t': os << "\\t";  break;
            default:
                if (c < 0x20) {
                    os << "\\u" << std::hex << std::setfill('0')
                       << std::setw(4) << static_cast<int>(c);
                } else {
                    os << c;
                }
        }
    }
    return os.str();
}

std::string make_result_json(const std::string & text, double duration_s) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(3)
       << "{\"text\":\"" << json_escape(text) << "\""
       << ",\"duration\":" << duration_s << "}";
    return os.str();
}

std::string make_error_json(int code, const std::string & message,
                            const std::string & type = "invalid_request_error") {
    std::ostringstream os;
    os << "{\"error\":{\"message\":\"" << json_escape(message)
       << "\",\"type\":\"" << json_escape(type)
       << "\",\"code\":" << code << "}}";
    return os.str();
}

// ────────────────────────────────────────────────────────────────────
// Base64 decoder
// ────────────────────────────────────────────────────────────────────

std::vector<uint8_t> base64_decode(const std::string & input) {
    static const std::string chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::vector<uint8_t> result;
    result.reserve(input.size() * 3 / 4);

    int val = 0, bits = -8;
    for (unsigned char c : input) {
        if (c == '=' || c == '\n' || c == '\r' || c == ' ') {
            continue;
        }
        auto pos = chars.find(static_cast<char>(c));
        if (pos == std::string::npos) {
            continue;
        }
        val = (val << 6) | static_cast<int>(pos);
        bits += 6;
        if (bits >= 0) {
            result.push_back(static_cast<uint8_t>((val >> bits) & 0xFF));
            bits -= 8;
        }
    }
    return result;
}

// ────────────────────────────────────────────────────────────────────
// Temporary file with automatic cleanup
// ────────────────────────────────────────────────────────────────────

struct temp_file {
    std::string path;

    explicit temp_file(const std::string & suffix = ".wav") {
        static std::atomic<uint64_t> counter{0};
        auto dir = std::filesystem::temp_directory_path();
        auto ts  = std::chrono::steady_clock::now().time_since_epoch().count();
        auto name = "voxtral_" + std::to_string(counter.fetch_add(1))
                   + "_" + std::to_string(ts) + suffix;
        path = (dir / name).string();
    }

    ~temp_file() {
        if (!path.empty()) {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    }

    temp_file(const temp_file &) = delete;
    temp_file & operator=(const temp_file &) = delete;
};

// ────────────────────────────────────────────────────────────────────
// Minimal JSON string extraction (avoids pulling a full JSON parser)
// ────────────────────────────────────────────────────────────────────

std::string json_get_string(const std::string & body, const std::string & key) {
    const std::string needle = "\"" + key + "\"";
    auto kpos = body.find(needle);
    if (kpos == std::string::npos) {
        return "";
    }
    auto colon = body.find(':', kpos + needle.size());
    if (colon == std::string::npos) {
        return "";
    }
    auto q1 = body.find('"', colon + 1);
    if (q1 == std::string::npos) {
        return "";
    }
    std::string result;
    for (size_t i = q1 + 1; i < body.size(); ++i) {
        if (body[i] == '\\' && i + 1 < body.size()) {
            result += body[++i];
        } else if (body[i] == '"') {
            break;
        } else {
            result += body[i];
        }
    }
    return result;
}

// ────────────────────────────────────────────────────────────────────
// Server parameters
// ────────────────────────────────────────────────────────────────────

struct server_params {
    std::string model;
    std::string host      = "0.0.0.0";
    int         port      = 8090;
    int         threads   = 4;
    int         max_tokens = 4096;
    voxtral_gpu_backend gpu       = voxtral_gpu_backend::auto_detect;
    voxtral_log_level   log_level = voxtral_log_level::info;
};

void print_usage(const char * prog) {
    std::cout
        << "voxtral-server -- HTTP transcription server (OpenAI Whisper-compatible)\n\n"
        << "usage: " << prog << " --model path.gguf [options]\n\n"
        << "options:\n"
        << "  --model PATH        GGUF model path (required)\n"
        << "  --host HOST         listen address   (default: 0.0.0.0)\n"
        << "  --port PORT         listen port      (default: 8090)\n"
        << "  --threads N         inference threads (default: 4)\n"
        << "  --max-tokens N      max decode tokens (default: 4096)\n"
        << "  --gpu BACKEND       auto|cuda|metal|vulkan|none (default: auto)\n"
        << "  --log-level LEVEL   error|warn|info|debug       (default: info)\n"
        << "  -h, --help          show this help\n\n"
        << "endpoints:\n"
        << "  GET  /health                       → {\"status\":\"ok\"}\n"
        << "  GET  /v1/models                    → model list\n"
        << "  POST /v1/audio/transcriptions      → OpenAI Whisper-compatible\n"
        << "       accepts multipart file OR JSON {\"audio_base64\":\"...\"}\n";
}

bool parse_i32(const std::string & s, int & out) {
    char * end = nullptr;
    long v = std::strtol(s.c_str(), &end, 10);
    if (!end || *end != '\0') {
        return false;
    }
    out = static_cast<int>(v);
    return true;
}

bool parse_gpu(const std::string & s, voxtral_gpu_backend & out) {
    std::string lc = s;
    std::transform(lc.begin(), lc.end(), lc.begin(), ::tolower);
    if (lc == "none")   { out = voxtral_gpu_backend::none;        return true; }
    if (lc == "auto")   { out = voxtral_gpu_backend::auto_detect; return true; }
    if (lc == "cuda")   { out = voxtral_gpu_backend::cuda;        return true; }
    if (lc == "metal")  { out = voxtral_gpu_backend::metal;       return true; }
    if (lc == "vulkan") { out = voxtral_gpu_backend::vulkan;      return true; }
    return false;
}

bool parse_log_level(const std::string & s, voxtral_log_level & out) {
    if (s == "error") { out = voxtral_log_level::error; return true; }
    if (s == "warn")  { out = voxtral_log_level::warn;  return true; }
    if (s == "info")  { out = voxtral_log_level::info;  return true; }
    if (s == "debug") { out = voxtral_log_level::debug; return true; }
    return false;
}

bool parse_args(int argc, char ** argv, server_params & p) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];

        auto need_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                return nullptr;
            }
            return argv[++i];
        };

        if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (a == "--model") {
            const char * v = need_value("--model");
            if (!v) { return false; }
            p.model = v;
        } else if (a == "--host") {
            const char * v = need_value("--host");
            if (!v) { return false; }
            p.host = v;
        } else if (a == "--port") {
            const char * v = need_value("--port");
            if (!v || !parse_i32(v, p.port)) {
                std::cerr << "invalid --port\n";
                return false;
            }
        } else if (a == "--threads") {
            const char * v = need_value("--threads");
            if (!v || !parse_i32(v, p.threads)) {
                std::cerr << "invalid --threads\n";
                return false;
            }
        } else if (a == "--max-tokens") {
            const char * v = need_value("--max-tokens");
            if (!v || !parse_i32(v, p.max_tokens)) {
                std::cerr << "invalid --max-tokens\n";
                return false;
            }
        } else if (a == "--gpu") {
            const char * v = need_value("--gpu");
            if (!v || !parse_gpu(v, p.gpu)) {
                std::cerr << "invalid --gpu (expected: auto|cuda|metal|vulkan|none)\n";
                return false;
            }
        } else if (a == "--log-level") {
            const char * v = need_value("--log-level");
            if (!v || !parse_log_level(v, p.log_level)) {
                std::cerr << "invalid --log-level\n";
                return false;
            }
        } else {
            std::cerr << "unknown option: " << a << "\n";
            return false;
        }
    }

    if (p.model.empty()) {
        std::cerr << "--model is required\n";
        return false;
    }
    return true;
}

// ────────────────────────────────────────────────────────────────────
// Global server pointer for signal handling
// ────────────────────────────────────────────────────────────────────

httplib::Server * g_svr = nullptr;

void on_signal(int) {
    if (g_svr) {
        g_svr->stop();
    }
}

const char * gpu_name(voxtral_gpu_backend gpu) {
    switch (gpu) {
        case voxtral_gpu_backend::none:        return "none";
        case voxtral_gpu_backend::auto_detect: return "auto";
        case voxtral_gpu_backend::cuda:        return "cuda";
        case voxtral_gpu_backend::metal:       return "metal";
        case voxtral_gpu_backend::vulkan:      return "vulkan";
        default:                               return "unknown";
    }
}

} // anonymous namespace

// ════════════════════════════════════════════════════════════════════
// main
// ════════════════════════════════════════════════════════════════════

int main(int argc, char ** argv) {
    server_params params;
    if (!parse_args(argc, argv, params)) {
        print_usage(argv[0]);
        return 1;
    }

    // ── Logger ──────────────────────────────────────────────────────

    voxtral_log_callback logger = [level = params.log_level](
            voxtral_log_level msg_level, const std::string & msg) {
        if (static_cast<int>(msg_level) > static_cast<int>(level)) {
            return;
        }
        const char * tag = "I";
        if      (msg_level == voxtral_log_level::error) { tag = "E"; }
        else if (msg_level == voxtral_log_level::warn)  { tag = "W"; }
        else if (msg_level == voxtral_log_level::debug) { tag = "D"; }
        std::cerr << "voxtral_" << tag << ": " << msg << "\n";
    };

    // ── Load model ──────────────────────────────────────────────────

    std::cerr << "voxtral-server: loading model " << params.model
              << "  (gpu=" << gpu_name(params.gpu)
              << ", threads=" << params.threads << ")\n";

    auto t_load = std::chrono::steady_clock::now();

    voxtral_model * model = voxtral_model_load_from_file(
        params.model, logger, params.gpu);
    if (!model) {
        std::cerr << "voxtral-server: failed to load model\n";
        return 2;
    }

    voxtral_context_params ctx_p;
    ctx_p.n_threads = params.threads;
    ctx_p.log_level = params.log_level;
    ctx_p.logger    = logger;
    ctx_p.gpu       = params.gpu;

    voxtral_context * ctx = voxtral_init_from_model(model, ctx_p);
    if (!ctx) {
        std::cerr << "voxtral-server: failed to create context\n";
        voxtral_model_free(model);
        return 3;
    }

    const double load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_load).count();
    std::cerr << std::fixed << std::setprecision(0)
              << "voxtral-server: model loaded in " << load_ms << " ms\n";

    // Inference is single-threaded; serialize with a mutex.
    std::mutex inference_mtx;
    std::atomic<uint64_t> n_requests{0};

    // ── HTTP server ─────────────────────────────────────────────────

    httplib::Server svr;
    g_svr = &svr;
    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);

    svr.set_payload_max_length(128 * 1024 * 1024);

    // CORS pre-flight for browser clients
    svr.Options(".*", [](const httplib::Request &, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        res.status = 204;
    });

    // ── GET /health ─────────────────────────────────────────────────

    svr.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    // ── GET /v1/models ──────────────────────────────────────────────

    svr.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(
            "{\"data\":[{\"id\":\"voxtral\",\"object\":\"model\","
            "\"owned_by\":\"local\"}]}",
            "application/json");
    });

    // ── POST /v1/audio/transcriptions ───────────────────────────────

    svr.Post("/v1/audio/transcriptions",
        [&](const httplib::Request & req, httplib::Response & res) {

        res.set_header("Access-Control-Allow-Origin", "*");
        const uint64_t req_id = n_requests.fetch_add(1) + 1;
        const auto t0 = std::chrono::steady_clock::now();

        // -- Determine desired response format --------------------------------
        std::string response_format = "json";
        if (req.has_file("response_format")) {
            response_format = req.get_file_value("response_format").content;
        }
        if (req.has_param("response_format")) {
            response_format = req.get_param_value("response_format");
        }

        // -- Extract audio data into a temp file ------------------------------
        temp_file tmp;
        bool have_audio = false;

        // Method A: multipart file upload (OpenAI Whisper-compatible)
        if (req.has_file("file")) {
            const auto & file = req.get_file_value("file");
            std::ofstream ofs(tmp.path, std::ios::binary);
            if (!ofs) {
                res.status = 500;
                res.set_content(
                    make_error_json(500, "Failed to write temp file", "server_error"),
                    "application/json");
                return;
            }
            ofs.write(file.content.data(),
                      static_cast<std::streamsize>(file.content.size()));
            ofs.close();
            have_audio = true;
        }

        // Method B: JSON body with base64-encoded audio
        if (!have_audio && !req.body.empty()) {
            std::string ct;
            if (req.has_header("Content-Type")) {
                ct = req.get_header_value("Content-Type");
            }
            if (ct.find("application/json") != std::string::npos) {
                std::string b64 = json_get_string(req.body, "audio_base64");
                if (b64.empty()) {
                    b64 = json_get_string(req.body, "file");
                }
                std::string rf = json_get_string(req.body, "response_format");
                if (!rf.empty()) {
                    response_format = rf;
                }

                if (!b64.empty()) {
                    auto decoded = base64_decode(b64);
                    if (decoded.empty()) {
                        res.status = 400;
                        res.set_content(
                            make_error_json(400, "Invalid base64 audio data"),
                            "application/json");
                        return;
                    }
                    std::ofstream ofs(tmp.path, std::ios::binary);
                    if (!ofs) {
                        res.status = 500;
                        res.set_content(
                            make_error_json(500, "Failed to write temp file",
                                            "server_error"),
                            "application/json");
                        return;
                    }
                    ofs.write(reinterpret_cast<const char *>(decoded.data()),
                              static_cast<std::streamsize>(decoded.size()));
                    ofs.close();
                    have_audio = true;
                }
            }
        }

        if (!have_audio) {
            res.status = 400;
            res.set_content(
                make_error_json(400,
                    "No audio provided. Send multipart field 'file' "
                    "or JSON {\"audio_base64\":\"...\"}"),
                "application/json");
            return;
        }

        // -- Transcribe (serialized) ------------------------------------------
        voxtral_result result;
        bool ok;
        {
            std::lock_guard<std::mutex> lock(inference_mtx);
            ok = voxtral_transcribe_file(
                *ctx, tmp.path, params.max_tokens, result);
        }

        const double elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        const double elapsed_s = elapsed_ms / 1000.0;

        if (!ok) {
            res.status = 500;
            res.set_content(
                make_error_json(500, "Transcription failed", "server_error"),
                "application/json");
            std::cerr << "[req " << req_id << "] FAILED  "
                      << std::fixed << std::setprecision(0)
                      << elapsed_ms << " ms\n";
            return;
        }

        const std::string & text = result.text;
        std::cerr << "[req " << req_id << "] OK  "
                  << std::fixed << std::setprecision(0) << elapsed_ms << " ms  "
                  << (text.size() > 80 ? text.substr(0, 80) + "..." : text)
                  << "\n";

        // -- Send response ----------------------------------------------------
        if (response_format == "text") {
            res.set_content(text, "text/plain; charset=utf-8");
        } else {
            res.set_content(make_result_json(text, elapsed_s),
                            "application/json");
        }
    });

    // ── Start listening ─────────────────────────────────────────────

    std::cerr << "\nvoxtral-server ready\n"
              << "  http://" << params.host << ":" << params.port << "/health\n"
              << "  http://" << params.host << ":" << params.port
              << "/v1/audio/transcriptions\n\n";

    if (!svr.listen(params.host, params.port)) {
        std::cerr << "voxtral-server: failed to bind "
                  << params.host << ":" << params.port << "\n";
        voxtral_free(ctx);
        voxtral_model_free(model);
        return 4;
    }

    std::cerr << "voxtral-server: shutting down\n";
    voxtral_free(ctx);
    voxtral_model_free(model);
    return 0;
}

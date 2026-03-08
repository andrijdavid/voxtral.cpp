#include "voxtral.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct cli_params {
    std::string model;
    std::string audio;
    std::string prompt;
    std::string dump_logits;
    std::string dump_logits_bin;
    std::string dump_tokens;
    std::string output_text;
    int32_t threads = 0;
    uint32_t seed = 0;
    int32_t max_tokens = 256;
    voxtral_log_level log_level = voxtral_log_level::info;
    voxtral_gpu_backend gpu = voxtral_gpu_backend::none;
    bool stdin_mode = false;
};

struct backend_reg_info {
    std::string name;
    size_t devices = 0;
};

std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

bool has_runtime_backend(const std::vector<backend_reg_info> & regs, const std::string & name) {
    const std::string name_lc = to_lower_copy(name);
    for (const auto & reg : regs) {
        if (to_lower_copy(reg.name) == name_lc) {
            return reg.devices > 0;
        }
    }
    return false;
}

std::vector<backend_reg_info> collect_runtime_backends() {
    std::vector<backend_reg_info> regs;
    const size_t n_reg = ggml_backend_reg_count();
    regs.reserve(n_reg);

    for (size_t i = 0; i < n_reg; ++i) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        if (!reg) {
            continue;
        }

        const char * name = ggml_backend_reg_name(reg);
        regs.push_back({
            name ? name : "unknown",
            ggml_backend_reg_dev_count(reg),
        });
    }

    return regs;
}

std::string format_build_backend_flags() {
    std::ostringstream os;
    bool first = true;
    auto append_flag = [&](const char * name, bool enabled) {
        if (!first) {
            os << ", ";
        }
        os << name << '=' << (enabled ? "ON" : "OFF");
        first = false;
    };

#ifdef GGML_USE_CPU
    append_flag("GGML_USE_CPU", true);
#else
    append_flag("GGML_USE_CPU", false);
#endif
#ifdef GGML_USE_BLAS
    append_flag("GGML_USE_BLAS", true);
#else
    append_flag("GGML_USE_BLAS", false);
#endif
#ifdef GGML_USE_METAL
    append_flag("GGML_USE_METAL", true);
#else
    append_flag("GGML_USE_METAL", false);
#endif
#ifdef GGML_USE_CUDA
    append_flag("GGML_USE_CUDA", true);
#else
    append_flag("GGML_USE_CUDA", false);
#endif
#ifdef GGML_USE_VULKAN
    append_flag("GGML_USE_VULKAN", true);
#else
    append_flag("GGML_USE_VULKAN", false);
#endif
#ifdef GGML_USE_SYCL
    append_flag("GGML_USE_SYCL", true);
#else
    append_flag("GGML_USE_SYCL", false);
#endif
#ifdef GGML_USE_OPENCL
    append_flag("GGML_USE_OPENCL", true);
#else
    append_flag("GGML_USE_OPENCL", false);
#endif
#ifdef GGML_USE_CANN
    append_flag("GGML_USE_CANN", true);
#else
    append_flag("GGML_USE_CANN", false);
#endif
#ifdef GGML_USE_WEBGPU
    append_flag("GGML_USE_WEBGPU", true);
#else
    append_flag("GGML_USE_WEBGPU", false);
#endif
#ifdef GGML_USE_RPC
    append_flag("GGML_USE_RPC", true);
#else
    append_flag("GGML_USE_RPC", false);
#endif
#ifdef GGML_USE_ZDNN
    append_flag("GGML_USE_ZDNN", true);
#else
    append_flag("GGML_USE_ZDNN", false);
#endif
#ifdef GGML_USE_ZENDNN
    append_flag("GGML_USE_ZENDNN", true);
#else
    append_flag("GGML_USE_ZENDNN", false);
#endif
#ifdef GGML_USE_HEXAGON
    append_flag("GGML_USE_HEXAGON", true);
#else
    append_flag("GGML_USE_HEXAGON", false);
#endif
#ifdef GGML_USE_VIRTGPU_FRONTEND
    append_flag("GGML_USE_VIRTGPU_FRONTEND", true);
#else
    append_flag("GGML_USE_VIRTGPU_FRONTEND", false);
#endif

    return os.str();
}

std::string format_runtime_backends(const std::vector<backend_reg_info> & regs) {
    if (regs.empty()) {
        return "none";
    }

    std::ostringstream os;
    for (size_t i = 0; i < regs.size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << regs[i].name << '(' << regs[i].devices << " dev";
        if (regs[i].devices != 1) {
            os << 's';
        }
        os << ')';
    }
    return os.str();
}

std::string format_runtime_availability(const std::vector<backend_reg_info> & regs) {
    static const std::array<const char *, 14> known_backends = {
        "CPU", "BLAS", "METAL", "CUDA", "VULKAN", "SYCL", "OPENCL",
        "CANN", "WEBGPU", "RPC", "ZDNN", "ZENDNN", "HEXAGON", "VIRTGPU",
    };

    std::ostringstream os;
    for (size_t i = 0; i < known_backends.size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << known_backends[i] << '='
           << (has_runtime_backend(regs, known_backends[i]) ? "yes" : "no");
    }
    return os.str();
}

void print_usage(const char * argv0) {
    std::cout
        << "usage: " << argv0 << " --model path.gguf --audio file.wav [options]\n"
        << "\n"
        << "options:\n"
        << "  --model PATH          GGUF model path\n"
        << "  --audio PATH          input wav (mono or stereo PCM16/float32)\n"
        << "  --threads N           cpu threads\n"
        << "  --seed N              rng seed (reserved for sampling)\n"
        << "  --prompt TEXT         prompt text (compatibility, currently ignored for realtime mode)\n"
        << "  --n-tokens N          max decode tokens (alias of --max-len)\n"
        << "  --max-len N           max decode tokens\n"
        << "  --verbose             equivalent to --log-level debug\n"
        << "  --log-level LEVEL     error|warn|info|debug\n"
        << "  --dump-logits PATH    write step-0 logits (first 32 values) as text\n"
        << "  --dump-logits-bin P   write full step-0 logits as float32 raw bytes\n"
        << "  --dump-tokens PATH    write generated token ids as a single line\n"
        << "  --output-text PATH    write decoded text to file (still prints to stdout)\n"
        << "  --gpu BACKEND         gpu backend: auto|cuda|metal|vulkan|none (default: none)\n"
        << "  --metal               alias for --gpu metal\n"
        << "  --stdin               interactive mode: read audio paths from stdin (one per line),\n"
        << "                        keeps model loaded between transcriptions.\n"
        << "                        Output ends with __VOXTRAL_END__ sentinel per file.\n"
        << "  -h, --help            show this help\n";
}

bool parse_i32(const std::string & s, int32_t & out) {
    char * end = nullptr;
    const long v = std::strtol(s.c_str(), &end, 10);
    if (!end || *end != '\0') {
        return false;
    }
    out = static_cast<int32_t>(v);
    return true;
}

bool parse_u32(const std::string & s, uint32_t & out) {
    char * end = nullptr;
    const unsigned long v = std::strtoul(s.c_str(), &end, 10);
    if (!end || *end != '\0') {
        return false;
    }
    out = static_cast<uint32_t>(v);
    return true;
}

bool parse_gpu(const std::string & s, voxtral_gpu_backend & out) {
    const std::string lc = to_lower_copy(s);
    if (lc == "none")   { out = voxtral_gpu_backend::none;        return true; }
    if (lc == "auto")   { out = voxtral_gpu_backend::auto_detect; return true; }
    if (lc == "cuda")   { out = voxtral_gpu_backend::cuda;        return true; }
    if (lc == "metal")  { out = voxtral_gpu_backend::metal;       return true; }
    if (lc == "vulkan") { out = voxtral_gpu_backend::vulkan;      return true; }
    return false;
}

bool parse_level(const std::string & s, voxtral_log_level & out) {
    if (s == "error") {
        out = voxtral_log_level::error;
        return true;
    }
    if (s == "warn") {
        out = voxtral_log_level::warn;
        return true;
    }
    if (s == "info") {
        out = voxtral_log_level::info;
        return true;
    }
    if (s == "debug") {
        out = voxtral_log_level::debug;
        return true;
    }
    return false;
}

bool parse_args(int argc, char ** argv, cli_params & p) {
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
            if (!v) {
                return false;
            }
            p.model = v;
        } else if (a == "--audio") {
            const char * v = need_value("--audio");
            if (!v) {
                return false;
            }
            p.audio = v;
        } else if (a == "--threads") {
            const char * v = need_value("--threads");
            if (!v || !parse_i32(v, p.threads)) {
                std::cerr << "invalid --threads\n";
                return false;
            }
        } else if (a == "--seed") {
            const char * v = need_value("--seed");
            if (!v || !parse_u32(v, p.seed)) {
                std::cerr << "invalid --seed\n";
                return false;
            }
        } else if (a == "--prompt") {
            const char * v = need_value("--prompt");
            if (!v) {
                return false;
            }
            p.prompt = v;
        } else if (a == "--n-tokens" || a == "--max-len") {
            const char * v = need_value(a.c_str());
            if (!v || !parse_i32(v, p.max_tokens)) {
                std::cerr << "invalid " << a << "\n";
                return false;
            }
        } else if (a == "--verbose") {
            p.log_level = voxtral_log_level::debug;
        } else if (a == "--log-level") {
            const char * v = need_value("--log-level");
            if (!v || !parse_level(v, p.log_level)) {
                std::cerr << "invalid --log-level\n";
                return false;
            }
        } else if (a == "--dump-logits") {
            const char * v = need_value("--dump-logits");
            if (!v) {
                return false;
            }
            p.dump_logits = v;
        } else if (a == "--dump-logits-bin") {
            const char * v = need_value("--dump-logits-bin");
            if (!v) {
                return false;
            }
            p.dump_logits_bin = v;
        } else if (a == "--dump-tokens") {
            const char * v = need_value("--dump-tokens");
            if (!v) {
                return false;
            }
            p.dump_tokens = v;
        } else if (a == "--output-text") {
            const char * v = need_value("--output-text");
            if (!v) {
                return false;
            }
            p.output_text = v;
        } else if (a == "--gpu") {
            const char * v = need_value("--gpu");
            if (!v || !parse_gpu(v, p.gpu)) {
                std::cerr << "invalid --gpu (expected: auto|cuda|metal|vulkan|none)\n";
                return false;
            }
        } else if (a == "--metal") {
            p.gpu = voxtral_gpu_backend::metal;
        } else if (a == "--stdin") {
            p.stdin_mode = true;
        } else {
            std::cerr << "unknown option: " << a << "\n";
            return false;
        }
    }

    if (p.model.empty()) {
        std::cerr << "--model is required\n";
        return false;
    }

    if (p.audio.empty() && !p.stdin_mode) {
        std::cerr << "--audio is required (or use --stdin for interactive mode)\n";
        return false;
    }

    if (p.max_tokens <= 0) {
        p.max_tokens = 1;
    }

    return true;
}

} // namespace

int main(int argc, char ** argv) {
    cli_params p;
    if (!parse_args(argc, argv, p)) {
        print_usage(argv[0]);
        return 1;
    }

    const auto t_start = std::chrono::steady_clock::now();
    auto finish = [&](int exit_code) {
        const auto elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_start).count();
        const std::vector<backend_reg_info> runtime_backends = collect_runtime_backends();

        std::cerr << std::fixed << std::setprecision(2);
        std::cerr << "[summary] processing_time_ms=" << elapsed_ms << "\n";
        std::cerr << "[summary] build_backend_flags: " << format_build_backend_flags() << "\n";
        std::cerr << "[summary] runtime_backends: " << format_runtime_backends(runtime_backends) << "\n";
        std::cerr << "[summary] runtime_available: " << format_runtime_availability(runtime_backends) << "\n";
        return exit_code;
    };

    voxtral_log_callback logger = [level = p.log_level](voxtral_log_level msg_level, const std::string & msg) {
        if (static_cast<int>(msg_level) > static_cast<int>(level)) {
            return;
        }

        const char * tag = "I";
        if (msg_level == voxtral_log_level::error) {
            tag = "E";
        } else if (msg_level == voxtral_log_level::warn) {
            tag = "W";
        } else if (msg_level == voxtral_log_level::debug) {
            tag = "D";
        }

        std::cerr << "voxtral_" << tag << ": " << msg << "\n";
    };

    voxtral_model * model = voxtral_model_load_from_file(p.model, logger, p.gpu);
    if (!model) {
        return finish(2);
    }

    voxtral_context_params ctx_p;
    ctx_p.n_threads = p.threads;
    // ctx_p.seed = p.seed;
    ctx_p.log_level = p.log_level;
    ctx_p.logger = logger;
    ctx_p.gpu = p.gpu;

    voxtral_context * ctx = voxtral_init_from_model(model, ctx_p);
    if (!ctx) {
        voxtral_model_free(model);
        return finish(3);
    }

    if (p.stdin_mode) {
        std::cerr << "voxtral: stdin mode ready, waiting for audio paths...\n";
        std::cout << "__VOXTRAL_READY__" << std::endl;

        std::string line;
        while (std::getline(std::cin, line)) {
            // Trim whitespace
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ')) {
                line.pop_back();
            }
            if (line.empty()) {
                continue;
            }

            const auto t_req = std::chrono::steady_clock::now();

            voxtral_result result;
            if (!voxtral_transcribe_file(*ctx, line, p.max_tokens, result)) {
                std::cout << "[error] transcription failed for: " << line << "\n";
            } else {
                const std::string text = result.text.empty() ? std::string("[no-transcript]") : result.text;
                std::cout << text << "\n";
            }

            const double req_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_req).count();
            std::cerr << std::fixed << std::setprecision(2)
                      << "[stdin] transcribed " << line << " in " << req_ms << " ms\n";

            std::cout << "__VOXTRAL_END__" << std::endl;
        }

        voxtral_free(ctx);
        voxtral_model_free(model);
        return finish(0);
    }

    voxtral_result result;
    if (!voxtral_transcribe_file(*ctx, p.audio, p.max_tokens, result)) {
        voxtral_free(ctx);
        voxtral_model_free(model);
        return finish(4);
    }

    const std::string printed_text = result.text.empty() ? std::string("[no-transcript]") : result.text;
    std::cout << printed_text << "\n";

    if (!result.tokens.empty()) {
        std::ostringstream os;
        os << "[tokens]";
        for (size_t i = 0; i < result.tokens.size(); ++i) {
            os << (i == 0 ? " " : " ") << result.tokens[i];
        }
        std::cout << os.str() << "\n";
    } else {
        std::cout << "[no-transcript]\n";
    }

    if (!p.output_text.empty()) {
        std::ofstream fout(p.output_text);
        if (!fout) {
            std::cerr << "failed to open --output-text output\n";
        } else {
            fout << printed_text << "\n";
        }
    }

    if (!p.dump_logits.empty()) {
        std::ofstream fout(p.dump_logits);
        if (!fout) {
            std::cerr << "failed to open --dump-logits output\n";
        } else {
            const size_t n = std::min<size_t>(32, result.first_step_logits.size());
            for (size_t i = 0; i < n; ++i) {
                fout << std::setprecision(9) << result.first_step_logits[i];
                if (i + 1 < n) {
                    fout << "\n";
                }
            }
        }
    }

    if (!p.dump_logits_bin.empty()) {
        std::ofstream fout(p.dump_logits_bin, std::ios::binary);
        if (!fout) {
            std::cerr << "failed to open --dump-logits-bin output\n";
        } else if (!result.first_step_logits.empty()) {
            fout.write(
                reinterpret_cast<const char *>(result.first_step_logits.data()),
                static_cast<std::streamsize>(result.first_step_logits.size() * sizeof(float)));
        }
    }

    if (!p.dump_tokens.empty()) {
        std::ofstream fout(p.dump_tokens);
        if (!fout) {
            std::cerr << "failed to open --dump-tokens output\n";
        } else {
            for (size_t i = 0; i < result.tokens.size(); ++i) {
                if (i > 0) {
                    fout << ' ';
                }
                fout << result.tokens[i];
            }
            fout << "\n";
        }
    }

    voxtral_free(ctx);
    voxtral_model_free(model);
    return finish(0);
}

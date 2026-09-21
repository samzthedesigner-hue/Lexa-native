#include <jni.h>
#include <android/log.h>
#include <string>
#include <mutex>
#include <thread>
#include <vector>
#include <algorithm>
#include "llama.h"

#define LOG_TAG "LexaJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static llama_model*        g_model = nullptr;
static llama_context*      g_ctx   = nullptr;
static const llama_vocab*  g_vocab = nullptr;
static std::mutex          g_mutex;
static int                 g_n_ctx = 2048;

static const int N_CTX_SMALL  = 2048;
static const int N_CTX_MEDIUM = 1024;
static const int N_CTX_LARGE  = 768;
static const int N_BATCH      = 256;
static const int MAX_TOKENS   = 512;

static const float TEMPERATURE    = 0.70f;
static const int   TOP_K          = 40;
static const float TOP_P          = 0.90f;
static const float MIN_P          = 0.05f;
static const float REPEAT_PENALTY = 1.15f;
static const int   REPEAT_LAST_N  = 128;

static const std::vector<std::string> STOP_STRINGS = {
    "\nUser:", "\n###", "</s>", "<|im_end|>", "<|eot_id|>", "<|endoftext|>"
};

static void log_cb(ggml_log_level level, const char* text, void*) {
    if (!text) return;
    if (level == GGML_LOG_LEVEL_ERROR) LOGE("%s", text);
    else LOGI("%s", text);
}

static void free_model_locked() {
    if (g_ctx)   { llama_free(g_ctx);         g_ctx   = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    g_vocab = nullptr;
}

static std::string trim_stops(const std::string& s) {
    size_t earliest = std::string::npos;
    for (const auto& stop : STOP_STRINGS) {
        size_t p = s.find(stop);
        if (p != std::string::npos && p < earliest) earliest = p;
    }
    if (earliest == std::string::npos) return s;
    return s.substr(0, earliest);
}

static int pick_ctx(int params_millions) {
    if (params_millions <= 500) return N_CTX_SMALL;
    if (params_millions <= 1200) return N_CTX_MEDIUM;
    return N_CTX_LARGE;
}

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM*, void*) {
    llama_log_set(log_cb, nullptr);
    llama_backend_init();
    LOGI("llama backend initialized");
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT void JNICALL JNI_OnUnload(JavaVM*, void*) {
    std::lock_guard<std::mutex> lk(g_mutex);
    free_model_locked();
    llama_backend_free();
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_press_ai_LlamaBridge_nativeLoadModel(JNIEnv* env, jobject, jstring jpath, jint params_millions) {
    if (!jpath) return JNI_FALSE;
    const char* cpath = env->GetStringUTFChars(jpath, nullptr);
    std::string path = cpath ? cpath : "";
    env->ReleaseStringUTFChars(jpath, cpath);

    std::lock_guard<std::mutex> lk(g_mutex);
    free_model_locked();

    g_n_ctx = pick_ctx(params_millions);

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_mmap     = true;
    mp.use_mlock    = false;

    g_model = llama_model_load_from_file(path.c_str(), mp);
    if (!g_model) { LOGE("load failed: %s", path.c_str()); return JNI_FALSE; }

    g_vocab = llama_model_get_vocab(g_model);

    int threads = std::max(2, std::min(4, (int)std::thread::hardware_concurrency() - 1));

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = g_n_ctx;
    cp.n_batch         = N_BATCH;
    cp.n_threads       = threads;
    cp.n_threads_batch = threads;

    g_ctx = llama_init_from_model(g_model, cp);
    if (!g_ctx) { LOGE("ctx init failed"); free_model_locked(); return JNI_FALSE; }

    LOGI("model loaded ok: %s ctx=%d threads=%d", path.c_str(), g_n_ctx, threads);
    return JNI_TRUE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_press_ai_LlamaBridge_nativeGenerate(JNIEnv* env, jobject, jstring jprompt) {
    if (!jprompt) return env->NewStringUTF("");

    const char* cprompt = env->GetStringUTFChars(jprompt, nullptr);
    std::string prompt = cprompt ? cprompt : "";
    env->ReleaseStringUTFChars(jprompt, cprompt);

    std::lock_guard<std::mutex> lk(g_mutex);
    if (!g_model || !g_ctx || !g_vocab) return env->NewStringUTF("ERROR: no model loaded");

    const int n_prompt = -llama_tokenize(
        g_vocab, prompt.c_str(), (int)prompt.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) return env->NewStringUTF("ERROR: tokenize failed");

    std::vector<llama_token> tokens(n_prompt);
    int n = llama_tokenize(
        g_vocab, prompt.c_str(), (int)prompt.size(),
        tokens.data(), (int)tokens.size(), true, true);
    if (n <= 0) return env->NewStringUTF("ERROR: tokenize failed");

    llama_memory_clear(llama_get_memory(g_ctx), true);

    if ((int)tokens.size() > g_n_ctx - 128) {
        int keep = g_n_ctx - 128;
        tokens.erase(tokens.begin(), tokens.begin() + (tokens.size() - keep));
    }

    llama_batch batch = llama_batch_get_one(tokens.data(), (int)tokens.size());
    if (llama_decode(g_ctx, batch) != 0) return env->NewStringUTF("ERROR: prefill decode failed");

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(TOP_K));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(TOP_P, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(MIN_P, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(TEMPERATURE));
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
        REPEAT_LAST_N, REPEAT_PENALTY, 0.0f, 0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    std::string out;
    const llama_token eos = llama_vocab_eos(g_vocab);
    const llama_token eot = llama_vocab_eot(g_vocab);

    for (int i = 0; i < MAX_TOKENS; ++i) {
        llama_token tok = llama_sampler_sample(smpl, g_ctx, -1);
        if (tok == eos || tok == eot) break;

        char buf[256];
        int w = llama_token_to_piece(g_vocab, tok, buf, sizeof(buf), 0, true);
        if (w > 0) out.append(buf, w);

        std::string trimmed = trim_stops(out);
        if (trimmed.size() != out.size()) { out = trimmed; break; }

        if (out.size() > 32000) break;

        llama_batch b = llama_batch_get_one(&tok, 1);
        if (llama_decode(g_ctx, b) != 0) break;
    }

    llama_sampler_free(smpl);
    return env->NewStringUTF(out.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_press_ai_LlamaBridge_nativeUnload(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lk(g_mutex);
    free_model_locked();
    LOGI("model unloaded");
}

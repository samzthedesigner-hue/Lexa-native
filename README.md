# lexa-native

Native runtime (`libllama.so`) for the Lexa Android app.

- Manual build via GitHub Actions → Actions tab → **Build libllama.so** → Run workflow
- Target: `arm64-v8a`, Android API 24+
- Sampling locked: temp 0.70, top_k 40, top_p 0.90, min_p 0.05, repeat_penalty 1.15
- Stop strings: `\nUser:`, `\n###`, `</s>`, `<|im_end|>`, `<|eot_id|>`, `<|endoftext|>`

## Consume

- https://github.com/<USER>/lexa-native/releases/latest/download/libllama.so
- https://github.com/<USER>/lexa-native/releases/latest/download/libllama.so.sha256

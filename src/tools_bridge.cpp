#include <jni.h>
#include <android/log.h>
#include <string>
#include <cstring>
#include "third_party/miniz/miniz.h"

#define LOG_TAG "LexaTools"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" JNIEXPORT jboolean JNICALL
Java_com_press_ai_ToolsBridge_nativeZipCreate(JNIEnv* env, jobject, jstring jzipPath) {
    const char* zipPath = env->GetStringUTFChars(jzipPath, nullptr);

    mz_zip_archive zip;
    memset(&zip, 0, sizeof(zip));
    mz_bool ok = mz_zip_writer_init_file(&zip, zipPath, 0);
    if (ok) {
        mz_zip_writer_finalize_archive(&zip);
        mz_zip_writer_end(&zip);
    } else {
        LOGE("zip create failed: %s", zipPath);
    }

    env->ReleaseStringUTFChars(jzipPath, zipPath);
    return ok ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_press_ai_ToolsBridge_nativeZipAddFile(JNIEnv* env, jobject, jstring jzipPath,
                                                jstring jentryName, jstring jcontent) {
    const char* zipPath = env->GetStringUTFChars(jzipPath, nullptr);
    const char* entryName = env->GetStringUTFChars(jentryName, nullptr);
    const char* content = env->GetStringUTFChars(jcontent, nullptr);
    jsize contentLen = env->GetStringUTFLength(jcontent);

    mz_bool ok = mz_zip_add_mem_to_archive_file_in_place(
        zipPath, entryName, content, (size_t) contentLen, nullptr, 0, MZ_DEFAULT_COMPRESSION);

    if (!ok) LOGE("zip add failed: %s -> %s", entryName, zipPath);

    env->ReleaseStringUTFChars(jzipPath, zipPath);
    env->ReleaseStringUTFChars(jentryName, entryName);
    env->ReleaseStringUTFChars(jcontent, content);
    return ok ? JNI_TRUE : JNI_FALSE;
}

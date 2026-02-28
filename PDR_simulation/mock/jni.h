#pragma once
#include <cstdint>

// ============================================================
// Minimal JNI type stubs for desktop (simulator) compilation.
// These are NEVER called at runtime — they only satisfy the
// linker so that Android JNI bridge functions compile as-is.
// ============================================================

// Primitive types
typedef int32_t  jint;
typedef int64_t  jlong;
typedef float    jfloat;
typedef double   jdouble;
typedef uint8_t  jboolean;
typedef int32_t  jsize;

// Reference types (opaque pointers)
typedef void*    jobject;
typedef void*    jclass;
typedef void*    jstring;
typedef void*    jfloatArray;
typedef void*    jarray;
typedef void*    jmethodID;
typedef void*    jfieldID;

// Constants
#define JNI_TRUE  1
#define JNI_FALSE 0
#define JNI_ABORT 2

// Export/calling-convention macros (no-op on desktop).
// Use #ifndef so the unity-build wrapper can pre-define
// JNIEXPORT as __attribute__((weak)) before including jni.h.
#ifndef JNIEXPORT
#define JNIEXPORT
#endif
#ifndef JNICALL
#define JNICALL
#endif

// ============================================================
// Stub JNIEnv — provides all methods used by PDR_EXT sources.
// Every method returns a safe default (0 / nullptr / empty).
// ============================================================
struct JNIEnv_ {
    // --- Array operations ---
    jsize GetArrayLength(jarray) { return 0; }
    void  GetFloatArrayRegion(jfloatArray, jsize, jsize, jfloat*) {}
    jfloatArray NewFloatArray(jsize) { return nullptr; }
    void  SetFloatArrayRegion(jfloatArray, jsize, jsize, const jfloat*) {}
    jfloat* GetFloatArrayElements(jfloatArray, jboolean*) { return nullptr; }
    void  ReleaseFloatArrayElements(jfloatArray, jfloat*, jint) {}

    // --- Class / Object operations ---
    jclass  FindClass(const char*) { return nullptr; }
    jclass  GetObjectClass(jobject) { return nullptr; }
    jobject NewObject(jclass, jmethodID, ...) { return nullptr; }
    jobject NewGlobalRef(jobject obj) { return obj; }
    void    DeleteLocalRef(jobject) {}

    // --- Method / Field IDs ---
    jmethodID GetMethodID(jclass, const char*, const char*) { return nullptr; }
    jfieldID  GetFieldID(jclass, const char*, const char*)  { return nullptr; }

    // --- Field accessors ---
    void    SetDoubleField(jobject, jfieldID, jdouble) {}
    void    SetIntField(jobject, jfieldID, jint) {}
    jlong   GetLongField(jobject, jfieldID) { return 0; }
    jobject GetObjectField(jobject, jfieldID) { return nullptr; }

    // --- Method invocation ---
    jint CallIntMethod(jobject, jmethodID, ...) { return 0; }

    // --- String operations ---
    const char* GetStringUTFChars(jstring, jboolean*) { return ""; }
    void ReleaseStringUTFChars(jstring, const char*) {}

    // --- Exception handling ---
    jboolean ExceptionCheck() { return JNI_FALSE; }
    void     ExceptionClear() {}
};

typedef JNIEnv_ JNIEnv;

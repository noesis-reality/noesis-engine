use jni::JNIEnv;
use jni::objects::{JClass, JString, JIntArray};
use jni::sys::{jlong, jintArray, jstring};
use std::ffi::{CString, CStr};
use std::ptr;

// FFI declarations to the existing Harmony C library
#[link(name = "openai_harmony")]
extern "C" {
    fn harmony_encoding_new() -> *mut std::ffi::c_void;
    fn harmony_encoding_free(wrapper: *mut std::ffi::c_void);
    fn harmony_encoding_encode_plain(
        wrapper: *const std::ffi::c_void,
        text: *const std::os::raw::c_char,
        tokens_out: *mut *mut u32,
        tokens_len: *mut usize,
    ) -> HarmonyResult;
    fn harmony_encoding_render_prompt(
        wrapper: *const std::ffi::c_void,
        system_msg: *const std::os::raw::c_char,
        user_msg: *const std::os::raw::c_char,
        assistant_prefix: *const std::os::raw::c_char,
        tokens_out: *mut *mut u32,
        tokens_len: *mut usize,
    ) -> HarmonyResult;
    fn harmony_encoding_decode(
        wrapper: *const std::ffi::c_void,
        tokens: *const u32,
        tokens_len: usize,
    ) -> *mut std::os::raw::c_char;
    fn harmony_encoding_stop_tokens(
        wrapper: *const std::ffi::c_void,
        tokens_out: *mut *mut u32,
        tokens_len: *mut usize,
    ) -> HarmonyResult;
    fn harmony_free_string(s: *mut std::os::raw::c_char);
    fn harmony_free_tokens(tokens: *mut u32, len: usize);
}

#[repr(C)]
struct HarmonyResult {
    success: bool,
    error_message: *mut std::os::raw::c_char,
}

/// Create a new Harmony encoder
#[no_mangle]
pub extern "system" fn Java_ai_noesisreality_harmony_HarmonyEngine_nativeCreateEncoder(
    _env: JNIEnv,
    _class: JClass,
) -> jlong {
    unsafe {
        let encoder = harmony_encoding_new();
        encoder as jlong
    }
}

/// Free a Harmony encoder
#[no_mangle]
pub extern "system" fn Java_ai_noesisreality_harmony_HarmonyEngine_nativeFreeEncoder(
    _env: JNIEnv,
    _class: JClass,
    encoder_ptr: jlong,
) {
    if encoder_ptr != 0 {
        unsafe {
            harmony_encoding_free(encoder_ptr as *mut std::ffi::c_void);
        }
    }
}

/// Encode plain text without Harmony formatting
#[no_mangle]
pub extern "system" fn Java_ai_noesisreality_harmony_HarmonyEngine_nativeEncodePlain(
    env: JNIEnv,
    _class: JClass,
    encoder_ptr: jlong,
    text: JString,
) -> jintArray {
    let encoder = encoder_ptr as *const std::ffi::c_void;
    if encoder.is_null() {
        return ptr::null_mut();
    }

    // Convert Java string to C string
    let text_str = match env.get_string(text) {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    
    let c_text = match CString::new(text_str.to_str().unwrap_or("")) {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    unsafe {
        let mut tokens_ptr: *mut u32 = ptr::null_mut();
        let mut tokens_len: usize = 0;

        let result = harmony_encoding_encode_plain(
            encoder,
            c_text.as_ptr(),
            &mut tokens_ptr,
            &mut tokens_len,
        );

        if !result.success {
            if !result.error_message.is_null() {
                harmony_free_string(result.error_message);
            }
            return ptr::null_mut();
        }

        if tokens_ptr.is_null() || tokens_len == 0 {
            return ptr::null_mut();
        }

        // Convert to Java int array
        let tokens_slice = std::slice::from_raw_parts(tokens_ptr, tokens_len);
        let java_tokens: Vec<i32> = tokens_slice.iter().map(|&t| t as i32).collect();
        
        let result_array = env.new_int_array(java_tokens.len() as i32).unwrap();
        env.set_int_array_region(result_array, 0, &java_tokens).unwrap();

        // Free the native tokens
        harmony_free_tokens(tokens_ptr, tokens_len);

        result_array
    }
}

/// Render a structured Harmony prompt
#[no_mangle]
pub extern "system" fn Java_ai_noesisreality_harmony_HarmonyEngine_nativeRenderPrompt(
    env: JNIEnv,
    _class: JClass,
    encoder_ptr: jlong,
    system_message: JString,
    user_message: JString,
    assistant_prefix: JString,
) -> jintArray {
    let encoder = encoder_ptr as *const std::ffi::c_void;
    if encoder.is_null() {
        return ptr::null_mut();
    }

    // Convert Java strings to C strings
    let user_str = match env.get_string(user_message) {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    let c_user = match CString::new(user_str.to_str().unwrap_or("")) {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    // Handle optional system message
    let c_system = if system_message.is_null() {
        None
    } else {
        match env.get_string(system_message) {
            Ok(s) => match CString::new(s.to_str().unwrap_or("")) {
                Ok(cs) => Some(cs),
                Err(_) => return ptr::null_mut(),
            },
            Err(_) => None,
        }
    };

    // Handle optional assistant prefix
    let c_assistant = if assistant_prefix.is_null() {
        None
    } else {
        match env.get_string(assistant_prefix) {
            Ok(s) => match CString::new(s.to_str().unwrap_or("")) {
                Ok(cs) => Some(cs),
                Err(_) => return ptr::null_mut(),
            },
            Err(_) => None,
        }
    };

    unsafe {
        let mut tokens_ptr: *mut u32 = ptr::null_mut();
        let mut tokens_len: usize = 0;

        let result = harmony_encoding_render_prompt(
            encoder,
            c_system.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
            c_user.as_ptr(),
            c_assistant.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
            &mut tokens_ptr,
            &mut tokens_len,
        );

        if !result.success {
            if !result.error_message.is_null() {
                harmony_free_string(result.error_message);
            }
            return ptr::null_mut();
        }

        if tokens_ptr.is_null() || tokens_len == 0 {
            return ptr::null_mut();
        }

        // Convert to Java int array
        let tokens_slice = std::slice::from_raw_parts(tokens_ptr, tokens_len);
        let java_tokens: Vec<i32> = tokens_slice.iter().map(|&t| t as i32).collect();
        
        let result_array = env.new_int_array(java_tokens.len() as i32).unwrap();
        env.set_int_array_region(result_array, 0, &java_tokens).unwrap();

        // Free the native tokens
        harmony_free_tokens(tokens_ptr, tokens_len);

        result_array
    }
}

/// Decode tokens back to text
#[no_mangle]
pub extern "system" fn Java_ai_noesisreality_harmony_HarmonyEngine_nativeDecode(
    env: JNIEnv,
    _class: JClass,
    encoder_ptr: jlong,
    tokens: JIntArray,
) -> jstring {
    let encoder = encoder_ptr as *const std::ffi::c_void;
    if encoder.is_null() {
        return ptr::null_mut();
    }

    // Convert Java int array to native u32 array
    let tokens_len = env.get_array_length(tokens).unwrap() as usize;
    let mut java_tokens = vec![0i32; tokens_len];
    env.get_int_array_region(tokens, 0, &mut java_tokens).unwrap();
    
    let native_tokens: Vec<u32> = java_tokens.iter().map(|&t| t as u32).collect();

    unsafe {
        let text_ptr = harmony_encoding_decode(
            encoder,
            native_tokens.as_ptr(),
            native_tokens.len(),
        );

        if text_ptr.is_null() {
            return ptr::null_mut();
        }

        let c_str = CStr::from_ptr(text_ptr);
        let text = match c_str.to_str() {
            Ok(s) => s,
            Err(_) => {
                harmony_free_string(text_ptr);
                return ptr::null_mut();
            }
        };

        let result = env.new_string(text).unwrap();
        harmony_free_string(text_ptr);

        result.into_inner()
    }
}

/// Get stop tokens
#[no_mangle]
pub extern "system" fn Java_ai_noesisreality_harmony_HarmonyEngine_nativeGetStopTokens(
    env: JNIEnv,
    _class: JClass,
    encoder_ptr: jlong,
) -> jintArray {
    let encoder = encoder_ptr as *const std::ffi::c_void;
    if encoder.is_null() {
        return ptr::null_mut();
    }

    unsafe {
        let mut tokens_ptr: *mut u32 = ptr::null_mut();
        let mut tokens_len: usize = 0;

        let result = harmony_encoding_stop_tokens(
            encoder,
            &mut tokens_ptr,
            &mut tokens_len,
        );

        if !result.success {
            if !result.error_message.is_null() {
                harmony_free_string(result.error_message);
            }
            return ptr::null_mut();
        }

        if tokens_ptr.is_null() || tokens_len == 0 {
            return ptr::null_mut();
        }

        // Convert to Java int array
        let tokens_slice = std::slice::from_raw_parts(tokens_ptr, tokens_len);
        let java_tokens: Vec<i32> = tokens_slice.iter().map(|&t| t as i32).collect();
        
        let result_array = env.new_int_array(java_tokens.len() as i32).unwrap();
        env.set_int_array_region(result_array, 0, &java_tokens).unwrap();

        // Free the native tokens
        harmony_free_tokens(tokens_ptr, tokens_len);

        result_array
    }
}
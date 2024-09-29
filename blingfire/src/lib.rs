//! blingfire is a thin Rust wrapper for the
//! [BlingFire](https://github.com/microsoft/BlingFire) tokenization library.

mod errors;

use blingfire_sys::{
    FreeModel as free_model_ffi, LoadModel as load_model_ffi, TextToIds as text_to_ids_ffi,
};

use snafu::{self, ensure};

use std::{convert::TryInto, ffi::CString, i32, os::raw::c_char};

pub use crate::errors::{Error, Result};

#[inline]
pub fn load_model(model_path: &str) -> Result<*mut std::ffi::c_void> {
    let c_str = CString::new(model_path).unwrap();
    let model_ptr = unsafe { load_model_ffi(c_str.as_ptr() as *const c_char) };
    ensure!(!model_ptr.is_null(), errors::LoadModelError);
    Ok(model_ptr)
}

#[inline]
pub fn text_to_ids(model_ptr: *mut std::ffi::c_void, source: &str) -> Result<Vec<i32>> {
    let src_byte_len = source.as_bytes().len();
    let mut destination = vec![0; src_byte_len];

    if source.is_empty() {
        return Ok(destination);
    } else {
        unsafe {
            text_to_ids_ffi(
                model_ptr,
                source.as_ptr() as *const c_char,
                src_byte_len.try_into().unwrap_or(i32::MAX),
                destination.as_mut_ptr(),
                destination.len().try_into().unwrap_or(i32::MAX),
                3,
            );
        }
        return Ok(destination);
    }
}

#[inline]
pub fn free_model(model_ptr: *mut std::ffi::c_void) -> Result<()> {
    let result = unsafe { free_model_ffi(model_ptr) };
    ensure!(result == 1, errors::FreeModelError);
    Ok(())
}

// /// The maximum valid size of the input text for the tokenizer functions.
// /// Re-exported from the C++ library.
// pub const MAX_TEXT_LENGTH: usize = FA_LIMITS_MAX_ARRAY_SIZE as usize;

// /// Tokenizes a piece of text into words separated by whitespace.
// ///
// /// The result of the tokenization operation is stored in the string
// /// `destination`. The string will first be cleared.
// ///
// /// ## Errors
// ///
// /// This method returns an error when the input string is too large or when the
// /// C++ function fails for an unknown reason (i.e. returns -1).
// ///
// /// ## Example
// ///
// /// ```
// /// # fn main() -> Result<(), blingfire::Error> {
// ///     let mut parsed = String::new();
// ///     blingfire::text_to_words("Cat,sat on   the mat.", &mut parsed)?;
// ///     assert_eq!(parsed.as_str(), "Cat , sat on the mat .");
// ///     # Ok(())
// /// # }
// /// ```
// #[inline]
// pub fn text_to_words(source: &str, destination: &mut String) -> Result<()> {
//     tokenize(text_to_words_ffi, source, destination)
// }

// /// Tokenizes a piece of text into sentences separated by whitespace.
// ///
// /// The result of the tokenization operation is stored in the string
// /// `destination`. The string will first be cleared.
// ///
// /// ## Errors
// ///
// /// This method returns an error when the input string is too large or when the
// /// C++ function fails for an unknown reason (i.e. returns -1).
// ///
// /// ## Example
// ///
// /// ```
// /// # fn main() -> Result<(), blingfire::Error> {
// ///     let mut parsed = String::new();
// ///     blingfire::text_to_sentences("Cat sat. Dog barked.", &mut parsed).unwrap();
// ///     assert_eq!(parsed.as_str(), "Cat sat.\nDog barked.");
// ///     # Ok(())
// /// # }
// /// ```
// #[inline]
// pub fn text_to_sentences(source: &str, destination: &mut String) -> Result<()> {
//     tokenize(text_to_sentences_ffi, source, destination)
// }

// type Tokenizer = unsafe extern "C" fn(*const c_char, c_int, *mut c_char, c_int) -> c_int;

// #[inline]
// fn tokenize(tokenizer: Tokenizer, source: &str, destination: &mut String) -> Result<()>
// where
// {
//     destination.clear();

//     if source.is_empty() {
//         return Ok(());
//     }

//     let source_len = source.len();
//     ensure!(
//         source_len <= MAX_TEXT_LENGTH,
//         errors::SourceTooLarge {
//             max_text_length: MAX_TEXT_LENGTH
//         }
//     );
//     let source_len = source_len as c_int;

//     loop {
//         let length = unsafe {
//             tokenizer(
//                 source.as_ptr() as *const c_char,
//                 source_len,
//                 destination.as_mut_ptr() as *mut c_char,
//                 destination.capacity().try_into().unwrap_or(i32::MAX),
//             )
//         };

//         // The C++ function returned -1, an unknown error.
//         ensure!(length > 0, errors::UnknownError);

//         if length as usize > destination.capacity() {
//             // There was not enough capacity in `destination` to store the parsed text.
//             // Although the C++ function allocated an internal buffer with the parsed text, that's
//             // not exposed. We'll have to reserve `length` bytes in `destination` (as
//             // `destination.len() == 0`) and parse the `source` string again.
//             destination.reserve_exact(length as usize);
//             continue;
//         } else {
//             // The text was successfully parsed, set the length to the return value (-1 for the
//             // null character).
//             unsafe {
//                 destination.as_mut_vec().set_len(length as usize - 1);
//             }
//             break;
//         }
//     }

//     Ok(())
// }

#[cfg(test)]
mod tests {
    use super::{free_model, load_model, text_to_ids};

    #[test]
    fn test_load_and_free_model() {
        let model_ptr = load_model("data/xlm_roberta.bling").unwrap();

        assert!(!model_ptr.is_null());

        free_model(model_ptr).unwrap();
    }

    #[test]
    fn test_tokenize() {
        let model_ptr = load_model("data/xlm_roberta.bling").unwrap();

        let s = "hello world";
        let ids = text_to_ids(model_ptr, s).unwrap();
        println!("{:?}", ids);

        free_model(model_ptr).unwrap();
    }
}

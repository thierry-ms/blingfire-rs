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

    let c_str = CString::new(source).unwrap();

    if source.is_empty() {
        return Ok(destination);
    } else {
        unsafe {
            text_to_ids_ffi(
                model_ptr,
                c_str.as_ptr() as *const c_char,
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

        let s = "Эpple pie. How do I renew my virtual smart card?: /Microsoft IT/ 'virtual' smart card certificates for DirectAccess are valid for one year. In order to get to microsoft.com we need to type pi@1.2.1.2.";
        let ids = text_to_ids(model_ptr, s).unwrap();
        println!("{:?}", ids);

        free_model(model_ptr).unwrap();
    }
}

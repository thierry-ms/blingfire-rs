//! blingfire is a thin Rust wrapper for the
//! [BlingFire](https://github.com/microsoft/BlingFire) tokenization library.

mod errors;

use blingfire_sys::{
    FreeModel as free_model_ffi, LoadModel as load_model_ffi, TextToIds as text_to_ids_ffi,
};

use rayon::prelude::*;
use snafu::{self, ensure};

use std::{convert::TryInto, ffi::CString, i32, os::raw::c_char};

pub use crate::errors::{Error, Result};

#[inline]
pub fn load_model(model_path: &str) -> Result<ModelWrapper> {
    let c_str = CString::new(model_path).unwrap();
    let model_ptr = unsafe { load_model_ffi(c_str.as_ptr() as *const c_char) };
    ensure!(!model_ptr.is_null(), errors::LoadModelError);
    Ok(ModelWrapper(Model { model_ptr }))
}

#[inline]
pub fn text_to_ids(model_wrapper: &ModelWrapper, source: &str) -> Result<Vec<i32>> {
    let src_byte_len = source.as_bytes().len();
    let mut destination = vec![0; std::cmp::min(src_byte_len, 500)];

    let c_str = CString::new(source).unwrap();

    if source.is_empty() {
        return Ok(destination);
    } else {
        unsafe {
            text_to_ids_ffi(
                model_wrapper.0.model_ptr,
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

pub struct Model {
    pub model_ptr: *mut ::std::os::raw::c_void,
}

pub struct ModelWrapper(Model);

unsafe impl Send for ModelWrapper {}
unsafe impl Sync for ModelWrapper {}

#[inline]
pub fn texts_to_ids(model_wrapper: &ModelWrapper, sources: Vec<String>) -> Result<Vec<Vec<i32>>> {
    sources
        .into_par_iter()
        .map(|source| text_to_ids(model_wrapper, &source))
        .collect()
}

#[inline]
pub fn free_model(model_wrapper: ModelWrapper) -> Result<()> {
    let result = unsafe { free_model_ffi(model_wrapper.0.model_ptr) };
    ensure!(result == 1, errors::FreeModelError);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{free_model, load_model, text_to_ids, texts_to_ids};
    use std::{
        fs::{read_to_string, File},
        io::Write,
    };

    #[test]
    fn test_load_and_free_model() {
        let model_wrapper = load_model("data/xlm_roberta.bling").unwrap();

        assert!(!model_wrapper.0.model_ptr.is_null());

        free_model(model_wrapper).unwrap();
    }

    #[test]
    fn test_tokenize() {
        let model_wrapper = load_model("data/xlm_roberta.bling").unwrap();

        let s = "Эpple pie. How do I renew my virtual smart card?: /Microsoft IT/ 'virtual' smart card certificates for DirectAccess are valid for one year. In order to get to microsoft.com we need to type pi@1.2.1.2.";
        text_to_ids(&model_wrapper, s).unwrap();

        free_model(model_wrapper).unwrap();
    }

    #[test]
    fn test_tokenize_batch() {
        let mut lines = Vec::<String>::new();
        for line in read_to_string("data/blingfire_input.utf8").unwrap().lines() {
            lines.push(line.to_string());
        }

        let model_wrapper = load_model("data/xlm_roberta.bling").unwrap();

        let mut data_file = File::create("data/blingfire_output_rs.utf8").expect("creation failed");

        for line in lines {
            let mut ids = text_to_ids(&model_wrapper, &line).unwrap();
            if let Some(last) = ids.iter().rposition(|x| *x != 0) {
                let actual_len = last + 1;
                ids.truncate(actual_len);
            }

            data_file.write("[".as_bytes()).unwrap();

            for i in 0..ids.len() {
                data_file.write(ids[i].to_string().as_bytes()).unwrap();
                if i < ids.len() - 1 {
                    data_file.write(", ".as_bytes()).unwrap();
                }
            }

            data_file.write("]\n".as_bytes()).unwrap();
        }

        free_model(model_wrapper).unwrap();
    }

    #[test]
    fn test_tokenize_batch_in_par() {
        let mut lines = Vec::<String>::new();
        for line in read_to_string("data/blingfire_input.utf8").unwrap().lines() {
            lines.push(line.to_string());
        }

        let model_wrapper = load_model("data/xlm_roberta.bling").unwrap();

        let mut data_file =
            File::create("data/blingfire_output_rs_in_par.utf8").expect("creation failed");

        let idss = texts_to_ids(&model_wrapper, lines).unwrap();

        for mut ids in idss {
            if let Some(last) = ids.iter().rposition(|x| *x != 0) {
                let actual_len = last + 1;
                ids.truncate(actual_len);
            }

            data_file.write("[".as_bytes()).unwrap();

            for i in 0..ids.len() {
                data_file.write(ids[i].to_string().as_bytes()).unwrap();
                if i < ids.len() - 1 {
                    data_file.write(", ".as_bytes()).unwrap();
                }
            }

            data_file.write("]\n".as_bytes()).unwrap();
        }

        free_model(model_wrapper).unwrap();
    }
}

#![allow(non_camel_case_types, non_upper_case_globals, non_snake_case)]
/* automatically generated by rust-bindgen 0.70.1 */

extern "C" {
    pub fn LoadModel(pszLdbFileName: *const ::std::os::raw::c_char) -> *mut ::std::os::raw::c_void;
}
extern "C" {
    pub fn TextToIds(
        ModelPtr: *mut ::std::os::raw::c_void,
        pInUtf8Str: *const ::std::os::raw::c_char,
        InUtf8StrByteCount: ::std::os::raw::c_int,
        pIdsArr: *mut i32,
        MaxIdsArrLength: ::std::os::raw::c_int,
        UnkId: ::std::os::raw::c_int,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn FreeModel(ModelPtr: *mut ::std::os::raw::c_void) -> ::std::os::raw::c_int;
}

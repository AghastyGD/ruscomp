mod huffman;
mod rle;
use pyo3::prelude::*;

#[pymodule]
fn ruscomp(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rle::rle_encode, m)?)?;
    m.add_function(wrap_pyfunction!(rle::rle_decode, m)?)?;
    m.add_function(wrap_pyfunction!(huffman::huffman_encode, m)?)?;
    m.add_function(wrap_pyfunction!(huffman::huffman_decode, m)?)?;
    Ok(())
}

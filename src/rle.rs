use pyo3::prelude::*;

#[pyfunction]
pub fn rle_encode(input: &str) -> String {
    if input.is_empty() {
        return String::new();
    }

    let mut result = String::new();
    let mut count = 1;
    let chars: Vec<char> = input.chars().collect();

    for i in 0..chars.len() -1 {
        if chars[i] == chars[i+1] {
            count += 1;
        } else {
            result.push_str(&format!("{}{}", count, chars[i]));
            count = 1;
        }
    }

    result.push_str(&format!("{}{}", count, chars[chars.len() - 1]));
    result
}

#[pyfunction]
pub fn rle_decode(encoded: &str) -> String {
    let mut result = String::new();
    let mut count = String::new();

    for ch in encoded.chars() {
        if ch.is_ascii_digit() {
            count.push(ch);
        } else {
            if let Ok(n) = count.parse::<usize>() {
                for _ in 0..n {
                    result.push(ch);
                }
            }
            count.clear();
        }
    }

    result
}

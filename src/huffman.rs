use bitvec::prelude::*;
use pyo3::{exceptions::PyValueError, prelude::*};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::rc::Rc;

type BitStorage = BitVec<u8, Msb0>;

#[derive(Debug, Clone)]
enum Node {
    Leaf(char),
    Internal(Rc<Node>, Rc<Node>),
}

#[derive(Debug, Clone)]
struct HuffmanNode {
    freq: usize,
    order: usize,
    node: Rc<Node>,
}

impl PartialEq for HuffmanNode {
    fn eq(&self, other: &Self) -> bool {
        self.freq == other.freq && self.order == other.order
    }
}

impl Eq for HuffmanNode {}

impl PartialOrd for HuffmanNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HuffmanNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering to make BinaryHeap behave like a min-heap.
        other
            .freq
            .cmp(&self.freq)
            .then_with(|| other.order.cmp(&self.order))
    }
}

fn build_huffman_tree(freq_map: &BTreeMap<char, usize>) -> Option<Rc<Node>> {
    let mut heap = BinaryHeap::new();
    let mut order_counter: usize = 0;

    for (&ch, &freq) in freq_map {
        order_counter += 1;
        heap.push(HuffmanNode {
            freq,
            order: order_counter,
            node: Rc::new(Node::Leaf(ch)),
        });
    }

    if heap.is_empty() {
        return None;
    }

    while heap.len() > 1 {
        let HuffmanNode {
            freq: left_freq,
            node: left_node,
            ..
        } = heap.pop().expect("heap guaranteed non-empty");
        let HuffmanNode {
            freq: right_freq,
            node: right_node,
            ..
        } = heap.pop().expect("heap guaranteed non-empty");

        order_counter += 1;
        heap.push(HuffmanNode {
            freq: left_freq + right_freq,
            order: order_counter,
            node: Rc::new(Node::Internal(left_node, right_node)),
        });
    }

    heap.pop()
        .map(|node| node.node)
}

fn generate_codes(node: &Node, prefix: BitStorage, codes: &mut HashMap<char, BitStorage>) {
    match node {
        Node::Leaf(ch) => {
            let mut canonical = prefix;
            if canonical.is_empty() {
                // Degenerate tree (single symbol) still gets one bit.
                canonical.push(false);
            }
            codes.insert(*ch, canonical);
        }
        Node::Internal(left, right) => {
            let mut left_prefix = prefix.clone();
            left_prefix.push(false);
            generate_codes(left, left_prefix, codes);

            let mut right_prefix = prefix;
            right_prefix.push(true);
            generate_codes(right, right_prefix, codes);
        }
    }
}

#[pyfunction]
pub fn huffman_encode(text: &str) -> PyResult<(Vec<u8>, usize, HashMap<String, String>)> {
    if text.is_empty() {
        return Ok((Vec::new(), 0, HashMap::new()));
    }

    let mut freq_map: BTreeMap<char, usize> = BTreeMap::new();
    for ch in text.chars() {
        *freq_map.entry(ch).or_insert(0) += 1;
    }

    let tree = build_huffman_tree(&freq_map)
        .ok_or_else(|| PyValueError::new_err("unable to build Huffman tree from input"))?;
    let mut codes: HashMap<char, BitStorage> = HashMap::new();
    generate_codes(&tree, BitStorage::new(), &mut codes);

    let mut encoded_bits = BitStorage::new();
    for ch in text.chars() {
        let code = codes
            .get(&ch)
            .expect("every character must have a generated Huffman code");
        encoded_bits.extend_from_bitslice(code.as_bitslice());
    }

    let bit_len = encoded_bits.len();
    let encoded_bytes = encoded_bits.into_vec();

    let mut codes_str: HashMap<String, String> = HashMap::new();
    for (symbol, bits) in codes {
        let code_str: String = bits
            .iter()
            .map(|bit| if *bit { '1' } else { '0' })
            .collect();
        codes_str.insert(symbol.to_string(), code_str);
    }

    Ok((encoded_bytes, bit_len, codes_str))
}

#[pyfunction]
pub fn huffman_decode(
    data: Vec<u8>,
    bit_len: usize,
    codes: HashMap<String, String>,
) -> PyResult<String> {
    if bit_len == 0 {
        return Ok(String::new());
    }

    let mut reverse: HashMap<String, char> = HashMap::with_capacity(codes.len());
    for (symbol, code) in codes {
        let mut chars = symbol.chars();
        let ch = chars
            .next()
            .ok_or_else(|| PyValueError::new_err("Huffman codes require non-empty symbols"))?;
        if chars.next().is_some() {
            return Err(PyValueError::new_err(
                "Huffman decode expects single-character symbols",
            ));
        }
        if code.is_empty() {
            return Err(PyValueError::new_err(
                "Huffman codes must contain at least one bit",
            ));
        }
        if reverse.insert(code, ch).is_some() {
            return Err(PyValueError::new_err(
                "Duplicate Huffman bit-pattern detected",
            ));
        }
    }

    if reverse.is_empty() {
        return Err(PyValueError::new_err(
            "No Huffman codes provided for decoding",
        ));
    }

    let bits = BitVec::<u8, Msb0>::from_vec(data);
    if bit_len > bits.len() {
        return Err(PyValueError::new_err(
            "bit_len exceeds number of bits provided",
        ));
    }

    let mut decoded = String::new();
    let mut current = String::new();
    for bit in bits.iter().take(bit_len) {
        current.push(if *bit { '1' } else { '0' });
        if let Some(&ch) = reverse.get(&current) {
            decoded.push(ch);
            current.clear();
        }
    }

    if current.is_empty() {
        Ok(decoded)
    } else {
        Err(PyValueError::new_err(
            "Trailing bits did not map to a Huffman symbol",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn huffman_round_trip_basic() {
        let text = "hello huffman";
        let (data, bit_len, codes) = huffman_encode(text).expect("encode should succeed");
        let decoded =
            huffman_decode(data, bit_len, codes).expect("decode should successfully round-trip");
        assert_eq!(decoded, text);
    }

    #[test]
    fn huffman_round_trip_single_symbol() {
        let text = "aaaaaa";
        let (data, bit_len, codes) = huffman_encode(text).expect("encode should succeed");
        assert!(bit_len >= text.len());
        let decoded =
            huffman_decode(data, bit_len, codes).expect("decode should successfully round-trip");
        assert_eq!(decoded, text);
    }
}

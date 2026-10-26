// 1046. 最后一块石头的重量
pub mod n1046 {
    pub fn last_stone_weight(stones: Vec<i32>) -> i32 {
        let mut heap = std::collections::BinaryHeap::from(stones);
        while heap.len() > 1 {
            let x = heap.pop().unwrap();
            let y = heap.pop().unwrap();
            if x != y {
                heap.push(x - y);
            }
        }
        heap.pop().unwrap_or(0)
    }
}

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

// 3264. K 次乘运算后的最终数组 I
pub mod n3264 {
    pub fn get_final_state(nums: Vec<i32>, k: i32, multiplier: i32) -> Vec<i32> {
        // Reverse 实现最小堆，元组 (value, index)
        let mut pq = std::collections::BinaryHeap::new();
        let mut arr = nums;
        for (idx, &val) in arr.iter().enumerate() {
            pq.push(std::cmp::Reverse((val, idx)));
        }

        for _ in 0..k {
            let std::cmp::Reverse((_, i)) = pq.pop().unwrap();
            arr[i] *= multiplier;
            pq.push(std::cmp::Reverse((arr[i], i)));
        }
        arr
    }
}

// 2558. 从数量最多的堆取走礼物
pub mod n2558 {
    pub fn pick_gifts(gifts: Vec<i32>, k: i32) -> i64 {
        let mut h = std::collections::BinaryHeap::from(gifts);
        for _ in 0..k {
            let top = h.pop().unwrap();
            h.push((top as f64).sqrt() as i32);
            if *h.peek().unwrap() == 1 {
                break;
            }
        }
        h.iter().map(|&x| x as i64).sum()
    }
}

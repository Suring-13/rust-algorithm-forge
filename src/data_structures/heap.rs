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

// 2336. 无限集中的最小数字
pub mod n2336 {
    use std::collections::BTreeSet;

    pub struct SmallestInfiniteSet {
        pub thres: i32,
        pub s: BTreeSet<i32>,
    }

    impl Default for SmallestInfiniteSet {
        fn default() -> Self {
            Self::new()
        }
    }

    impl SmallestInfiniteSet {
        pub fn new() -> Self {
            Self {
                thres: 1,
                s: BTreeSet::new(),
            }
        }

        pub fn pop_smallest(&mut self) -> i32 {
            if self.s.is_empty() {
                let ans = self.thres;
                self.thres += 1;
                ans
            } else {
                let &val = self.s.iter().next().unwrap();
                self.s.remove(&val);
                val
            }
        }

        pub fn add_back(&mut self, num: i32) {
            if num < self.thres {
                self.s.insert(num);
            }
        }
    }
}

// 2530. 执行 K 次操作后的最大分数
pub mod n2530 {
    pub fn max_kelements(nums: Vec<i32>, k: i32) -> i64 {
        let mut h = std::collections::BinaryHeap::from(nums);

        let mut ans = 0i64;

        for _ in 0..k {
            let mx = h.pop().unwrap();

            ans += mx as i64;

            h.push((mx + 2) / 3);
        }

        ans
    }
}

// 3066. 超过阈值的最少操作数 II
pub mod n3066 {
    pub fn min_operations(nums: Vec<i32>, k: i32) -> i32 {
        let mut h = std::collections::BinaryHeap::new();
        for x in nums {
            h.push(std::cmp::Reverse(x as i64));
        }

        let mut ans = 0;
        while h.len() > 1
            && let Some(std::cmp::Reverse(x)) = h.peek()
            && *x < k as i64
        {
            let std::cmp::Reverse(x) = h.pop().unwrap();
            let std::cmp::Reverse(y) = h.pop().unwrap();
            h.push(std::cmp::Reverse(x * 2 + y));
            ans += 1;
        }
        ans
    }
}

// 1962. 移除石子使总数最小
pub mod n1962 {
    pub fn min_stone_sum(piles: Vec<i32>, k: i32) -> i32 {
        let mut h = std::collections::BinaryHeap::from(piles);
        for _ in 0..k {
            let top = h.pop().unwrap();
            h.push((top + 1) / 2);
            if *h.peek().unwrap() == 1 {
                break;
            }
        }
        h.iter().sum()
    }
}

// 703. 数据流中的第 K 大元素
pub mod n703 {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    pub struct KthLargest {
        pub k: usize,
        pub heap: BinaryHeap<Reverse<i32>>,
    }

    impl KthLargest {
        pub fn new(k: i32, nums: Vec<i32>) -> Self {
            let k = k as usize;
            let mut heap = BinaryHeap::new();
            for v in nums {
                heap.push(Reverse(v));
                if heap.len() > k {
                    heap.pop();
                }
            }
            Self { k, heap }
        }

        pub fn add(&mut self, val: i32) -> i32 {
            self.heap.push(Reverse(val));
            if self.heap.len() > self.k {
                self.heap.pop();
            }
            self.heap.peek().unwrap().0
        }
    }
}

// 3275. 第 K 近障碍物查询
pub mod n3275 {
    pub fn results_array(queries: Vec<Vec<i32>>, k: i32) -> Vec<i32> {
        let k = k as usize;
        let mut ans = vec![-1; queries.len()];
        let mut heap = std::collections::BinaryHeap::new();

        for (idx, q) in queries.into_iter().enumerate() {
            let d = q[0].abs() + q[1].abs();
            heap.push(d);

            if heap.len() > k {
                heap.pop();
            }
            if heap.len() == k {
                ans[idx] = *heap.peek().unwrap();
            }
        }
        ans
    }
}

// 1441. 用栈操作构建数组
pub mod n1441 {
    pub fn build_array(target: Vec<i32>, _n: i32) -> Vec<String> {
        let mut res = Vec::new();
        let mut cur = 1;
        for num in target {
            while cur < num {
                res.push("Push".into());
                res.push("Pop".into());
                cur += 1;
            }
            res.push("Push".into());
            cur += 1;
        }
        res
    }
}

// 844. 比较含退格的字符串
pub mod n844 {
    pub fn backspace_compare(s: String, t: String) -> bool {
        fn build(s: &str) -> String {
            let mut ret = Vec::new();
            for ch in s.chars() {
                if ch != '#' {
                    ret.push(ch);
                } else if !ret.is_empty() {
                    ret.pop();
                }
            }
            ret.into_iter().collect()
        }

        build(&s) == build(&t)
    }
}

// 682. 棒球比赛
pub mod n682 {
    pub fn cal_points(operations: Vec<String>) -> i32 {
        let mut st = vec![];
        for op in operations {
            match op.as_bytes()[0] {
                b'+' => st.push(st[st.len() - 2] + st[st.len() - 1]),
                b'D' => st.push(st[st.len() - 1] * 2),
                b'C' => {
                    st.pop();
                }
                _ => st.push(op.parse::<i32>().unwrap()),
            }
        }
        st.iter().sum()
    }
}

// 2390. 从字符串中移除星号
pub mod n2390 {
    pub fn remove_stars(s: String) -> String {
        let mut st = Vec::new();
        for c in s.bytes() {
            if c == b'*' {
                st.pop();
            } else {
                st.push(c);
            }
        }
        String::from_utf8(st).unwrap()
    }
}

// 3412. 计算字符串的镜像分数
pub mod n3412 {
    pub fn calculate_score(s: String) -> i64 {
        // 26个字母，每个维护一个栈保存下标
        let mut stk: Vec<Vec<usize>> = vec![Vec::new(); 26];
        let mut ans: i64 = 0;

        for (i, ch) in s.bytes().enumerate() {
            let c = (ch - b'a') as usize;
            let mirror = 25 - c;

            if let Some(prev_idx) = stk[mirror].pop() {
                ans += (i - prev_idx) as i64;
            } else {
                stk[c].push(i);
            }
        }
        ans
    }
}

// 71. 简化路径
pub mod n71 {
    pub fn simplify_path(path: String) -> String {
        let mut stk = vec![];
        for s in path.split('/') {
            match s {
                "" | "." => continue,
                ".." => {
                    stk.pop();
                }
                _ => stk.push(s),
            }
        }
        format!("/{}", stk.join("/"))
    }
}

// 3170. 删除星号以后字典序最小的字符串
pub mod n3170 {
    pub fn clear_stars(s: String) -> String {
        let mut chars: Vec<char> = s.chars().collect();
        let mut stacks: Vec<Vec<usize>> = vec![vec![]; 26];

        for i in 0..chars.len() {
            let c = chars[i];
            if c != '*' {
                let idx = (c as u8 - b'a') as usize;
                stacks[idx].push(i);
            } else {
                // 从小到大遍历26个栈，找到第一个非空栈
                for st in &mut stacks {
                    if !st.is_empty() {
                        let pos = st.pop().unwrap();
                        chars[pos] = '*';
                        break;
                    }
                }
            }
        }

        chars.into_iter().filter(|&ch| ch != '*').collect()
    }
}

// 155. 最小栈
pub mod n155 {
    pub struct MinStack {
        pub st: Vec<i64>,
        pub mn: i64,
    }

    impl Default for MinStack {
        fn default() -> Self {
            Self::new()
        }
    }

    impl MinStack {
        pub fn new() -> Self {
            Self {
                st: vec![],
                mn: i64::MAX / 2, // 防止 val‑mn 溢出
            }
        }

        pub fn push(&mut self, val: i32) {
            // 压入差值：val - push之前的最小值
            self.st.push(val as i64 - self.mn);
            // 更新全局最小值
            self.mn = self.mn.min(val as i64);
        }

        pub fn pop(&mut self) {
            let diff = self.st.pop().unwrap();
            // diff < 0：代表这次入栈时更新过最小值，弹出要恢复旧最小值
            self.mn -= diff.min(0);
        }

        pub fn top(&self) -> i32 {
            let diff = *self.st.last().unwrap();
            // diff>0:原值=mn+diff；diff<=0:原值就是mn
            (self.mn + diff.max(0)) as i32
        }

        pub fn get_min(&self) -> i32 {
            self.mn as i32
        }
    }
}

// 1381. 设计一个支持增量操作的栈
pub mod n1381 {
    pub struct CustomStack {
        pub stk: Vec<i32>,
        pub add: Vec<i32>,
        pub top: i32,
    }

    impl CustomStack {
        pub fn new(max_size: i32) -> Self {
            let size = max_size as usize;
            Self {
                stk: vec![0; size],
                add: vec![0; size],
                top: -1,
            }
        }

        pub fn push(&mut self, x: i32) {
            if self.top != (self.stk.len() - 1) as i32 {
                self.top += 1;
                let idx = self.top as usize;
                self.stk[idx] = x;
            }
        }

        pub fn pop(&mut self) -> i32 {
            if self.top == -1 {
                return -1;
            }
            let idx = self.top as usize;
            let ret = self.stk[idx] + self.add[idx];
            if self.top != 0 {
                self.add[idx - 1] += self.add[idx];
            }
            self.add[idx] = 0;
            self.top -= 1;
            ret
        }

        pub fn increment(&mut self, k: i32, val: i32) {
            let lim = std::cmp::min(k - 1, self.top);
            if lim >= 0 {
                let lim_idx = lim as usize;
                self.add[lim_idx] += val;
            }
        }
    }
}

// 895. 最大频率栈
pub mod n895 {
    use std::collections::HashMap;

    pub struct FreqStack {
        // 栈套栈：stacks[i] 存放出现频率为 i+1 的元素栈
        pub stacks: Vec<Vec<i32>>,
        // key:值，value:该值当前出现次数
        pub cnt: HashMap<i32, usize>,
    }

    impl Default for FreqStack {
        fn default() -> Self {
            Self::new()
        }
    }

    impl FreqStack {
        pub fn new() -> Self {
            FreqStack {
                stacks: Vec::new(),
                cnt: HashMap::new(),
            }
        }

        pub fn push(&mut self, val: i32) {
            // 获取当前 val 的计数，不存在则0
            let count = *self.cnt.get(&val).unwrap_or(&0);

            if count == self.stacks.len() {
                self.stacks.push(vec![val]);
            } else {
                self.stacks[count].push(val);
            }

            *self.cnt.entry(val).or_insert(0) += 1;
        }

        pub fn pop(&mut self) -> i32 {
            // 弹出最右侧栈的栈顶
            let val = self.stacks.last_mut().unwrap().pop().unwrap();

            // 如果当前最高频栈空了，移除这个栈
            if self.stacks.last().unwrap().is_empty() {
                self.stacks.pop();
            }

            // 计数减一
            *self.cnt.get_mut(&val).unwrap() -= 1;

            val
        }
    }
}

// 1172. 餐盘栈
pub mod n1172 {
    use std::collections::BinaryHeap;

    pub struct DinnerPlates {
        pub capacity: usize,
        pub stacks: Vec<Vec<i32>>,
        // 小顶堆：保存未满栈下标；Rust BinaryHeap 是大顶堆，存负数实现小顶堆
        pub heap: BinaryHeap<std::cmp::Reverse<usize>>,
    }

    impl DinnerPlates {
        pub fn new(capacity: i32) -> Self {
            Self {
                capacity: capacity as usize,
                stacks: Vec::new(),
                heap: BinaryHeap::new(),
            }
        }

        pub fn push(&mut self, val: i32) {
            // 如果堆顶下标已经越界，清空堆
            if let Some(&std::cmp::Reverse(top_idx)) = self.heap.peek()
                && top_idx >= self.stacks.len()
            {
                self.heap.clear();
            }

            if let Some(&std::cmp::Reverse(top_idx)) = self.heap.peek() {
                // 存在未满栈
                self.stacks[top_idx].push(val);
                if self.stacks[top_idx].len() == self.capacity {
                    self.heap.pop(); // 栈满，移出堆
                }
            } else {
                // 全部栈已满，新建栈
                self.stacks.push(vec![val]);
                if self.capacity > 1 {
                    let new_idx = self.stacks.len() - 1;
                    self.heap.push(std::cmp::Reverse(new_idx));
                }
            }
        }

        pub fn pop(&mut self) -> i32 {
            self.pop_at_stack(self.stacks.len() as i32 - 1)
        }

        pub fn pop_at_stack(&mut self, index: i32) -> i32 {
            let idx = index as usize;
            // 非法情况
            if index < 0 || idx >= self.stacks.len() || self.stacks[idx].is_empty() {
                return -1;
            }

            // 如果之前是满栈，弹出一个之后变成未满，下标加入堆
            if self.stacks[idx].len() == self.capacity {
                self.heap.push(std::cmp::Reverse(idx));
            }

            let val = self.stacks[idx].pop().unwrap();

            // 清除末尾连续空栈（懒删除）
            while let Some(last) = self.stacks.last() {
                if last.is_empty() {
                    self.stacks.pop();
                } else {
                    break;
                }
            }
            val
        }
    }
}

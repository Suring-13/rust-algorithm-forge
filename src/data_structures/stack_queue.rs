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

// 1472. 设计浏览器历史记录
pub mod n1472 {
    pub struct BrowserHistory {
        pub history: Vec<String>,
        pub cur: usize, // 当前页面是 history[cur]
    }

    impl BrowserHistory {
        pub fn new(homepage: String) -> Self {
            Self {
                history: vec![homepage],
                cur: 0,
            }
        }

        pub fn visit(&mut self, url: String) {
            self.cur += 1;
            self.history.truncate(self.cur); // 把浏览历史前进的记录全部删除
            self.history.push(url); // 从当前页跳转访问 url 对应的页面
        }

        pub fn back(&mut self, steps: i32) -> String {
            self.cur = self.cur.saturating_sub(steps as usize); // 后退 steps 步
            self.history[self.cur].clone()
        }

        pub fn forward(&mut self, steps: i32) -> String {
            self.cur = (self.cur + steps as usize).min(self.history.len() - 1); // 前进 steps 步
            self.history[self.cur].clone()
        }
    }
}

// 946. 验证栈序列
pub mod n946 {
    pub fn validate_stack_sequences(pushed: Vec<i32>, popped: Vec<i32>) -> bool {
        let mut stack = Vec::new();
        let mut i = 0;
        for &num in &pushed {
            stack.push(num);
            // 栈不为空 且栈顶等于popped[i]，持续弹出
            while let Some(&top) = stack.last() {
                if top == popped[i] {
                    stack.pop();
                    i += 1;
                } else {
                    break;
                }
            }
        }
        stack.is_empty()
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

// 636. 函数的独占时间
pub mod n636 {
    pub fn exclusive_time(n: i32, logs: Vec<String>) -> Vec<i32> {
        let mut ans = vec![0; n as usize];
        let mut stack = std::collections::VecDeque::new();
        let mut cur = -1;

        for log in logs {
            let parts: Vec<&str> = log.split(':').collect();
            let idx: usize = parts[0].parse().unwrap();
            let ts: i32 = parts[2].parse().unwrap();

            if parts[1] == "start" {
                if let Some(&top) = stack.back() {
                    ans[top] += ts - cur;
                }
                stack.push_back(idx);
                cur = ts;
            } else {
                let func = stack.pop_back().unwrap();
                ans[func] += ts - cur + 1;
                cur = ts + 1;
            }
        }
        ans
    }
}

// 2434. 使用机器人打印字典序最小的字符串
pub mod n2434 {
    pub fn robot_with_string(s: String) -> String {
        let n = s.len();

        // 计算后缀最小值
        let mut suf_min = vec![u8::MAX; n + 1];

        for (i, ch) in s.bytes().enumerate().rev() {
            suf_min[i] = suf_min[i + 1].min(ch);
        }

        let mut ans = Vec::with_capacity(n);

        let mut st = vec![];

        for (i, ch) in s.bytes().enumerate() {
            st.push(ch);

            while let Some(&top) = st.last() {
                if top > suf_min[i + 1] {
                    break;
                }

                ans.push(st.pop().unwrap());
            }
        }
        String::from_utf8(ans).unwrap()
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

// 2696. 删除子串后的字符串最小长度
pub mod n2696 {
    pub fn min_length(s: String) -> i32 {
        let mut stack = Vec::new();
        for c in s.chars() {
            if let Some(&last) = stack.last()
                && ((c == 'B' && last == 'A') || (c == 'D' && last == 'C'))
            {
                stack.pop();
                continue;
            }
            stack.push(c);
        }
        stack.len() as i32
    }
}

// 1047. 删除字符串中的所有相邻重复项
pub mod n1047 {
    pub fn remove_duplicates(s: String) -> String {
        let mut st = vec![];
        for ch in s.bytes() {
            if !st.is_empty() && st[st.len() - 1] == ch {
                st.pop();
            } else {
                st.push(ch);
            }
        }
        String::from_utf8(st).unwrap()
    }
}

// 1544. 整理字符串
pub mod n1544 {
    pub fn make_good(s: String) -> String {
        let mut stack = Vec::new();
        for &b in s.as_bytes() {
            if let Some(&top) = stack.last() {
                // 大小写ASCII差值为 32
                if top != b && (top ^ 32) == b {
                    stack.pop();
                    continue;
                }
            }
            stack.push(b);
        }

        String::from_utf8(stack).unwrap()
    }
}

// 3561. 移除相邻字符
pub mod n3561 {
    pub fn resulting_string(s: String) -> String {
        fn is_consecutive(x: u8, y: u8) -> bool {
            let d = (x as i32 - y as i32).abs();
            d == 1 || d == 25
        }

        let mut st = Vec::<u8>::new();
        for &b in s.as_bytes() {
            if let Some(&top) = st.last() {
                if is_consecutive(b, top) {
                    st.pop();
                } else {
                    st.push(b);
                }
            } else {
                st.push(b);
            }
        }
        String::from_utf8(st).unwrap()
    }
}

// 1003. 检查替换后的词是否有效
pub mod n1003 {
    pub fn is_valid(s: &str) -> bool {
        let mut st = Vec::new();
        for c in s.bytes() {
            if c > b'a' && (st.is_empty() || c - st.pop().unwrap() != 1) {
                return false;
            }
            if c < b'c' {
                st.push(c);
            }
        }
        st.is_empty()
    }
}

// 3834. 合并相邻且相等的元素
pub mod n3834 {
    pub fn merge_adjacent(nums: Vec<i32>) -> Vec<i64> {
        let mut st = Vec::new();
        for x in nums {
            let mut x = x as i64;
            while let Some(&last) = st.last() {
                if last == x {
                    st.pop();
                    x *= 2;
                } else {
                    break;
                }
            }
            st.push(x);
        }
        st
    }
}

// 2216. 美化数组的最少删除数
pub mod n2216 {
    pub fn min_deletion(nums: Vec<i32>) -> i32 {
        // stack：用来维护始终满足 beautiful 规则的前缀数组
        let mut stack = Vec::new();
        // ops：记录删除操作的总次数
        let mut ops = 0;

        for &num in &nums {
            stack.push(num);
            let stack_top_idx = stack.len() - 1;
            // 栈顶下标为奇数（第二个元素、第四个……）
            if stack_top_idx & 1 == 1 {
                // 判断相邻两个是否相等，违反 beautiful 条件
                if stack[stack_top_idx - 1] == stack[stack_top_idx] {
                    stack.pop();
                    ops += 1;
                }
            }
        }

        // 题目要求最终合法数组长度必须是偶数
        if stack.len() % 2 == 1 {
            ops += 1;
        }

        ops
    }
}

// 1209. 删除字符串中的所有相邻重复项 II
pub mod n1209 {
    pub fn remove_duplicates(s: String, k: i32) -> String {
        let mut stack: Vec<(char, i32)> = Vec::new();

        for c in s.chars() {
            match stack.last_mut() {
                // 和栈顶字符不一样，直接压入
                Some(&mut (top_c, _)) if top_c != c => {
                    stack.push((c, 1));
                }
                // 字符相同，判断是否刚好凑够k个
                Some((_, cnt)) => {
                    if *cnt == k - 1 {
                        stack.pop();
                    } else {
                        *cnt += 1;
                    }
                }
                // 栈为空的情况
                None => {
                    stack.push((c, 1));
                }
            }
        }

        // 拼接结果：每个字符重复对应次数
        stack
            .into_iter()
            .flat_map(|(ch, count)| std::iter::repeat_n(ch, count as usize))
            .collect()
    }
}

// 3703. 移除K-平衡子字符串
pub mod n3703 {
    pub fn remove_substring(s: String, k: i32) -> String {
        let mut stack: Vec<(char, usize)> = Vec::new();
        let k = k as usize;

        for c in s.chars() {
            if let Some(top) = stack.last_mut() {
                if top.0 == c {
                    top.1 += 1;
                } else {
                    stack.push((c, 1));
                }
            } else {
                stack.push((c, 1));
            }

            if c == ')' && stack.len() > 1 {
                let len = stack.len();
                let top_cnt = stack[len - 1].1;
                let prev_cnt = stack[len - 2].1;
                if top_cnt == k && prev_cnt >= k {
                    stack.pop();
                    let prev = stack.last_mut().unwrap();
                    prev.1 -= k;
                    if prev.1 == 0 {
                        stack.pop();
                    }
                }
            }
        }

        stack
            .iter()
            .flat_map(|&(ch, cnt)| std::iter::repeat_n(ch, cnt))
            .collect()
    }
}

// 1717. 删除子字符串的最大得分
pub mod n1717 {
    pub fn maximum_gain(s: &str, x: i32, y: i32) -> i32 {
        let (high_val, high_first, high_second, low_val, low_first, low_second) = if x > y {
            (x, 'a', 'b', y, 'b', 'a')
        } else {
            (y, 'b', 'a', x, 'a', 'b')
        };

        // 第一轮：先删高分对子
        let mut stack1 = Vec::new();
        let mut ans = 0;
        for c in s.chars() {
            if let Some(&top) = stack1.last()
                && top == high_first
                && c == high_second
            {
                ans += high_val;
                stack1.pop();
                continue;
            }
            stack1.push(c);
        }

        // 第二轮：在剩余字符串删除低分对子
        let mut stack2 = Vec::new();
        for c in stack1 {
            if let Some(&top) = stack2.last()
                && top == low_first
                && c == low_second
            {
                ans += low_val;
                stack2.pop();
                continue;
            }
            stack2.push(c);
        }
        ans
    }
}

// 2197. 替换数组中的非互质数
pub mod n2197 {
    pub fn replace_non_coprimes(nums: Vec<i32>) -> Vec<i32> {
        fn gcd(mut a: i32, mut b: i32) -> i32 {
            while a != 0 {
                (a, b) = (b % a, a);
            }

            b
        }

        fn lcm(a: i32, b: i32) -> i32 {
            a / gcd(a, b) * b
        }

        let mut st = vec![];

        for mut x in nums {
            while !st.is_empty() && gcd(x, *st.last().unwrap()) > 1 {
                x = lcm(x, st.pop().unwrap());
            }

            st.push(x);
        }

        st
    }
}

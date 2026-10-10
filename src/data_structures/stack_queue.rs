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

// 735. 小行星碰撞
pub mod n735 {
    pub fn asteroid_collision(asteroids: Vec<i32>) -> Vec<i32> {
        let mut st = vec![];

        for x in asteroids {
            if x > 0 {
                // x 向右
                st.push(x);
                continue;
            }

            let mut ok = true;
            while let Some(&top) = st.last() {
                if top < 0 {
                    // 栈顶小行星向左
                    break;
                }

                if top <= -x {
                    // 栈顶小行星爆炸
                    st.pop();
                }

                if top >= -x {
                    // x 爆炸
                    ok = false;
                    break;
                }
            }

            if ok {
                // x 没有爆炸
                st.push(x);
            }
        }

        st
    }
}

// 2751. 机器人碰撞
pub mod n2751 {
    pub fn survived_robots_healths(
        positions: Vec<i32>,
        mut healths: Vec<i32>,
        directions: String,
    ) -> Vec<i32> {
        // 创建一个下标数组，对下标数组排序，这样不会打乱输入顺序
        let mut idx = (0..positions.len()).collect::<Vec<_>>();
        idx.sort_unstable_by_key(|&i| positions[i]);

        let directions = directions.as_bytes();
        let mut st = vec![];

        for i in idx {
            if directions[i] == b'R' {
                // 机器人 i 向右
                st.push(i);
                continue;
            }
            while let Some(&j) = st.last() {
                // 栈顶机器人向右
                if healths[j] > healths[i] {
                    // 栈顶机器人的健康度大
                    healths[i] = 0; // 移除机器人 i
                    healths[j] -= 1;
                    break;
                }
                if healths[j] == healths[i] {
                    // 健康度一样大，都移除
                    healths[i] = 0;
                    healths[j] = 0;
                    st.pop();
                    break;
                }
                // 机器人 i 的健康度大
                healths[i] -= 1;
                healths[j] = 0; // 移除机器人 j
                st.pop();
            }
        }

        // 返回幸存机器人的健康度
        healths.into_iter().filter(|&h| h > 0).collect()
    }
}

// 853. 车队
pub mod n853 {
    pub fn car_fleet(target: i32, position: Vec<i32>, speed: Vec<i32>) -> i32 {
        let mut cars: Vec<(i32, i32)> = position.into_iter().zip(speed).collect();
        // 按 position 升序排序
        cars.sort_by_key(|&(p, _)| p);

        let mut st: Vec<(i32, i32)> = Vec::new();
        for (p, v) in cars {
            while let Some(&(top_p, top_v)) = st.last() {
                // (target - top_p) * v <= (target - p) * top_v, 原来的除法移项换成乘法，避免除法不准确
                // 等价 top到达时间 <= 当前车到达时间 → 当前会被前面的车追上合并
                if (target - top_p) as i64 * v as i64 <= (target - p) as i64 * top_v as i64 {
                    st.pop();
                } else {
                    break;
                }
            }
            st.push((p, v));
        }
        st.len() as i32
    }
}

// 1776. 车队 II
pub mod n1776 {
    pub fn get_collision_times(cars: Vec<Vec<i32>>) -> Vec<f64> {
        let n = cars.len();
        let mut ans = vec![-1.0; n];
        let mut st: Vec<usize> = Vec::new(); // 栈：栈底时间大，栈顶时间小

        for i in (0..n).rev() {
            let p = cars[i][0] as f64;
            let v = cars[i][1] as f64;

            while let Some(&j) = st.last() {
                let q = cars[j][0] as f64;
                let w = cars[j][1] as f64;

                if v > w {
                    let t = (q - p) / (v - w);
                    if ans[j] < 0.0 || t <= ans[j] {
                        ans[i] = t;
                        break;
                    }
                }
                st.pop();
            }
            st.push(i);
        }
        ans
    }
}

// 20. 有效的括号
pub mod n20 {
    pub fn is_valid(s: String) -> bool {
        if !s.len().is_multiple_of(2) {
            // s 长度必须是偶数
            return false;
        }
        let mut st = vec![];
        for c in s.bytes() {
            match c {
                b'(' => st.push(b')'), // 入栈对应的右括号
                b'[' => st.push(b']'),
                b'{' => st.push(b'}'),
                _ => {
                    // c 是右括号
                    if st.pop() != Some(c) {
                        return false; // 没有左括号，或者左括号类型不对
                    }
                }
            }
        }
        st.is_empty() // 所有左括号必须匹配完毕
    }
}

// 921. 使括号有效的最少添加
pub mod n921 {
    pub fn min_add_to_make_valid(s: String) -> i32 {
        let mut left_count = 0;
        let mut right_count = 0;
        for c in s.chars() {
            if c == '(' {
                left_count += 1;
            } else {
                if left_count == 0 {
                    right_count += 1;
                } else {
                    left_count -= 1;
                }
            }
        }
        left_count + right_count
    }
}

// 1021. 删除最外层的括号
pub mod n1021 {
    pub fn remove_outer_parentheses(s: String) -> String {
        let mut res = String::new();
        let mut level = 0;
        for &b in s.as_bytes() {
            if b == b')' {
                level -= 1;
            }
            if level >= 1 {
                res.push(b as char);
            }
            if b == b'(' {
                level += 1;
            }
        }
        res
    }
}

// 1614. 括号的最大嵌套深度
pub mod n1614 {
    pub fn max_depth(s: String) -> i32 {
        let mut ans = 0;
        let mut size = 0;
        for &b in s.as_bytes() {
            if b == b'(' {
                size += 1;
                ans = ans.max(size);
            } else if b == b')' {
                size -= 1;
            }
        }
        ans
    }
}

// 1190. 反转每对括号间的子串
pub mod n1190 {
    pub fn reverse_parentheses(s: String) -> String {
        let mut stk: Vec<String> = Vec::new();
        let mut buf = String::new();
        for ch in s.chars() {
            if ch == '(' {
                // 把当前累积字符串压栈，清空临时串，准备接收括号内字符
                stk.push(buf);
                buf = String::new();
            } else if ch == ')' {
                // 反转当前括号内的字符串
                buf = buf.chars().rev().collect();
                // 弹出外层前缀，拼接
                buf = stk.pop().unwrap() + &buf;
            } else {
                // 普通字符直接追加
                buf.push(ch);
            }
        }
        buf
    }
}

// 856. 括号的分数
pub mod n856 {
    pub fn score_of_parentheses(s: String) -> i32 {
        let mut stk = vec![0];
        for c in s.chars() {
            if c == '(' {
                stk.push(0);
            } else {
                let cur = stk.pop().unwrap();
                let prev = stk.pop().unwrap();
                stk.push(prev + std::cmp::max(cur * 2, 1));
            }
        }
        stk[0]
    }
}

// 1249. 移除无效的括号
pub mod n1249 {
    pub fn min_remove_to_make_valid(s: String) -> String {
        let mut t: Vec<Option<char>> = s.chars().map(Some).collect();
        let mut stack = Vec::new();

        for (i, ch) in t.iter_mut().enumerate() {
            match ch {
                Some('(') => {
                    stack.push(i);
                }
                Some(')') => {
                    if stack.is_empty() {
                        *ch = None;
                    } else {
                        stack.pop();
                    }
                }
                _ => (),
            }
        }

        // 栈剩下多余左括号，标记删除
        for idx in stack {
            t[idx] = None;
        }

        // 取出所有 Some 拼接
        t.into_iter().flatten().collect()
    }
}

// 1963. 使字符串平衡的最小交换次数
pub mod n1963 {
    pub fn min_swaps(s: String) -> i32 {
        let mut c = 0;
        for b in s.bytes() {
            if b == b'[' || c == 0 {
                c += 1;
            } else {
                c -= 1;
            }
        }
        c / 2
    }
}

// 678. 有效的括号字符串
pub mod n678 {
    pub fn check_valid_string(s: String) -> bool {
        let mut left_stack: Vec<usize> = Vec::new();
        let mut asterisk_stack: Vec<usize> = Vec::new();

        for (i, c) in s.chars().enumerate() {
            match c {
                '(' => left_stack.push(i),
                '*' => asterisk_stack.push(i),
                ')' => {
                    if left_stack.pop().is_some() {
                        // 优先消耗左括号
                    } else if asterisk_stack.pop().is_some() {
                        // 消耗星号
                    } else {
                        return false;
                    }
                }
                _ => unreachable!(),
            }
        }

        // 剩余的左括号，用后面的*抵消
        while !left_stack.is_empty() && !asterisk_stack.is_empty() {
            let left_idx = left_stack.pop().unwrap();
            let star_idx = asterisk_stack.pop().unwrap();
            if left_idx > star_idx {
                return false;
            }
        }

        // 最后左括号栈必须空
        left_stack.is_empty()
    }
}

// 1111. 有效括号的嵌套深度
pub mod n1111 {
    pub fn max_depth_after_split(seq: String) -> Vec<i32> {
        let mut res = Vec::with_capacity(seq.len());
        let mut depth = 0;
        for c in seq.chars() {
            // 遍历到左括号，连续括号个数加 1
            if c == '(' {
                depth += 1;
                res.push(depth & 1);
            } else {
                // 遍历到右括号，与当前栈顶左括号分在一组，因此先取模，再 -1
                res.push(depth & 1);
                depth -= 1;
            }
        }
        res
    }
}

// 1541. 平衡括号字符串的最少插入次数
pub mod n1541 {
    pub fn min_insertions(s: String) -> i32 {
        let s = s.as_bytes();
        let n = s.len();
        let mut left = 0; // 未配对的左括号个数
        let mut ans = 0;
        let mut i = 0;

        while i < n {
            if s[i] == b'(' {
                left += 1; // 未配对的左括号
                i += 1;
                continue;
            }

            // 当前是右括号
            if left > 0 {
                left -= 1; // 消耗一个左括号和这组))配对
            } else {
                ans += 1; // 没有左括号，先补一个左括号
            }

            // 检查下一位是不是右括号，凑一对))
            if i < n - 1 && s[i + 1] == b')' {
                i += 2;
            } else {
                ans += 1; // 缺少第二个右括号，补一个
                i += 1;
            }
        }

        // 剩下未匹配的左括号，每个需要2个右括号
        ans + left * 2
    }
}

// 2116. 判断一个括号字符串是否有效
pub mod n2116 {
    pub fn can_be_valid(s: String, locked: String) -> bool {
        if !s.len().is_multiple_of(2) {
            return false;
        }
        let mut mn = 0;
        let mut mx = 0;
        for (b, lock) in s.bytes().zip(locked.bytes()) {
            if lock == b'1' {
                // 不能改
                let d = if b == b'(' { 1 } else { -1 };
                mx += d;
                if mx < 0 {
                    // c 不能为负
                    return false;
                }
                mn += d;
            } else {
                // 可以改
                mx += 1; // 改成左括号，c 加一
                mn -= 1; // 改成右括号，c 减一
            }
            if mn < 0 {
                // c 不能为负
                mn = 1; // 此时 c 的取值范围都是奇数，最小的正奇数是 1
            }
        }
        mn == 0 // 说明最终 c 能是 0
    }
}

// 32. 最长有效括号
pub mod n32 {
    pub fn longest_valid_parentheses(s: String) -> i32 {
        let mut st = vec![-1]; // 未配对括号的下标，其中栈底元素表示永远无法配对的括号下标
        let mut ans = 0;

        for (i, ch) in s.bytes().enumerate() {
            if ch == b'(' {
                st.push(i as i32); // 保存左括号的下标
            } else if st.len() > 1 {
                st.pop(); // 右括号与栈顶的左括号配对
                ans = ans.max(i as i32 - *st.last().unwrap()); // 从 st.last()+1 到 i 都已配对，长为 i - st.last()
            } else {
                // s[i] 是永远无法配对的右括号
                st[0] = i as i32; // 替换栈底
            }
        }

        ans
    }
}

// 1006. 笨阶乘
pub mod n1006 {
    pub fn clumsy(n: i32) -> i32 {
        let mut stack = std::collections::VecDeque::new();
        stack.push_back(n);
        let mut n = n - 1;

        let mut index = 0;
        while n > 0 {
            match index % 4 {
                0 => {
                    let top = stack.pop_back().unwrap();
                    stack.push_back(top * n);
                }
                1 => {
                    let top = stack.pop_back().unwrap();
                    stack.push_back(top / n);
                }
                2 => {
                    stack.push_back(n);
                }
                3 => {
                    stack.push_back(-n);
                }
                _ => unreachable!(),
            }
            index += 1;
            n -= 1;
        }

        stack.into_iter().sum()
    }
}

// 150. 逆波兰表达式求值
pub mod n150 {
    pub fn eval_rpn(tokens: Vec<String>) -> i32 {
        let mut stack = vec![];
        for token in tokens {
            if let Ok(val) = token.parse::<i32>() {
                stack.push(val);
            } else {
                let num2 = stack.pop().unwrap(); // 注意 num1 和 num2不能搞反，因为有除法
                let num1 = stack.pop().unwrap();
                match token.as_str() {
                    "+" => stack.push(num1 + num2),
                    "-" => stack.push(num1 - num2),
                    "*" => stack.push(num1 * num2),
                    "/" => stack.push(num1 / num2),
                    _ => unreachable!(),
                }
            }
        }
        stack[0]
    }
}

// 394. 字符串解码
pub mod n394 {
    pub fn decode_string(s: String) -> String {
        let mut stack: Vec<(u32, String)> = Vec::new();
        let mut res = String::new();
        let mut multi = 0;

        for c in s.chars() {
            match c {
                '[' => {
                    stack.push((multi, res));
                    res = String::new();
                    multi = 0;
                }
                ']' => {
                    let (cur_multi, last_res) = stack.pop().unwrap();
                    res = last_res + &res.repeat(cur_multi as usize);
                }
                '0'..='9' => {
                    multi = multi * 10 + c.to_digit(10).unwrap();
                }
                _ => {
                    res.push(c);
                }
            }
        }
        res
    }
}

// 8. 字符串转换整数 (atoi)
pub mod n8 {
    pub fn my_atoi(s: String) -> i32 {
        let s = s.as_bytes();
        let n = s.len();

        // 跳过前导空格
        let mut i = 0;
        while i < n && s[i] == b' ' {
            i += 1;
        }

        // 处理正负号
        let mut sign = 1;
        if i < n && (s[i] == b'+' || s[i] == b'-') {
            if s[i] == b'-' {
                sign = -1;
            }
            i += 1;
        }

        // 处理数字
        let mut num = 0;
        while i < n && (b'0' <= s[i] && s[i] <= b'9') {
            let d = (s[i] - b'0') as i32;
            // 如果 num*10+d > i32::MAX，则最终答案已确定，提前返回
            if num > i32::MAX / 10 || num * 10 > i32::MAX - d {
                // 避免溢出
                return if sign > 0 { i32::MAX } else { i32::MIN };
            }
            num = num * 10 + d;
            i += 1;
        }

        sign * num
    }
}

// 224. 基本计算器
pub mod n224 {
    pub fn calculate(s: String) -> i32 {
        /*
         * 创建两个栈 numStack 和 opStack:
         * numStack ： 存放所有的数字
         * opStack ：存放所有的数字以外的操作
         */
        let mut num_stack = vec![];
        let mut op_stack = vec![];

        // 这里的优先级划分按照「数学」进行划分即可
        // 使用 map 维护一个运算符优先级
        let op_weight = std::collections::HashMap::from([
            ('+', 1),
            ('-', 1),
            ('*', 2),
            ('/', 2),
            ('%', 2),
            ('^', 3),
        ]);

        // 去掉字符串的所有空格
        let s = s.replace(" ", "").chars().collect::<Vec<char>>();

        // 为了防止第一个数为负数，先往 nums 加个 0
        // 例如 -1 + 2 情况
        num_stack.push(0);

        /*
         * 遍历字符串(分情况讨论):
         * 1. 如果取到 ( , 直接加入 op_stack 中，等待与之匹配的 )
         * 2. 如果取到 ) , 使用现有的 num_stack 和 op_stack 进行计算，
         * 再弹出左括号(也就是和当前右括号相匹配的左括号)
         * 3. 如果取到一个数字, 从当前位置开始继续往后取, 将整一个连续数字整体取出, 加入 num_stack
         * 4. 如果取到运算符, 说明之前已经是一个完整的计算表达式, 如果有操作符而且这个操作符不是左括号，
         * 而且只有「栈顶运算符」比「当前运算符」优先级高/同等，才可以使用现有的 num_stack 和 op_stack 进行计算,
         * 计算结果放到 num_stack, 相到于得到了当前运算符的左操作数，最后放入当前运算符
         */
        let mut i = 0;
        while i < s.len() {
            let c = s[i];
            if c == '(' {
                op_stack.push(c);
            } else if c == ')' {
                /*
                 * 如果运算符栈顶不是 (，说明括号内有运算，直接进行计算，再弹出左括号
                 * 否则说明括号内没有运算，直接弹出左括号, 例如(1)
                 * 因为混合运算，括号前可能有多个运算符可以计算，例如(1+2*3)，所以需要使用while循环把可以计算的先算完
                 * 重复计算到最近一个左括号为止
                 */
                while !op_stack.is_empty() {
                    if *op_stack.last().unwrap() != '(' {
                        cal(&mut num_stack, &mut op_stack);
                    } else {
                        op_stack.pop();
                        break;
                    }
                }
            } else if c.is_ascii_digit() {
                // 将从 i 位置开始后面的连续数字整体取出，加入 numStack
                let mut number = 0;
                let mut j = i;
                while j < s.len() && s[j].is_ascii_digit() {
                    number = number * 10 + s[j].to_digit(10).unwrap();
                    j += 1;
                }
                num_stack.push(number as i32);
                // 此时的j位置是第一个不为数字的位置，将遍历的位置调整到j-1
                i = j - 1;
            } else {
                // 为防止 () 内出现的首个字符为运算符，将 (- 替换为 (0-，(+ 替换为 (0+
                if i > 0 && s[i - 1] == '(' {
                    num_stack.push(0);
                }

                /*
                 * 使用现有的 num_stack 和 op_stack 计算表达式
                 * 运算符栈不为空而且前一个是运算符不是括号才可以计算，例如(3+2+1)在读取到第一个+时不能计算
                 * 因为是混合运算，当一个新的运算符要入栈时，之前的表达式可能有多个运算符可以计算
                 * 需要把栈内的可以计算的都计算完，例如1*2-3/4+2在读到加号时，不仅前面的/要计算，前面的-也需要计算
                 * 所以需要使用while循环把能计算的都两两计算完
                 */
                while !op_stack.is_empty() && *op_stack.last().unwrap() != '(' {
                    // 只有满足「栈顶运算符」比「当前运算符」优先级高/同等，才进行运算
                    if op_weight.get(op_stack.last().unwrap()) >= op_weight.get(&c) {
                        cal(&mut num_stack, &mut op_stack);
                    } else {
                        break;
                    }
                }
                // 操作符入栈
                op_stack.push(c);
            }
            i += 1;
        }

        // 如果最后一个是数字，读取完之后没有进行计算
        // 此时要把剩余的计算完, 可能有多个都没有计算
        // 例如3+2*2, 所以要用while循环
        while !op_stack.is_empty() {
            cal(&mut num_stack, &mut op_stack);
        }

        // 使用现有的 num栈 和 op栈 进行表达式计算
        fn cal(num_stack: &mut Vec<i32>, op_stack: &mut Vec<char>) {
            if num_stack.len() < 2 || op_stack.is_empty() {
                return;
            }
            // 取出两个数和一个运算符进行计算
            let num2 = num_stack.pop().unwrap();
            let num1 = num_stack.pop().unwrap();

            let op = op_stack.pop().unwrap();
            match op {
                '+' => num_stack.push(num1 + num2),
                '-' => num_stack.push(num1 - num2),
                '*' => num_stack.push(num1 * num2),
                '/' => num_stack.push(num1 / num2),
                '%' => num_stack.push(num1 % num2),
                '^' => num_stack.push((num1 as f64).powf(num2 as f64) as i32),
                _ => unreachable!(),
            }
        }

        *num_stack.last().unwrap()
    }
}

// 726. 原子的数量
pub mod n726 {
    pub fn count_of_atoms(formula: String) -> String {
        let mut i = 0;
        let chars: Vec<char> = formula.chars().collect();
        let n = chars.len();

        // 解析原子符号
        fn parse_atom(chars: &[char], i: &mut usize) -> String {
            let mut atom = String::new();
            atom.push(chars[*i]);
            *i += 1;
            while *i < chars.len() && chars[*i].is_lowercase() {
                atom.push(chars[*i]);
                *i += 1;
            }
            atom
        }

        // 解析数字，无数字返回1
        fn parse_num(chars: &[char], i: &mut usize) -> i32 {
            if *i == chars.len() || !chars[*i].is_ascii_digit() {
                return 1;
            }
            let mut num = 0;
            while *i < chars.len() && chars[*i].is_ascii_digit() {
                num = num * 10 + (chars[*i].to_digit(10).unwrap() as i32);
                *i += 1;
            }
            num
        }

        let mut stack: Vec<std::collections::HashMap<String, i32>> = Vec::new();
        stack.push(std::collections::HashMap::new());

        while i < n {
            let ch = chars[i];
            match ch {
                '(' => {
                    i += 1;
                    stack.push(std::collections::HashMap::new());
                }
                ')' => {
                    i += 1;
                    let mul = parse_num(&chars, &mut i);
                    let inner = stack.pop().unwrap();
                    let top = stack.last_mut().unwrap();
                    for (atom, cnt) in inner {
                        *top.entry(atom).or_insert(0) += cnt * mul;
                    }
                }
                _ => {
                    let atom = parse_atom(&chars, &mut i);
                    let cnt = parse_num(&chars, &mut i);
                    *stack.last_mut().unwrap().entry(atom).or_insert(0) += cnt;
                }
            }
        }

        // 取出结果，排序
        let map = stack.pop().unwrap();
        let mut pairs: Vec<_> = map.into_iter().collect();
        pairs.sort_by(|a, b| a.0.cmp(&b.0));

        let mut ans = String::new();
        for (name, count) in pairs {
            ans.push_str(&name);
            if count > 1 {
                ans.push_str(&count.to_string());
            }
        }
        ans
    }
}

// 1106. 解析布尔表达式
pub mod n1106 {
    pub fn parse_bool_expr(expression: String) -> bool {
        let mut stk = Vec::new();
        for c in expression.chars() {
            if c == ',' {
                continue;
            }
            if c != ')' {
                stk.push(c);
                continue;
            }
            let mut t = 0;
            let mut f = 0;
            while *stk.last().unwrap() != '(' {
                match stk.pop().unwrap() {
                    't' => t += 1,
                    'f' => f += 1,
                    _ => unreachable!(),
                }
            }
            stk.pop(); // pop '('
            let op = stk.pop().unwrap();
            match op {
                '!' => {
                    // ! : f==1 → true(t) else f
                    stk.push(if f == 1 { 't' } else { 'f' });
                }
                '&' => {
                    // & 没有f才是t
                    stk.push(if f == 0 { 't' } else { 'f' });
                }
                '|' => {
                    // | 有t就是t
                    stk.push(if t > 0 { 't' } else { 'f' });
                }
                _ => unreachable!(),
            }
        }
        stk.last().unwrap() == &'t'
    }
}

// 591. 标签验证器
pub mod n591 {
    pub fn is_valid(s: String) -> bool {
        const CDATA1: &str = "<![CDATA[";
        const CDATA2: &str = "]]>";
        let mut stack = Vec::new();
        let s_bytes = s.as_bytes();
        let n = s_bytes.len();
        let mut i = 0;

        while i < n {
            // 匹配 CDATA 开始标记 <![CDATA[ 长度9
            if i + 8 < n && &s[i..i + 9] == CDATA1 {
                if i == 0 {
                    return false;
                }
                let mut j = i + 9;
                let mut ok = false;
                while j < n && !ok {
                    if j + 2 < n && &s[j..j + 3] == CDATA2 {
                        j += 3;
                        ok = true;
                    } else {
                        j += 1;
                    }
                }
                if !ok {
                    return false;
                }
                i = j;
            } else if s_bytes[i] == b'<' {
                if i == n - 1 {
                    return false;
                }
                let is_end = s_bytes[i + 1] == b'/';
                let p = if is_end { i + 2 } else { i + 1 };
                let mut j = p;
                // 走到 >
                while j < n && s_bytes[j] != b'>' {
                    if !s_bytes[j].is_ascii_uppercase() {
                        return false;
                    }
                    j += 1;
                }
                if j == n {
                    return false;
                }
                let len = j - p;
                if !(1..=9).contains(&len) {
                    return false;
                }
                let tag = &s[p..j];
                i = j + 1;
                if !is_end {
                    stack.push(tag.to_string());
                } else {
                    if stack.is_empty() || stack.pop().unwrap() != tag {
                        return false;
                    }
                    // 栈空但是后面还有剩余字符，非法
                    if stack.is_empty() && i < n {
                        return false;
                    }
                }
            } else {
                // 普通文本，不能出现在最开头
                if i == 0 {
                    return false;
                }
                i += 1;
            }
        }
        stack.is_empty()
    }
}

// 736. Lisp 语法解析
pub mod n736 {
    use std::collections::{HashMap, HashSet};

    /// 单遍字符扫描 + 多栈模拟
    /// 变量作用域方案：每个变量维护一个值栈，进入let压值，退出let批量pop回滚
    pub fn evaluate(expression: String) -> i32 {
        // global_var2stack: key=变量名，value=该变量多层作用域的值栈
        // 例：x在外层=2，内层let赋值为3，则 global_var2stack["x"] = [2,3]
        // 内层let结束时，pop一次，x回到2
        let mut global_var2stack: HashMap<String, Vec<i32>> = HashMap::new();

        // temp_var_set_stack：栈，每个元素是HashSet，存放当前let层新定义的全部变量
        // 遇到let的右括号`)`时，遍历集合，逐个把变量的值栈pop，撤销本层赋值
        let mut temp_var_set_stack: Vec<HashSet<String>> = Vec::new();

        // value_stack：值栈，保存数字、子表达式计算的中间结果
        let mut value_stack: Vec<i32> = Vec::new();

        // let_count：栈，每个元素代表当前let内部读到的token计数器
        // let语法：(let v1 e1 v2 e2 ... expr)
        // token是成对出现：变量,值,变量,值...，偶数token代表一个变量赋值完成
        let mut let_count: Vec<usize> = Vec::new();

        // unassigned：待赋值变量栈
        // 读到奇数位置token（变量名），先压入unassigned；读到后面偶数位置的值，再弹出变量完成绑定
        let mut unassigned: Vec<String> = Vec::new();

        // op_stack：运算符栈，存放 "let" / "add" / "mult"
        // 遇到左括号后的关键字入栈；遇到右括号弹出，执行对应逻辑
        let mut op_stack: Vec<String> = Vec::new();

        // token_map：关键字集合，区分token是运算符还是普通变量名
        let mut token_map = HashSet::new();
        token_map.insert("let");
        token_map.insert("add");
        token_map.insert("mult");

        // 将输入字符串转为字符数组，方便按索引访问
        let expr_chars: Vec<char> = expression.chars().collect();
        let mut i = 0; // 字符游标
        let n = expr_chars.len(); // 总字符长度

        // 单遍循环扫描整个表达式字符串
        while i < n {
            let c = expr_chars[i];

            // 空格、左括号直接跳过，左括号仅标记表达式开始，真正运算符在后面token
            if c == ' ' || c == '(' {
                i += 1;
                continue;
            }
            // ========== 遇到右括号：当前表达式结束，执行运算/作用域回滚 ==========
            else if c == ')' {
                // 弹出当前表达式对应的运算符
                let op = op_stack.pop().unwrap();
                if op.as_str() == "let" {
                    // 是let表达式：需要撤销本层所有变量赋值，恢复外层作用域
                    let top_set = temp_var_set_stack.pop().unwrap();
                    // 遍历本let层定义的所有变量，每个变量的值栈pop一次
                    for tmp_var in top_set {
                        if let Some(vstack) = global_var2stack.get_mut(&tmp_var) {
                            vstack.pop();
                        }
                    }
                    // 弹出本层let的token计数器
                    let_count.pop();
                } else {
                    // add / mult二元运算：从值栈取出两个数计算
                    // 栈是后进先出，后入栈的是val2，先入栈的是val1
                    let val2 = value_stack.pop().unwrap();
                    let val1 = value_stack.pop().unwrap();
                    let res = if op.as_str() == "add" {
                        val1 + val2
                    } else {
                        val1 * val2
                    };
                    // 计算结果压回值栈，作为外层表达式的一个token值
                    value_stack.push(res);
                }

                // 如果外层仍然是let表达式，这个右括号产生的结果算作let内的一个token，计数+1
                if let Some(top_op) = op_stack.last()
                    && top_op.as_str() == "let"
                {
                    *let_count.last_mut().unwrap() += 1;
                }
                i += 1;
            }
            // ========== 解析数字（包含负数，以数字字符或者负号'-'开头） ==========
            else if c.is_ascii_digit() || c == '-' {
                let mut minus = 1; // 符号标记，默认正数
                let mut num = 0;

                // 如果当前字符是负号，标记符号，游标前进一位读取数字部分
                if c == '-' {
                    minus = -1;
                    i += 1;
                }
                // 持续读取字符，直到遇到空格或右括号，拼接完整数字
                while i < n && expr_chars[i] != ' ' && expr_chars[i] != ')' {
                    num = num * 10 + (expr_chars[i].to_digit(10).unwrap() as i32);
                    i += 1;
                }
                // 应用正负号
                num *= minus;
                // 数字压入值栈
                value_stack.push(num);

                // 如果外层是let，读到数字代表读到一个token，let计数器+1
                if let Some(top_op) = op_stack.last()
                    && top_op.as_str() == "let"
                {
                    *let_count.last_mut().unwrap() += 1;
                }
            }
            // ========== 解析字母开头token：变量名 / 关键字(let/add/mult) ==========
            else {
                let mut token = String::new();
                // 持续读取字符，拼接token，直到空格或右括号终止
                while i < n && expr_chars[i] != ' ' && expr_chars[i] != ')' {
                    token.push(expr_chars[i]);
                    i += 1;
                }

                // 判断当前token是不是关键字运算符
                if token_map.contains(token.as_str()) {
                    // 如果是let，新建一层上下文：计数器置0 + 新建集合记录本层变量
                    if token.as_str() == "let" {
                        let_count.push(0);
                        temp_var_set_stack.push(HashSet::new());
                    }
                    // 运算符入栈
                    op_stack.push(token);
                } else {
                    // token不是关键字：是变量名
                    if let Some(top_op) = op_stack.last() {
                        if top_op.as_str() == "let" {
                            // 当前上下文是let表达式，token计数+1
                            *let_count.last_mut().unwrap() += 1;
                            let cnt = *let_count.last().unwrap();
                            // 判断下一个字符是不是')'，用来区分是不是let最后的返回表达式
                            let next_is_close = (i < n) && expr_chars[i] == ')';

                            // cnt奇数：变量名；并且后面不是右括号，说明后面跟着值，是待赋值变量
                            if cnt % 2 == 1 && !next_is_close {
                                unassigned.push(token.clone());
                                // 将变量加入本层集合，退出let时需要回滚
                                temp_var_set_stack.last_mut().unwrap().insert(token);
                            } else {
                                // 偶数位置，或者是let末尾返回表达式：读取变量当前栈顶值，压入值栈
                                let v_stack = global_var2stack.get(&token).unwrap();
                                value_stack.push(*v_stack.last().unwrap());
                            }
                        } else {
                            // 当前是add/mult上下文，直接读取变量值压入值栈
                            let v_stack = global_var2stack.get(&token).unwrap();
                            value_stack.push(*v_stack.last().unwrap());
                        }
                    } else {
                        // 顶层表达式，直接读取变量值， 题目的测试用例永远不会走到这里
                        let v_stack = global_var2stack.get(&token).unwrap();
                        value_stack.push(*v_stack.last().unwrap());
                    }
                }
            }

            // ========== 【每次循环末尾执行】let变量绑定逻辑 ==========
            // 当let内部token计数是>=2的偶数，代表读到了变量对应的值，执行赋值
            if let Some(top_op) = op_stack.last()
                && top_op.as_str() == "let"
            {
                let cnt = *let_count.last().unwrap();
                if cnt >= 2 && cnt.is_multiple_of(2) {
                    // 弹出待赋值变量名
                    let var = unassigned.pop().unwrap();
                    // 弹出刚刚读取到的值
                    let val = value_stack.pop().unwrap();
                    // 变量的值栈压入新值；不存在则新建Vec
                    global_var2stack.entry(var).or_default().push(val);
                }
            }
        }

        // 扫描全部字符结束，最终结果在值栈栈顶
        *value_stack.last().unwrap()
    }
}

// 1096. 花括号展开 II
pub mod n1096 {
    use std::collections::HashSet;

    pub fn brace_expansion_ii(expression: String) -> Vec<String> {
        let mut stack: Vec<(HashSet<String>, HashSet<String>)> = Vec::new();
        let mut res = HashSet::new();
        let mut cur: HashSet<String> = HashSet::from_iter(vec![String::new()]);

        for ch in expression.chars() {
            match ch {
                'a'..='z' => {
                    // 字母：每个字符串后面追加这个字符
                    let mut new_cur = HashSet::new();
                    for s in &cur {
                        let mut ns = s.clone();
                        ns.push(ch);
                        new_cur.insert(ns);
                    }
                    cur = new_cur;
                }
                ',' => {
                    // 取并集，cur并入res，cur重置为空串
                    res.extend(cur.drain());
                    cur.insert(String::new());
                }
                '{' => {
                    // 递：压栈，重置局部 res, cur
                    stack.push((res, cur));
                    res = HashSet::new();
                    cur = HashSet::from_iter(vec![String::new()]);
                }
                '}' => {
                    // 归：合并当前 res | cur，然后出栈恢复，做笛卡尔积
                    let mut sub_res = res;
                    sub_res.extend(cur);

                    let (prev_res, prev_cur) = stack.pop().unwrap();
                    res = prev_res;

                    // 笛卡尔积 prev_cur × sub_res
                    let mut new_cur = HashSet::new();
                    for s in &prev_cur {
                        for t in &sub_res {
                            let mut nt = s.clone();
                            nt.push_str(t);
                            new_cur.insert(nt);
                        }
                    }
                    cur = new_cur;
                }
                _ => unreachable!(),
            }
        }

        // 最后合并 res | cur，排序返回
        let mut total = res;
        total.extend(cur);
        let mut ans: Vec<String> = total.into_iter().collect();
        ans.sort();
        ans
    }
}

// 1896. 反转表达式值的最少操作次数
pub mod n1896 {
    /// states栈：每个元素是元组 (p, q)
    ///   p：该子表达式结果为0时，将它翻转为1的最小操作次数
    ///   q：该子表达式结果为1时，将它翻转为0的最小操作次数
    /// ops栈：保存运算符与左括号 '('、'&'、'|'
    pub fn min_operations_to_flip(expression: String) -> i32 {
        let mut states: Vec<(i32, i32)> = Vec::new();
        let mut ops: Vec<char> = Vec::new();

        for c in expression.chars() {
            if c == '0' || c == '1' || c == ')' {
                if c == '0' {
                    // 字符是0，代表子表达式值为0
                    // p=0：当前已是0，翻转成1需要代价1；q=1：值为1翻转成0代价0
                    states.push((0, 1));
                } else if c == '1' {
                    // 字符是1，代表子表达式值为1
                    // p=1：值0翻1代价0；q=0：当前已是1，翻转成0需要代价1
                    states.push((1, 0));
                } else {
                    // 遇到右括号，弹出栈顶的左括号
                    ops.pop();
                }

                // 运算符栈不为空，且栈顶不是左括号，执行二元运算合并
                if !ops.is_empty() && ops[ops.len() - 1] != '(' {
                    let op = ops.pop().unwrap();
                    let (p2, q2) = states.pop().unwrap();
                    let (p1, q1) = states.pop().unwrap();

                    if op == '&' {
                        // 与运算
                        let p = std::cmp::min(p1, p2);
                        let q = std::cmp::min(q1 + q2, 1 + std::cmp::min(q1, q2));
                        states.push((p, q));
                    } else {
                        // 或运算 |
                        let p = std::cmp::min(p1 + p2, 1 + std::cmp::min(p1, p2));
                        let q = std::cmp::min(q1, q2);
                        states.push((p, q));
                    }
                }
            } else {
                // 字符为 '('、'&'、'|'，压入运算符栈
                ops.push(c);
            }
        }

        // 取出最终表达式的状态，取p、q最大值作为答案
        let (p, q) = states.last().unwrap();
        *p.max(q)
    }
}

// 770. 基本计算器 IV
pub mod n770 {
    use std::collections::HashMap;
    /// 计算含有变量、括号、+ - * 的多项式表达式，并返回化简后的多项式项列表
    /// 规则：
    /// 1. evalvars 里的变量会被 evalints 的值替换为常数
    /// 2. * 优先级高于 +、-
    /// 3. 输出项排序：变量个数从多到少；变量数目相同时，按变量名字典序升序
    /// 4. 系数为0的项丢弃；常数项直接输出数字，变量项格式：系数*变量1*变量2
    pub fn basic_calculator_iv(
        expression: String,
        evalvars: Vec<String>,
        evalints: Vec<i32>,
    ) -> Vec<String> {
        // ========== 1. 建立变量 -> 值的映射表 ==========
        // 将evalvars和evalints一一对应，表达式中出现该变量直接替换成对应常数
        let var_map: HashMap<&str, i32> = evalvars
            .iter()
            .zip(evalints.iter())
            .map(|(var_name, val)| (var_name.as_str(), *val))
            .collect();

        // ========== 2. 分词：按空格分割表达式字符串得到token列表 ==========
        // 题目输入表达式各个符号、数字、变量之间都用空格隔开
        let tokens: Vec<&str> = expression.split_whitespace().collect();

        // ========== 3. 运算符优先级定义 ==========
        // ( 优先级最低，入栈只是标记左括号边界
        // + - 同一优先级；*优先级最高
        let pri: HashMap<&str, i32> = [("(", 0), ("+", 1), ("-", 1), ("*", 2)]
            .into_iter()
            .collect();

        // ========== 多项式类型定义 ==========
        // 多项式：HashMap，key是变量名数组（Box<[String]>），value是该项系数
        // 空数组代表【常数项】，没有变量
        type Poly = HashMap<Box<[String]>, i32>;

        // 双栈：数字栈存放多项式，符号栈存放运算符/左括号
        let mut num_stack: Vec<Poly> = Vec::new();
        let mut op_stack: Vec<&str> = Vec::new();

        // ========== 辅助函数：取出栈顶两个多项式和运算符，完成一次二元计算 ==========
        // num_stack: 多项式栈，op_stack：运算符栈
        let cal_two = |num_stack: &mut Vec<Poly>, op_stack: &mut Vec<&str>| {
            // 栈元素不足，无法计算，直接返回
            if num_stack.len() < 2 || op_stack.is_empty() {
                return;
            }
            // 弹出运算符；注意栈顺序：先弹出b，再弹出a，计算 a op b
            let op = op_stack.pop().unwrap();
            let b = num_stack.pop().unwrap();
            let a = num_stack.pop().unwrap();
            let mut res_poly = Poly::new();

            match op {
                // ========== 多项式乘法 ==========
                // (sum a_i * term_i) * (sum b_j * term_j) = sum (a_i*b_j)*(term_i*term_j)
                "*" => {
                    for (vars_a, coeff_a) in a.iter() {
                        for (vars_b, coeff_b) in b.iter() {
                            // 合并两组变量，并且排序，保证 a*b 和 b*a 是同一个key
                            let mut combined_vars: Vec<String> =
                                [vars_a.as_ref(), vars_b.as_ref()].concat();
                            combined_vars.sort();
                            // 转为Box切片，作为HashMap的key
                            let key = combined_vars.into_boxed_slice();
                            // 累加系数
                            *res_poly.entry(key).or_insert(0) += coeff_a * coeff_b;
                        }
                    }
                }
                // ========== 多项式加法 / 减法 ==========
                "+" | "-" => {
                    // 先把a全部复制到结果多项式
                    res_poly = a;
                    // 遍历b的每一项，加到/减去结果多项式
                    for (vars_b, &coeff_b) in b.iter() {
                        let entry = res_poly.entry(vars_b.clone()).or_insert(0);
                        if op == "+" {
                            *entry += coeff_b;
                        } else {
                            *entry -= coeff_b;
                        }
                    }
                }
                // 其他符号不处理
                _ => {}
            }
            // 将计算完成的多项式压回数字栈
            num_stack.push(res_poly);
        };

        // ========== 辅助函数：将单个token（数字/变量）转为多项式 ==========
        let convert = |token: &str| -> Poly {
            let mut poly = Poly::new();
            // 情况1：token是需要替换的变量，直接变成常数项
            if let Some(&val) = var_map.get(token) {
                // 空变量数组 = 常数项
                poly.insert(Vec::new().into_boxed_slice(), val);
            }
            // 情况2：token是纯数字，转为常数
            else if token.chars().all(|ch| ch.is_ascii_digit()) {
                let num: i32 = token.parse().unwrap();
                poly.insert(Vec::new().into_boxed_slice(), num);
            }
            // 情况3：普通变量，该项系数=1
            else {
                let vars = vec![token.to_string()];
                poly.insert(vars.into_boxed_slice(), 1);
            }
            poly
        };

        // ========== 遍历所有token，执行表达式求值（双栈调度） ==========
        for t in tokens {
            // Case 1：当前token是运算符 ( + - * )
            if pri.contains_key(t) {
                // 栈顶运算符优先级 >= 当前运算符，先计算栈里的
                while let Some(&top_op) = op_stack.last() {
                    if pri[top_op] >= pri[t] {
                        cal_two(&mut num_stack, &mut op_stack);
                    } else {
                        break;
                    }
                }
                // 当前运算符入符号栈
                op_stack.push(t);
            }
            // Case 2：token是【可能包裹前后括号】的数字/变量，例如 "((a))"
            else {
                // 把token转为字符数组，方便取下标，处理连续的左、右括号
                let s_chars: Vec<char> = t.chars().collect();
                let mut l = 0;
                let mut r = s_chars.len();
                let mut right_paren_cnt = 0;

                // 剥离开头连续左括号：每遇到一个 '(', 压入符号栈
                while l < r && s_chars[l] == '(' {
                    op_stack.push("(");
                    l += 1;
                }

                // 统计末尾连续右括号数量，不进栈，后面统一处理
                while l < r && s_chars[r - 1] == ')' {
                    right_paren_cnt += 1;
                    r -= 1;
                }

                // 剥离括号之后中间真正内容，转多项式压入数字栈
                let inner_token: String = s_chars[l..r].iter().collect();
                num_stack.push(convert(&inner_token));

                // 循环处理每一个右括号：一直计算直到遇到左括号
                for _ in 0..right_paren_cnt {
                    // 不断计算，直到栈顶是左括号
                    while op_stack.last() != Some(&"(") {
                        cal_two(&mut num_stack, &mut op_stack);
                    }
                    // 弹出左括号，丢弃
                    op_stack.pop();
                }
            }
        }

        // ========== 所有token处理完毕，符号栈剩余运算符全部计算 ==========
        while !op_stack.is_empty() {
            cal_two(&mut num_stack, &mut op_stack);
        }

        // ========== 提取最终多项式，过滤系数=0的项 ==========
        let final_poly = num_stack.into_iter().next().unwrap_or_default();
        let mut terms: Vec<_> = final_poly
            .into_iter()
            .filter(|(_vars, coeff)| *coeff != 0)
            .collect();

        // ========== 按题目要求排序 ==========
        // 1. 变量个数越多越靠前；
        // 2. 变量个数相同，按变量名字典序升序
        terms.sort_by(|(vars_a, _coeff_a), (vars_b, _coeff_b)| {
            if vars_a.len() != vars_b.len() {
                // 变量长度降序
                vars_b.len().cmp(&vars_a.len())
            } else {
                // 变量列表字典序升序
                vars_a.as_ref().cmp(vars_b.as_ref())
            }
        });

        // ========== 拼接成输出字符串数组 ==========
        let mut result = Vec::new();
        for (vars, coeff) in terms {
            if vars.is_empty() {
                // 常数项，直接输出数字
                result.push(coeff.to_string());
            } else {
                // 变量项：系数*var1*var2
                let var_str = vars.join("*");
                result.push(format!("{}*{}", coeff, var_str));
            }
        }

        result
    }
}

// 2296. 设计一个文本编辑器
pub mod n2296 {
    pub struct TextEditor {
        pub left: Vec<char>,
        pub right: Vec<char>,
    }

    impl Default for TextEditor {
        fn default() -> Self {
            Self::new()
        }
    }

    impl TextEditor {
        pub fn new() -> Self {
            TextEditor {
                left: Vec::new(),
                right: Vec::new(),
            }
        }

        pub fn add_text(&mut self, text: String) {
            self.left.extend(text.chars());
        }

        pub fn delete_text(&mut self, k: i32) -> i32 {
            let pre = self.left.len();
            let k = k as usize;
            let cut = self.left.len().saturating_sub(k);
            self.left.truncate(cut);
            (pre - self.left.len()) as i32
        }

        fn text(&self) -> String {
            let take = self.left.len().min(10);
            self.left[self.left.len() - take..].iter().collect()
        }

        pub fn cursor_left(&mut self, k: i32) -> String {
            let mut k = k as usize;
            while k > 0 && !self.left.is_empty() {
                let c = self.left.pop().unwrap();
                self.right.push(c);
                k -= 1;
            }
            self.text()
        }

        pub fn cursor_right(&mut self, k: i32) -> String {
            let mut k = k as usize;
            while k > 0 && !self.right.is_empty() {
                let c = self.right.pop().unwrap();
                self.left.push(c);
                k -= 1;
            }
            self.text()
        }
    }
}

// 933. 最近的请求次数
pub mod n933 {
    use std::collections::VecDeque;

    pub struct RecentCounter {
        pub q: VecDeque<i32>,
    }

    impl Default for RecentCounter {
        fn default() -> Self {
            Self::new()
        }
    }

    impl RecentCounter {
        pub fn new() -> Self {
            RecentCounter { q: VecDeque::new() }
        }

        pub fn ping(&mut self, t: i32) -> i32 {
            self.q.push_back(t);
            // 队首元素小于 t‑3000 就出队
            while self.q[0] < t - 3000 {
                self.q.pop_front();
            }
            self.q.len() as i32
        }
    }
}

// 3829. 设计共享出行系统
pub mod n3829 {
    use std::collections::{HashSet, VecDeque};

    pub struct RideSharingSystem {
        pub riders: VecDeque<i32>,
        pub drivers: VecDeque<i32>,
        pub waiting_riders: HashSet<i32>,
    }

    impl Default for RideSharingSystem {
        fn default() -> Self {
            Self::new()
        }
    }

    impl RideSharingSystem {
        pub fn new() -> Self {
            RideSharingSystem {
                riders: VecDeque::new(),
                drivers: VecDeque::new(),
                waiting_riders: HashSet::new(),
            }
        }

        pub fn add_rider(&mut self, rider_id: i32) {
            self.riders.push_back(rider_id);
            self.waiting_riders.insert(rider_id);
        }

        pub fn add_driver(&mut self, driver_id: i32) {
            self.drivers.push_back(driver_id);
        }

        pub fn match_driver_with_rider(&mut self) -> Vec<i32> {
            // 弹出队列头部已经取消的乘客
            while let Some(&front) = self.riders.front() {
                if !self.waiting_riders.contains(&front) {
                    self.riders.pop_front();
                } else {
                    break;
                }
            }

            // 无乘客 或者 无司机
            if self.riders.is_empty() || self.drivers.is_empty() {
                return vec![-1, -1];
            }

            let driver = self.drivers.pop_front().unwrap();
            let rider = self.riders.pop_front().unwrap();
            // 匹配成功，移除等待标记
            self.waiting_riders.remove(&rider);
            vec![driver, rider]
        }

        pub fn cancel_rider(&mut self, rider_id: i32) {
            // 不存在也不报错
            self.waiting_riders.remove(&rider_id);
        }
    }
}

// 950. 按递增顺序显示卡牌
pub mod n950 {
    pub fn deck_revealed_increasing(mut deck: Vec<i32>) -> Vec<i32> {
        deck.sort_unstable_by(|a, b| b.cmp(a)); // 降序
        let mut q = std::collections::VecDeque::new();
        for x in deck {
            if !q.is_empty() {
                q.rotate_right(1);
            }
            q.push_front(x);
        }
        q.into_iter().collect()
    }
}

// 649. Dota2 参议院
pub mod n649 {
    pub fn predict_party_victory(s: String) -> String {
        let mut rd = std::collections::VecDeque::new(); // Radiant R
        let mut dd = std::collections::VecDeque::new(); // Dire D
        let n = s.len();
        for (idx, ch) in s.chars().enumerate() {
            match ch {
                'R' => rd.push_back(idx),
                'D' => dd.push_back(idx),
                _ => (),
            }
        }

        while !rd.is_empty() && !dd.is_empty() {
            let a = rd.pop_front().unwrap();
            let b = dd.pop_front().unwrap();
            if a < b {
                // R先发言，D被干掉，R进入下一轮，下标+n
                rd.push_back(a + n);
            } else {
                // D先发言，R被干掉，D进入下一轮，下标+n
                dd.push_back(b + n);
            }
        }

        if !rd.is_empty() {
            "Radiant".to_string()
        } else {
            "Dire".to_string()
        }
    }
}

// 1670. 设计前中后队列
pub mod n1670 {
    use std::collections::VecDeque;

    pub struct FrontMiddleBackQueue {
        pub left: VecDeque<i32>,
        pub right: VecDeque<i32>,
    }

    impl Default for FrontMiddleBackQueue {
        fn default() -> Self {
            Self::new()
        }
    }

    impl FrontMiddleBackQueue {
        pub fn new() -> Self {
            Self {
                left: VecDeque::new(),
                right: VecDeque::new(),
            }
        }

        // 调整长度，保证 0 <= right.len() - left.len() <= 1
        // 从而保证可以在正中间插入删除元素
        pub fn balance(&mut self) {
            if self.left.len() > self.right.len() {
                self.right.push_front(self.left.pop_back().unwrap());
            } else if self.right.len() > self.left.len() + 1 {
                self.left.push_back(self.right.pop_front().unwrap());
            }
        }

        pub fn push_front(&mut self, val: i32) {
            self.left.push_front(val);
            self.balance();
        }

        pub fn push_middle(&mut self, val: i32) {
            if self.left.len() < self.right.len() {
                self.left.push_back(val);
            } else {
                self.right.push_front(val);
            }
        }

        pub fn push_back(&mut self, val: i32) {
            self.right.push_back(val);
            self.balance();
        }

        pub fn pop_front(&mut self) -> i32 {
            if self.right.is_empty() {
                // 整个队列为空
                return -1;
            }
            let val = if self.left.is_empty() {
                self.right.pop_front().unwrap()
            } else {
                self.left.pop_front().unwrap()
            };
            self.balance();
            val
        }

        pub fn pop_middle(&mut self) -> i32 {
            if self.right.is_empty() {
                // 整个队列为空
                return -1;
            }
            if self.left.len() == self.right.len() {
                return self.left.pop_back().unwrap();
            }
            self.right.pop_front().unwrap()
        }

        pub fn pop_back(&mut self) -> i32 {
            if self.right.is_empty() {
                // 整个队列为空
                return -1;
            }
            let val = self.right.pop_back().unwrap();
            self.balance();
            val
        }
    }
}

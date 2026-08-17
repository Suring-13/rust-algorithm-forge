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

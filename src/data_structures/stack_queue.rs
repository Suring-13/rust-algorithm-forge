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

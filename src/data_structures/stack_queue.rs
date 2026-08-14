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

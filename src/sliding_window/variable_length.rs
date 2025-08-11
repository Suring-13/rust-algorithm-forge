// 3. 无重复字符的最长子串
pub mod n3 {
    pub fn length_of_longest_substring(s: String) -> i32 {
        let s = s.as_bytes();
        let mut ans = 0;
        let mut left = 0;
        let mut window = [false; 128]; // 也可以用哈希集合，这里为了效率用的数组

        for (right, &c) in s.iter().enumerate() {
            let c = c as usize;

            // 如果窗口内已经包含 c，那么再加入一个 c 会导致窗口内有重复元素
            // 所以要在加入 c 之前，先移出窗口内的 c
            while window[c] {
                // 窗口内有 c
                window[s[left] as usize] = false;
                left += 1; // 缩小窗口
            }

            window[c] = true; // 加入 c

            ans = ans.max(right - left + 1); // 更新窗口长度最大值
        }

        ans as _
    }
}

// 3090. 每个字符最多出现两次的最长子字符串
pub mod n3090 {
    use std::collections::HashMap;

    pub fn maximum_length_substring(s: String) -> i32 {
        let mut ans = 0;
        let mut left = 0;
        let mut cnt = HashMap::new();
        let chars: Vec<char> = s.chars().collect();

        for (i, &c) in chars.iter().enumerate() {
            // 增加当前字符计数
            *cnt.entry(c).or_insert(0) += 1;

            // 如果当前字符出现次数超过2，移动左指针
            while *cnt.get(&c).unwrap() > 2 {
                let left_char = chars[left];
                *cnt.get_mut(&left_char).unwrap() -= 1;
                left += 1;
            }

            // 更新最大长度
            ans = ans.max(i - left + 1);
        }

        ans as i32
    }
}

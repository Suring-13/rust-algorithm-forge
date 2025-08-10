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

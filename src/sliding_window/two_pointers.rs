// 344. 反转字符串
pub mod n344 {
    pub fn reverse_string(s: &mut Vec<char>) {
        let n = s.len();
        let mut left = 0;
        let mut right = n - 1;
        while left < right {
            s.swap(left, right);
            left += 1;
            right -= 1;
        }
    }
}

// 3643. 垂直翻转子矩阵
pub mod n3643 {
    use std::mem::swap;

    pub fn reverse_submatrix(mut grid: Vec<Vec<i32>>, x: i32, y: i32, k: i32) -> Vec<Vec<i32>> {
        let (x, y, k) = (x as usize, y as usize, k as usize);
        let (mut l, mut r) = (x, x + k - 1);
        while l < r {
            // 使用split_at_mut将网格分割为两部分，避免同时获取多个可变引用
            let (left, right) = grid.split_at_mut(r);
            // left包含0..r的行，right包含r..的行，right[0]就是原来的grid[r]
            let row_l = &mut left[l];
            let row_r = &mut right[0];

            // 交换 l 行与 r 行的 [y, y+k) 列元素
            for j in y..y + k {
                swap(&mut row_l[j], &mut row_r[j]);
            }
            l += 1;
            r -= 1;
        }
        grid
    }
}

// 125. 验证回文串
pub mod n125 {
    pub fn is_palindrome(s: String) -> bool {
        let s = s.as_bytes();
        let mut i = 0;
        let mut j = s.len() - 1;
        while i < j {
            if !s[i].is_ascii_alphanumeric() {
                i += 1;
            } else if !s[j].is_ascii_alphanumeric() {
                j -= 1;
            } else if s[i].to_ascii_lowercase() == s[j].to_ascii_lowercase() {
                i += 1;
                j -= 1;
            } else {
                return false;
            }
        }
        true
    }
}

// 1750. 删除字符串两端相同字符后的最短长度
pub mod n1750 {
    pub fn minimum_length(s: String) -> i32 {
        let s_bytes = s.as_bytes(); // 转为字节数组，避免字符索引开销
        let mut left = 0;
        let mut right = s_bytes.len() - 1;

        while left < right && s_bytes[left] == s_bytes[right] {
            let c = s_bytes[left];
            // 收缩左指针：跳过所有与 c 相同的字符
            while left <= right && s_bytes[left] == c {
                left += 1;
            }
            // 收缩右指针：跳过所有与 c 相同的字符
            while right >= left && s_bytes[right] == c {
                right -= 1;
            }
        }

        // 计算剩余长度
        if left > right {
            0
        } else {
            (right - left + 1) as i32
        }
    }
}

// 2105. 给植物浇水 II
pub mod n2105 {
    pub fn minimum_refill(plants: Vec<i32>, capacity_a: i32, capacity_b: i32) -> i32 {
        let mut ans = 0;

        let mut a = capacity_a;
        let mut b = capacity_b;

        let mut i = 0;
        let mut j = plants.len() - 1;

        while i < j {
            // Alice 给植物 i 浇水
            if a < plants[i] {
                // 没有足够的水，重新灌满水罐
                ans += 1;
                a = capacity_a;
            }
            a -= plants[i];
            i += 1;

            // Bob 给植物 j 浇水
            if b < plants[j] {
                // 没有足够的水，重新灌满水罐
                ans += 1;
                b = capacity_b;
            }
            b -= plants[j];
            j -= 1;
        }

        // 如果 Alice 和 Bob 到达同一株植物，那么当前水罐中水更多的人会给这株植物浇水
        if i == j && a.max(b) < plants[i] {
            // 没有足够的水，重新灌满水罐
            ans += 1;
        }

        ans
    }
}

// 977. 有序数组的平方
pub mod n977 {
    pub fn sorted_squares(nums: Vec<i32>) -> Vec<i32> {
        let n = nums.len();
        let mut ans = vec![0; n];
        let mut i = 0;
        let mut j = n - 1;
        for p in (0..n).rev() {
            let x = nums[i] * nums[i];
            let y = nums[j] * nums[j];
            if x > y {
                ans[p] = x;
                i += 1;
            } else {
                ans[p] = y;
                j -= 1;
            }
        }
        ans
    }
}

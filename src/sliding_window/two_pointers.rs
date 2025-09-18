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

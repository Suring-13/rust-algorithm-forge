// 668. 乘法表中第k小的数
pub mod n668 {
    pub fn find_kth_number(m: i32, n: i32, k: i32) -> i32 {
        // 转换为u32避免负数/溢出（乘法表数值非负，且m/n/k均为正整数）
        let m = m as u32;
        let n = n as u32;
        let k = k as u32;

        // 检查函数：判断≤x的数字数量是否≥k
        let check = |x: u32| -> bool {
            let mut cnt = 0u32;
            // 遍历每一行（1~m）
            for i in 1..=m {
                // 该行≤x的数量：min(x//i, n)
                let row_cnt = n.min(x / i);
                cnt += row_cnt;
                // 提前终止：数量已达标，无需继续遍历
                if cnt >= k {
                    return true;
                }
            }
            cnt >= k
        };

        let mut left = 0u32;
        let mut right = m * n; // 上界为m*n

        while left < right {
            let mid = (left + right) / 2;
            if check(mid) {
                right = mid; // 满足条件，尝试更小的x
            } else {
                left = mid + 1; // 不满足，需要更大的x
            }
        }

        right as i32
    }
}

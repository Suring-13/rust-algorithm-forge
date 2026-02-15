// 3281. 范围内整数的最大得分
pub mod n3281 {
    pub fn max_possible_score(mut start: Vec<i32>, d: i32) -> i32 {
        start.sort_unstable();
        let n = start.len();

        // 二分范围
        let mut left = 0i64;
        let mut right = ((start[n - 1] as i64 + d as i64 - start[0] as i64) / (n - 1) as i64) + 1;

        // 检查能否达到 score
        let check = |score: i64| -> bool {
            let mut x = i64::MIN;
            for &s in &start {
                let s = s as i64;
                x = s.max(x + score);
                if x > s + d as i64 {
                    return false;
                }
            }
            true
        };

        while left < right {
            let mid = left + (right - left) / 2;
            if check(mid) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        left as i32 - 1
    }
}

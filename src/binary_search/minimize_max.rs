// 410. 分割数组的最大值
pub mod n410 {
    pub fn split_array(nums: Vec<i32>, k: i32) -> i32 {
        let check = |mx: i32| -> bool {
            let mut cnt = 1;
            let mut s = 0;
            for &x in &nums {
                if s + x <= mx {
                    s += x;
                    continue;
                }
                if cnt == k {
                    return true;
                }
                cnt += 1; // 新划分一段
                s = x;
            }
            false
        };

        let mut left = *nums.iter().max().unwrap();
        let mut right = nums.iter().sum::<i32>();
        while left < right {
            let mid = left + (right - left) / 2;
            if check(mid) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        right
    }
}
